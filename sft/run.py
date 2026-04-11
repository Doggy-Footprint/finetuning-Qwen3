import os
import json
import yaml
import re
import collections
import string
import subprocess
import shutil
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer

from mlx_lm import load, batch_generate
from mlx_lm.sample_utils import make_sampler

# ==========================================
# Common Helper Functions
# ==========================================
def normalize_answer(s):
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def calculate_scores(truth, sentence):
    truth_toks = normalize_answer(truth).split()
    sentence_toks = normalize_answer(sentence).split()

    if len(truth_toks) == 0 or len(sentence_toks) == 0:
        score = 1.0 if truth_toks == sentence_toks else 0.0
        return score, score, score

    common = collections.Counter(truth_toks) & collections.Counter(sentence_toks)
    num_same = sum(common.values())

    if num_same == 0: return 0.0, 0.0, 0.0
    precision = 1.0 * num_same / len(sentence_toks)
    recall = 1.0 * num_same / len(truth_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1

def get_squad2_scores(gold_answers, sentence, target_sentence):
    if not gold_answers:
        norm_target = normalize_answer(target_sentence)
        norm_sentence = normalize_answer(sentence)
        if norm_target in norm_sentence:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 0.0
            
    best_scores = (0.0, 0.0, 0.0)
    best_f1 = -1.0
    for a in gold_answers:
        p, r, f1 = calculate_scores(a, sentence)
        if f1 > best_f1:
            best_f1 = f1
            best_scores = (p, r, f1)
    return best_scores

def format_output(response):
    # [thinking], [answer], [reference] 구조를 모두 파싱하는 정규식
    pattern_with_thinking = r"\[thinking\](.*?)\[answer\](.*?)\[reference\](.*)"
    # 모델이 [thinking]을 빼먹었을 경우를 대비한 Fallback 정규식
    pattern_fallback = r"\[answer\](.*?)\[reference\](.*)"
    
    match = re.search(pattern_with_thinking, response, re.DOTALL | re.IGNORECASE)
    if match:
        # match.group(1)은 thinking 내용이므로 평가에서는 버립니다.
        answer = match.group(2).strip()
        reference = match.group(3).strip()
        return answer, reference
    
    match_fallback = re.search(pattern_fallback, response, re.DOTALL | re.IGNORECASE)
    if match_fallback:
        answer = match_fallback.group(1).strip()
        reference = match_fallback.group(2).strip()
        return answer, reference
        
    return None, None

def get_hyp_name(config):
    return f"r{config['LORA_R']}_dr{config['LORA_DROPOUT']}_lr{config['LEARNING_RATE']}_tr{config['TRAINING_DATASET_SIZE']}_ep{config['NUM_EPOCHS']}_comp{config['DATA_COMPOSITION_RATIO']}_ep{config['NUM_EPOCHS']}"

def get_paths(config):
    root_dir = config["ROOT_DIR"]
    hyp_name = get_hyp_name(config)
    
    data_dir = os.path.join(root_dir, "data")
    
    return {
        "ROOT_DIR": root_dir,
        "DATA_DIR": data_dir,
        "ADAPTER_PATH": os.path.join(root_dir, "adapters", hyp_name),
        "RESULT_PATH": os.path.join(root_dir, "results", f"{hyp_name}.json"),
        "CONFIG_FILE": os.path.join(root_dir, "configs", f"sft_config_{hyp_name}.yaml"),
        "MD_SUMMARY_PATH": os.path.join(root_dir, "results_summary.md")
    }

# ==========================================
# Data Preparation & Config
# ==========================================
def prepare_data(config, paths):
    data_dir = paths["DATA_DIR"]

    os.makedirs(data_dir, exist_ok=True)
    raw_dataset = load_dataset("squad_v2", split="train")

    answerable = raw_dataset.filter(lambda x: len(x["answers"]["text"]) > 0 and len(x["context"]) < config["MAX_SEQ_LENGTH"])
    unanswerable = raw_dataset.filter(lambda x: len(x["answers"]["text"]) == 0 and len(x["context"]) < config["MAX_SEQ_LENGTH"])

    answerable_size = int((config["TRAINING_DATASET_SIZE"] + config["TEST_DATASET_SIZE"]) * (1 - config["DATA_COMPOSITION_RATIO"]))
    unanswerable_size = config["TRAINING_DATASET_SIZE"] + config["TEST_DATASET_SIZE"] - answerable_size

    selected_answerable = answerable.shuffle(seed=config["SEED"]).select(range(answerable_size))
    selected_unanswerable = unanswerable.shuffle(seed=config["SEED"]).select(range(unanswerable_size))

    concat_dataset = concatenate_datasets([selected_answerable, selected_unanswerable]).shuffle(seed=config["SEED"])
    split_data = concat_dataset.train_test_split(test_size=config["TEST_DATASET_SIZE"] / (config["TRAINING_DATASET_SIZE"] + config["TEST_DATASET_SIZE"]), seed=config["SEED"])

    tokenizer_hf = AutoTokenizer.from_pretrained(config["MODEL_ID"], trust_remote_code=True)

    def format_to_jsonl(dataset_split, file_name):
        file_path = os.path.join(data_dir, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            for example in dataset_split:
                question = example['question']
                
                # 정답이 없는 경우 (Unanswerable)
                if len(example["answers"]["text"]) == 0:
                    ans_text = config["TARGET_SENTENCE"]
                    ref_text = 'none'
                    # 모델이 학습할 '거절 논리' 주입
                    thinking_text = f"The question asks about '{question}'. I have carefully scanned the provided context, but there is no mention or relevant information to answer this specific question. Therefore, I must refuse to answer."
                
                # 정답이 있는 경우 (Answerable)
                else:
                    ans_text = example["answers"]["text"][0]
                    ref_text = ans_text
                    # 모델이 학습할 '탐색 논리' 주입
                    thinking_text = f"The question asks about '{question}'. Scanning the context, I found the exact information needed. The relevant part is '{ref_text}'. This directly answers the question."

                messages = [
                    {"role": "system", "content": config["SYSTEM_PROMPT"].format(target=config["TARGET_SENTENCE"])},
                    {"role": "user", "content": f"[context]: {example['context']}\n[question]: {question}"},
                    # assistant의 응답에 [thinking] 섹션을 추가
                    {"role": "assistant", "content": f"[thinking]\n{thinking_text}\n[answer]\n{ans_text}\n[reference]\n{ref_text}"}
                ]
                
                json_record = {
                    "messages": messages,
                    "context": example["context"],
                    "question": question,
                    "gold_answers": example["answers"]["text"] if len(example["answers"]["text"]) > 0 else []
                }
                f.write(json.dumps(json_record, ensure_ascii=False) + "\n")

    format_to_jsonl(split_data["train"], "train.jsonl")
    format_to_jsonl(split_data["test"], "valid.jsonl")
    print(f"✅ 데이터 변환 완료. (저장위치: {data_dir})")

def prepare_hyp_params(config, paths):
    iters = config["TRAINING_DATASET_SIZE"] // (config["BATCH_SIZE"] * config["GRAD_ACCUMULATION_STEPS"])
    all_iters = iters * config["NUM_EPOCHS"]

    yaml_config = {
        "model": config["MODEL_ID"],
        "data": paths["DATA_DIR"],
        "train_type": "lora",
        "train_mode": "sft",
        "batch_size": config["BATCH_SIZE"],
        "grad_accumulation_steps": config["GRAD_ACCUMULATION_STEPS"],
        "epochs": config["NUM_EPOCHS"],
        "iters": all_iters,
        "learning_rate": config["LEARNING_RATE"],
        "num_layers": config["NUM_LAYERS"],
        "max_seq_length": config["MAX_SEQ_LENGTH"],
        "lora_parameters": {"rank": config["LORA_R"], "dropout": config["LORA_DROPOUT"], "scale": config["LORA_SCALE"]},
        "optimizer": config["OPTIMIZER"],
        "lr_schedule": {
            "name": config["LR_SCHEDULE"],
            "warmup": int(config["WARMUP_RATIO"] * all_iters),
            "warmup_init": config["WARMUP_LEARNING_RATE"],
            "arguments": [config["LEARNING_RATE"], all_iters, config["WARMUP_LEARNING_RATE"]]
        },
        "grad_checkpoint": config["GRAD_CHECKPOINT"],
        
        # Validation Loss 및 Checkpoint 저장 설정 추가
        "steps_per_report": max(1, all_iters // 20),
        "steps_per_eval": config["EVAL_EVERY_STEPS"],  # 평가 주기
        "val_batches": config["VAL_BATCHES"],          # 평가 시 사용할 배치 수
        "save_every": config["SAVE_EVERY_STEPS"],      # 체크포인트 저장 주기
        "adapter_path": paths["ADAPTER_PATH"],
    }

    os.makedirs(os.path.dirname(paths["CONFIG_FILE"]), exist_ok=True)
    with open(paths["CONFIG_FILE"], "w", encoding="utf-8") as f:
        yaml.dump(yaml_config, f)
    print(f"📝 LoRA 설정 파일 생성 완료: {paths['CONFIG_FILE']}")

# ==========================================
# Training & Evaluation
# ==========================================
def train(config):
    paths = get_paths(config)
    prepare_data(config, paths)
    prepare_hyp_params(config, paths)
    
    train_command = ["python", "-m", "mlx_lm.lora", "--train", "-c", paths["CONFIG_FILE"], "--mask-prompt"]
    
    train_loss_history_raw = dict()
    val_loss_history_raw = dict()

    print(f"\n🚀 [TRAIN] SFT 파인튜닝 시작: {get_hyp_name(config)}")
    with subprocess.Popen(train_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
        for line in process.stdout:
            if 'Calculating loss...:' in line: continue
            print(line, end='')
            
            # Train loss 추출 (정규식 업데이트)
            train_match = re.search(r'Iter\s+(\d+):\s+Train\s+loss\s+([0-9.]+)', line)
            if train_match:
                train_loss_history_raw[int(train_match.group(1))] = train_match.group(2)[:4]
                
            # Val loss 추출 (정규식 업데이트)
            val_match = re.search(r'Iter\s+(\d+):\s+Val\s+loss\s+([0-9.]+)', line)
            if val_match:
                val_loss_history_raw[int(val_match.group(1))] = val_match.group(2)[:4]
                
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, train_command)

    os.makedirs(paths["ADAPTER_PATH"], exist_ok=True)

    steps = sorted(train_loss_history_raw.keys() | val_loss_history_raw.keys())

    train_loss = ', '.join([str(train_loss_history_raw.get(step, '')) for step in steps])
    val_loss = ', '.join([str(val_loss_history_raw.get(step, '')) for step in steps])
    
    # Train과 Val Loss를 딕셔너리로 묶어서 저장
    loss_history_raw = {
        "train_loss": train_loss_history_raw,
        "val_loss": val_loss_history_raw,
    }
    loss_history = {
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    
    with open(os.path.join(paths["ADAPTER_PATH"], "loss_history.json"), "w", encoding="utf-8") as f:
        json.dump(loss_history_raw, f, indent=4)
        
    shutil.copy2(paths["CONFIG_FILE"], os.path.join(paths["ADAPTER_PATH"], "training_config.yaml"))
    print(f"✅ 학습 완료! LoRA 어댑터 및 체크포인트 저장 위치: {paths['ADAPTER_PATH']}")
    
    return loss_history

def run_evaluation(config, loss_history=None, pass_base_model=False, specific_checkpoint=None, adv_metrics=None):
    paths = get_paths(config)
    test_data_path = os.path.join(paths["DATA_DIR"], "valid.jsonl")
    
    # 특정 체크포인트(예: 0000100_adapters.safetensors)를 평가할지, 최종본을 평가할지 선택
    adapter_path = paths["ADAPTER_PATH"]
    if specific_checkpoint:
        print(f"⚠️ 특정 체크포인트를 사용합니다: {specific_checkpoint}")
        # mlx-lm은 어댑터 디렉토리를 주면 내부 safetensors를 읽으므로, 경로를 재지정
        adapter_path = os.path.join(paths["ADAPTER_PATH"], specific_checkpoint)
    
    if not os.path.exists(adapter_path) or not os.path.exists(test_data_path):
        print(f"❌ 저장된 모델({adapter_path})이나 테스트 데이터를 찾을 수 없습니다. 학습을 먼저 진행해주세요.")
        return
    
    print(f"\n🚀 [EVAL] MLX 추론 평가 시작: {get_hyp_name(config)}")

    base_model, base_tokenizer = load(config["MODEL_ID"])
    sft_model, sft_tokenizer = load(config["MODEL_ID"], adapter_path=paths["ADAPTER_PATH"])

    testing_samples = []
    with open(test_data_path, "r", encoding="utf-8") as f:
        batch = []
        for line in f:
            data = json.loads(line)
            q_type = "unanswerable" if len(data["gold_answers"]) == 0 else "answerable"
            batch.append({
                "context": data["context"], "question": data["question"],
                "type": q_type, "gold_answers": data["gold_answers"]
            })
            if len(batch) == config["INFER_BATCH_SIZE"]:
                testing_samples.append(batch)
                batch = []
        if batch:
            testing_samples.append(batch)
            
    print(f"총 {config['TEST_DATASET_SIZE']}개의 테스트 데이터에 대해 평가를 진행합니다.") 
    
    results = []
    metrics = {
        "unanswerable": {"base_p": 0.0, "base_r": 0.0, "base_f1": 0.0, "sft_p": 0.0, "sft_r": 0.0, "sft_f1": 0.0, "count": 0},
        "answerable": {"base_p": 0.0, "base_r": 0.0, "base_f1": 0.0, "sft_p": 0.0, "sft_r": 0.0, "sft_f1": 0.0, "count": 0}
    }
    
    for batch in tqdm(testing_samples, desc='Evaluating SFT trained model over base model'):
        messages = [[
            {"role": "system", "content": config["SYSTEM_PROMPT"].format(target=config["TARGET_SENTENCE"])},
            {"role": "user", "content": f"[context]: {chat['context']}\n[question]: {chat['question']}"}
        ] for chat in batch]

        prompts = [base_tokenizer.apply_chat_template(msg, tokenize=True, add_generation_prompt=True, enable_thinking=False) for msg in messages]
        sampler = make_sampler(temp=0.2)

        base_res_texts = [config.get('TARGET_SENTENCE')]*len(batch) if pass_base_model else batch_generate(base_model, base_tokenizer, prompts, max_tokens=200, sampler=sampler).texts
        sft_responses = batch_generate(sft_model, sft_tokenizer, prompts, max_tokens=200, sampler=sampler)

        for base_res, sft_res, data in zip(base_res_texts, sft_responses.texts, batch):
            ans_base, _ = format_output(base_res)
            ans_sft, _ = format_output(sft_res)

            p_base, r_base, f1_base = get_squad2_scores(data["gold_answers"], ans_base if ans_base else base_res, config["TARGET_SENTENCE"])
            p_sft, r_sft, f1_sft = get_squad2_scores(data["gold_answers"], ans_sft if ans_sft else sft_res, config["TARGET_SENTENCE"])

            q_type = data['type']
            metrics[q_type]["base_p"] += p_base
            metrics[q_type]["base_r"] += r_base
            metrics[q_type]["base_f1"] += f1_base
            metrics[q_type]["sft_p"] += p_sft
            metrics[q_type]["sft_r"] += r_sft
            metrics[q_type]["sft_f1"] += f1_sft
            metrics[q_type]["count"] += 1

            if pass_base_model:
                p_base, r_base, f1_base = "N/A", "N/A", "N/A"

            results.append({
                "type": q_type, "question": data["question"],
                "gold_answers": data["gold_answers"] if data["gold_answers"] else [config["TARGET_SENTENCE"]],
                "base_prediction": base_res, "sft_prediction": sft_res,
                "base_scores": {"precision": p_base, "recall": r_base, "f1": f1_base},
                "sft_scores": {"precision": p_sft, "recall": r_sft, "f1": f1_sft}
            })
            
    for q_type in ["unanswerable", "answerable"]:
        count = metrics[q_type]["count"]
        if count > 0:
            for key in ["base_p", "base_r", "base_f1", "sft_p", "sft_r", "sft_f1"]:
                metrics[q_type][f"avg_{key}"] = (metrics[q_type][key] / count) * 100

    os.makedirs(os.path.dirname(paths["RESULT_PATH"]), exist_ok=True)
    if not loss_history and os.path.exists(os.path.join(paths["ADAPTER_PATH"], "loss_history.json")):
        with open(os.path.join(paths["ADAPTER_PATH"], "loss_history.json"), "r") as f:
            loss_history = json.load(f)

    final_output = {
        "config": config,
        "loss_history": loss_history,
        "summary_metrics": metrics,
        "detailed_results": results
    }
    with open(paths["RESULT_PATH"], "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)

    update_summary_markdown(config, metrics, loss_history, paths, adv_metrics)
    print(f"✅ 전체 결과가 {paths['RESULT_PATH']} 및 Markdown 리포트에 저장되었습니다.")

def update_summary_markdown(config, metrics, loss_history, paths, adv_metrics=None):
    md_path = paths["MD_SUMMARY_PATH"]
    hyp_name = get_hyp_name(config)
    
    is_new_file = not os.path.exists(md_path)

    # train_loss_history = ",".join([f"{L:.3f}" for L in loss_history.get('train_loss')]) if loss_history.get('train_loss') else "N/A"
    # val_loss_history = ",".join([f"{L:.3f}" for L in loss_history.get('val_loss')]) if loss_history.get('val_loss') else "N/A"
    train_loss_history = loss_history.get('train_loss', "N/A")
    val_loss_history = loss_history.get('val_loss', "N/A")
    
    u_base = f"{metrics['unanswerable'].get('avg_base_p', 0):.1f}/{metrics['unanswerable'].get('avg_base_r', 0):.1f}/{metrics['unanswerable'].get('avg_base_f1', 0):.1f}"
    u_sft = f"{metrics['unanswerable'].get('avg_sft_p', 0):.1f}/{metrics['unanswerable'].get('avg_sft_r', 0):.1f}/{metrics['unanswerable'].get('avg_sft_f1', 0):.1f}"
    
    a_base = f"{metrics['answerable'].get('avg_base_p', 0):.1f}/{metrics['answerable'].get('avg_base_r', 0):.1f}/{metrics['answerable'].get('avg_base_f1', 0):.1f}"
    a_sft = f"{metrics['answerable'].get('avg_sft_p', 0):.1f}/{metrics['answerable'].get('avg_sft_r', 0):.1f}/{metrics['answerable'].get('avg_sft_f1', 0):.1f}"

    adv_str = "N/A | N/A | N/A | N/A"
    if adv_metrics:
        m_base = f"{(adv_metrics['Masking']['base_success'] / max(adv_metrics['Masking']['count'], 1) * 100):.1f}%"
        m_sft = f"{(adv_metrics['Masking']['sft_success'] / max(adv_metrics['Masking']['count'], 1) * 100):.1f}%"
        d_base = f"{(adv_metrics['Distractor']['base_success'] / max(adv_metrics['Distractor']['count'], 1) * 100):.1f}%"
        d_sft = f"{(adv_metrics['Distractor']['sft_success'] / max(adv_metrics['Distractor']['count'], 1) * 100):.1f}%"
        adv_str = f"{m_base} | **{m_sft}** | {d_base} | **{d_sft}**"

    with open(md_path, "a", encoding="utf-8") as f:
        if is_new_file:
            setup_markdown(f)
        # f.write(f"| `{hyp_name}` | {config['TITLE']} | {u_base} | **{u_sft}** | {a_base} | **{a_sft}** | {train_loss_history} | {val_loss_history} |\n")
        f.write(f"| `{hyp_name}` | {config['TITLE']} | {u_base} | **{u_sft}** | {a_base} | **{a_sft}** | {adv_str} | {train_loss_history} | {val_loss_history} |\n")

def setup_markdown(md_file):
    md_file.write("# SFT Experiment Summary\n\n")
    md_file.write("| Experiment Name | Title | Unanswerable (Base) | Unanswerable (SFT) | Answerable (Base) | Answerable (SFT) | Masking Def (Base) | Masking Def (SFT) | Distractor Def (Base) | Distractor Def (SFT) | Train Loss History | Val Loss History |\n")
    md_file.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")        

# ==========================================
# MMLU Benchmark Evaluation
# ==========================================
def run_mmlu_evaluation(config, default_model=False):
    """
    lm-evaluation-harness를 사용하여 MMLU 벤치마크를 수행합니다.
    (https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md 참조)
    """
    paths = get_paths(config)
    adapter_path = paths["ADAPTER_PATH"]
    
    if not default_model and not os.path.exists(adapter_path):
        print("❌ 저장된 LoRA 어댑터를 찾을 수 없습니다. 학습을 먼저 진행해주세요.")
        return

    # 저장 경로를 구분하기 위한 실험명 추출
    hyp_name = "default_base_model" if default_model else get_hyp_name(config)
    print(f"\n🧠 [MMLU EVAL] 일반화 성능 평가 시작: {hyp_name}")
    print("이 작업은 모델의 망각(Catastrophic Forgetting) 여부를 확인하기 위해 실행되며, 다소 시간이 소요될 수 있습니다.")
    
    # 모델 인자 구성: mlx_lm 백엔드에서는 peft 대신 adapter_path 매개변수를 사용합니다.
    model_args = f"pretrained={config['MODEL_ID']}"
    if not default_model:
        model_args += f",peft={adapter_path}"
        
    # 평가 결과를 저장할 출력 디렉토리 설정
    output_dir = os.path.join(paths["ROOT_DIR"], "results", "mmlu", hyp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # lm-evaluation-harness 명령어 구성
    mmlu_command = [
        "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", "mmlu",            # 서브태스크만 원할 경우 "mmlu_abstract_algebra,mmlu_anatomy" 형태로 지정
        "--num_fewshot", "0",
        "--batch_size", "auto",       # 메모리에 맞게 동적으로 배치 사이즈 자동 튜닝
        "--output_path", output_dir,  # 결과 파일 저장 경로 지정
        "--log_samples"               # 각 샘플에 대한 예측 결과(로그) 상세 기록
        # "--limit", "10"             # 디버깅 용도: 태스크당 10개의 문제만 빠르게 평가할 경우 주석 해제
        # "--apply_chat_template"     # 지시어(Instruction) 튜닝이 강하게 된 모델의 경우 주석 해제 고려
    ]
    
    print(f"▶ 실행 명령어: {' '.join(mmlu_command)}")
    
    try:
        # 실시간 로그 확인을 위해 subprocess.run 사용
        subprocess.run(mmlu_command, check=True)
        print(f"\n✅ MMLU 평가 완료. 상세 결과와 메트릭은 {output_dir} 에 저장되었습니다.")
    except FileNotFoundError:
        print("\n❌ `lm_eval` 명령어를 찾을 수 없습니다. 터미널에서 다음 명령어로 패키지가 설치되어 있는지 확인해주세요:")
        print("pip install lm-eval") # MLX 전용 플러그인이 필요한 경우 pip install lm-eval[mlx]
    except subprocess.CalledProcessError as e:
        print(f"\n❌ MMLU 평가 중 오류가 발생했습니다: {e}")

# ==========================================
# Adversarial Testing
# ==========================================
def create_adversarial_batch(batch):
    adv_batch = []
    for data in batch:
        q_type = data["type"]
        context = data["context"]
        question = data["question"]
        gold_answers = data["gold_answers"]

        if q_type == "answerable" and len(gold_answers) > 0:
            # 전략 1: Answer Masking (정답 지우기)
            # 정답을 [REDACTED]로 치환하여 대답 불가능한 상태로 만듦
            for ans in gold_answers:
                if ans in context:
                    adv_context = context.replace(ans, "[REDACTED]")
                    adv_batch.append({
                        "adv_type": "Masking",
                        "context": adv_context,
                        "question": question,
                        "type": "unanswerable", # 정답이 사라졌으므로 unanswerable 처리
                        "gold_answers": [],
                        "original_answer": ans
                    })
        elif q_type == "unanswerable":
            # 전략 2: Distractor Injection (함정 문장 추가)
            # 질문과 단어는 겹치지만 실제 정답은 없는 문장을 문맥 끝에 추가
            distractor = f" People often ask about '{question.replace('?', '')}', but no factual information is provided in the official records."
            adv_batch.append({
                "adv_type": "Distractor",
                "context": context + distractor,
                "question": question,
                "type": "unanswerable", # 여전히 unanswerable 이어야 함
                "gold_answers": []
            })
    return adv_batch

def run_adversarial_evaluation(config):
    paths = get_paths(config)
    test_data_path = os.path.join(paths["DATA_DIR"], "valid.jsonl")
    
    if not os.path.exists(paths["ADAPTER_PATH"]):
        print("❌ 저장된 LoRA 모델을 찾을 수 없습니다.")
        return
    
    print(f"\n🕵️‍♂️ [ADVERSARIAL EVAL] 적대적 평가 시작: {get_hyp_name(config)}")

    base_model, base_tokenizer = load(config["MODEL_ID"])
    sft_model, sft_tokenizer = load(config["MODEL_ID"], adapter_path=paths["ADAPTER_PATH"])

    # 데이터 로드 (기존 평가와 동일)
    testing_samples = []
    with open(test_data_path, "r", encoding="utf-8") as f:
        batch = []
        for line in f:
            data = json.loads(line)
            q_type = "unanswerable" if len(data["gold_answers"]) == 0 else "answerable"
            batch.append({
                "context": data["context"], "question": data["question"],
                "type": q_type, "gold_answers": data["gold_answers"]
            })
            if len(batch) == config["INFER_BATCH_SIZE"]:
                testing_samples.append(batch)
                batch = []
        if batch:
            testing_samples.append(batch)

    adv_metrics = {
        "Masking": {"base_fail": 0, "sft_fail": 0, "base_success": 0, "sft_success": 0, "count": 0},
        "Distractor": {"base_fail": 0, "sft_fail": 0, "base_success": 0, "sft_success": 0, "count": 0}
    }
    
    for batch in tqdm(testing_samples, desc='Running Adversarial Attacks'):
        adv_batch = create_adversarial_batch(batch)
        if not adv_batch:
            continue

        messages = [[
            {"role": "system", "content": config["SYSTEM_PROMPT"].format(target=config["TARGET_SENTENCE"])},
            {"role": "user", "content": f"[context]: {chat['context']}\n[question]: {chat['question']}"}
        ] for chat in adv_batch]

        prompts = [base_tokenizer.apply_chat_template(msg, tokenize=True, add_generation_prompt=True, enable_thinking=False) for msg in messages]
        sampler = make_sampler(temp=0.2)

        base_responses = batch_generate(base_model, base_tokenizer, prompts, max_tokens=200, sampler=sampler)
        sft_responses = batch_generate(sft_model, sft_tokenizer, prompts, max_tokens=200, sampler=sampler)

        for base_res, sft_res, data in zip(base_responses.texts, sft_responses.texts, adv_batch):
            ans_base, _ = format_output(base_res)
            ans_sft, _ = format_output(sft_res)
            
            ans_base = ans_base if ans_base else base_res
            ans_sft = ans_sft if ans_sft else sft_res

            target = normalize_answer(config["TARGET_SENTENCE"])
            
            # 성공(Success) 조건: 모델이 방어에 성공하여 "TARGET_SENTENCE(응답 불가)"를 반환함
            base_success = 1 if target in normalize_answer(ans_base) else 0
            sft_success = 1 if target in normalize_answer(ans_sft) else 0

            adv_type = data["adv_type"]
            adv_metrics[adv_type]["base_success"] += base_success
            adv_metrics[adv_type]["base_fail"] += (1 - base_success)
            adv_metrics[adv_type]["sft_success"] += sft_success
            adv_metrics[adv_type]["sft_fail"] += (1 - sft_success)
            adv_metrics[adv_type]["count"] += 1

    print("\n" + "="*50)
    print(f"🛡️ 적대적 공격 방어율 (가짜 정보/함정에 속지 않고 '모른다'고 한 비율)")
    print("="*50)
    for adv_type, metrics in adv_metrics.items():
        if metrics["count"] > 0:
            base_def_rate = (metrics["base_success"] / metrics["count"]) * 100
            sft_def_rate = (metrics["sft_success"] / metrics["count"]) * 100
            print(f"[{adv_type} 공격] 총 {metrics['count']}건")
            print(f" - Base 모델 방어율: {base_def_rate:.1f}%")
            print(f" - SFT  모델 방어율: {sft_def_rate:.1f}%")
    print("="*50)

    return adv_metrics

# ==========================================
# Main Execution Entry Point
# ==========================================
def get_base_config():
    return {
        "SEED": 2374,
        "ROOT_DIR": os.path.dirname(os.path.abspath(__file__)),
        "MODEL_ID": "Qwen/Qwen3-0.6B",
        
        # PROMPT
        "SYSTEM_PROMPT": """You are a meticulous reading comprehension AI. Your task is to answer the [question] based STRICTLY on the provided [context].
        Always format your output EXACTLY as follows:
        [thinking]
        Write your step-by-step reasoning here. Identify what the question is asking, and explicitly state whether the relevant information exists in the context.
        [answer]
        Provide the exact answer here. If the information is missing, write exactly "{target}".
        [reference]
        Provide the exact wording from the context that supports your answer. If missing, write "none".""",
        "TARGET_SENTENCE": "Insufficient information.",        
        # FIXED TRAINING PARAMS
        "BATCH_SIZE": 4,
        "GRAD_ACCUMULATION_STEPS": 4,
        "NUM_LAYERS": -1, # all layers
        "MAX_SEQ_LENGTH": 800,
        "OPTIMIZER": "adamw",
        "LR_SCHEDULE": "cosine_decay",
        "WARMUP_RATIO": 0.1,
        "WARMUP_LEARNING_RATE": 0,
        "GRAD_CHECKPOINT": True,
        "INFER_BATCH_SIZE": 32,
        "LORA_R": 16,
        "LORA_DROPOUT": 0.05,
        "LORA_SCALE": 2.0,
        
        # 🌟 새로 추가된 Validation & Checkpoint 관련 파라미터 🌟 # TODO
        "EVAL_EVERY_STEPS": 50,    # N 스텝마다 Validation Loss 평가
        "VAL_BATCHES": 20,         # 평가 시 20 배치를 사용 (약 80개의 검증 데이터 기준)
        "SAVE_EVERY_STEPS": 40,   # 100 스텝마다 체크포인트 저장 (overfitting 모니터링 용도)
    }

# 🌟🌟 여러 실험 케이스 등록 🌟🌟
EXPERIMENT_CASES = [
    {
        **get_base_config(),
        "LEARNING_RATE": 1e-6,
        "TRAINING_DATASET_SIZE": 2000,
        "TEST_DATASET_SIZE": 2000,
        "DATA_COMPOSITION_RATIO": 0.35,
        "NUM_EPOCHS": 1,
        "TITLE": "prompt 변경 후 학습률 별 추이 확인",
    },
    # {
    #     **get_base_config(),
    #     "LEARNING_RATE": 2e-6,
    #     "TRAINING_DATASET_SIZE": 2000,
    #     "TEST_DATASET_SIZE": 2000,
    #     "DATA_COMPOSITION_RATIO": 0.35,
    #     "NUM_EPOCHS": 1,
    #     "TITLE": "prompt 변경 후 학습률 별 추이 확인(2)",
    # },
    # {
    #     **get_base_config(),
    #     "LEARNING_RATE": 3e-6,
    #     "TRAINING_DATASET_SIZE": 2000,
    #     "TEST_DATASET_SIZE": 2000,
    #     "DATA_COMPOSITION_RATIO": 0.35,
    #     "NUM_EPOCHS": 1,
    #     "TITLE": "prompt 변경 후 학습률 별 추이 확인(3)",
    # },
    # {
    #     **get_base_config(),
    #     "LEARNING_RATE": 4e-6,
    #     "TRAINING_DATASET_SIZE": 2000,
    #     "TEST_DATASET_SIZE": 2000,
    #     "DATA_COMPOSITION_RATIO": 0.35,
    #     "NUM_EPOCHS": 1,
    #     "TITLE": "prompt 변경 후 학습률 별 추이 확인(4)",
    # },
    # {
    #     **get_base_config(),
    #     "LEARNING_RATE": 5e-6,
    #     "TRAINING_DATASET_SIZE": 2000,
    #     "TEST_DATASET_SIZE": 2000,
    #     "DATA_COMPOSITION_RATIO": 0.35,
    #     "NUM_EPOCHS": 1,
    #     "TITLE": "prompt 변경 후 학습률 별 추이 확인(5)",
    # },
    # {
    #     **get_base_config(),
    #     "LEARNING_RATE": 6e-6,
    #     "TRAINING_DATASET_SIZE": 2000,
    #     "TEST_DATASET_SIZE": 2000,
    #     "DATA_COMPOSITION_RATIO": 0.35,
    #     "NUM_EPOCHS": 1,
    #     "TITLE": "prompt 변경 후 학습률 별 추이 확인(6)",
    # },
]

def main():
    print("="*50)
    print("🍎 SQuAD 데이터 기반 환각 제어 SFT (다중 케이스 지원)")
    print("="*50)
    print(f"등록된 실험 케이스: {len(EXPERIMENT_CASES)}개")
    print("1: 전체 케이스 순차 학습 및 평가 진행 (Train + Eval)")
    print("2: 특정 케이스 번호만 실행")
    print("3: 적대적 평가 실행")
    print("4: 🧠 MMLU 벤치마크 평가 (lm-eval 필요)")
    print("5: 🚀 전체 케이스 학습, 평가 및 적대적 평가 동시 진행 (Train + Eval + Adv -> MD 저장)")
    print("0: 종료")
    
    while True:
        choice = input("\n원하는 작업 번호를 입력하세요: ").strip()
        
        if choice == "1":
            paths = get_paths(EXPERIMENT_CASES[0])
            purpose = input("purpose of this run: ")
            # with open(paths["MD_SUMMARY_PATH"], "a", encoding="utf-8") as f:
            #     f.write(f"|{purpose}|---|---|---|---|---|---|---|---|---|---|---|\n")
            for idx, config in enumerate(EXPERIMENT_CASES):
                print(f"\n[{idx+1}/{len(EXPERIMENT_CASES)}] 실험 시작: {get_hyp_name(config)}")
                loss_history = train(config)
                run_evaluation(config, loss_history, pass_base_model=True)
            break
            
        elif choice == "2":
            case_idx = input(f"실행할 케이스 번호를 입력하세요 (1 ~ {len(EXPERIMENT_CASES)}): ").strip()
            if case_idx.isdigit() and 1 <= int(case_idx) <= len(EXPERIMENT_CASES):
                config = EXPERIMENT_CASES[int(case_idx)-1]
                loss_history = train(config)
                run_evaluation(config, loss_history)
            else:
                print("잘못된 케이스 번호입니다.")
            break
        
        elif choice == "3":
            case_idx = input(f"적대적 평가를 실행할 케이스 번호를 입력하세요 (1 ~ {len(EXPERIMENT_CASES)}): ").strip()
            if case_idx.isdigit() and 1 <= int(case_idx) <= len(EXPERIMENT_CASES):
                config = EXPERIMENT_CASES[int(case_idx)-1]
                run_adversarial_evaluation(config)
            else:
                print("잘못된 케이스 번호입니다.")
            break
            
        elif choice == "4":
            case_idx = input(f"MMLU를 실행할 케이스 번호를 입력하세요 (-1 for default model) (1 ~ {len(EXPERIMENT_CASES)}): ").strip()
            if case_idx.isdigit() and 1 <= int(case_idx) <= len(EXPERIMENT_CASES):
                config = EXPERIMENT_CASES[int(case_idx)-1]
                run_mmlu_evaluation(config)
            elif case_idx == "-1":
                config = EXPERIMENT_CASES[0]
                run_mmlu_evaluation(config, default_model=True)
            else:
                print("잘못된 케이스 번호입니다.")
            break
        elif choice == "5":
            paths = get_paths(EXPERIMENT_CASES[0])
            purpose = input("purpose of this run (Train+Eval+Adv): ")
            # with open(paths["MD_SUMMARY_PATH"], "a", encoding="utf-8") as f:
            #     f.write(f"|**{purpose}**|---|---|---|---|---|---|---|---|---|---|---|\n")
            for idx, config in enumerate(EXPERIMENT_CASES):
                print(f"\n[{idx+1}/{len(EXPERIMENT_CASES)}] 🚀 학습 시작: {get_hyp_name(config)}")
                loss_history = train(config)
                print(f"\n[{idx+1}/{len(EXPERIMENT_CASES)}] 🛡️ 적대적 평가 진행 중...")
                adv_metrics = run_adversarial_evaluation(config)
                print(f"\n[{idx+1}/{len(EXPERIMENT_CASES)}] 📊 최종 평가 및 리포트 작성 중...")
                run_evaluation(config, loss_history, pass_base_model=True, adv_metrics=adv_metrics)
            break
            
        elif choice == "0":
            print("프로그램을 종료합니다.")
            break
        else:
            print("⚠️ 잘못된 입력입니다.")

if __name__ == "__main__":
    main()