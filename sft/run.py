import os
import json
import yaml
import re
import collections
import string
import subprocess
import shutil
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset, concatenate_datasets

from transformers import AutoTokenizer

from mlx_lm import load, batch_generate
from mlx_lm.sample_utils import make_sampler

# ==========================================
# Global Constans
# ==========================================
SEED = 2374
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "Qwen/Qwen3-0.6B"

# ==========================================
# Training Hyper Parameters
# ==========================================
LORA_R = 32
LORA_DROPOUT = 0.05
LORA_SCALE = 2.0
LEARNING_RATE = 2e-5

TRAINING_DATASET_SIZE =1500 # 🌟🌟🌟🌟🌟
TEST_DATASET_SIZE = 4000 # 🌟🌟🌟🌟🌟
DATA_COMPOSITION_RATIO = 0.33 # unanswerable / total

BATCH_SIZE = 4
GRAD_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 1

NUM_LAYERS = -1 # all layers
MAX_SEQ_LENGTH = 800

OPTIMIZER = "adamw"
LR_SCHEDULE = "cosine_decay"
WARMUP_RATIO = 0.1
WARMUP_LEARNING_RATE = 0
GRAD_CHECKPOINT = True

def get_hyp():
    return f"""r{LORA_R};dr{LORA_DROPOUT};lr{str(LEARNING_RATE)};tr{TRAINING_DATASET_SIZE};te{TEST_DATASET_SIZE};bs{BATCH_SIZE*GRAD_ACCUMULATION_STEPS};ep{NUM_EPOCHS};opt{OPTIMIZER};layers{NUM_LAYERS}"""

# MLX params
ITERS = (TRAINING_DATASET_SIZE // (BATCH_SIZE * GRAD_ACCUMULATION_STEPS))
ALL_ITERS = ITERS * NUM_EPOCHS
DATA_DIR = f"{ROOT_DIR}/data"
ADAPTER_PATH = f"{ROOT_DIR}/adapters/{get_hyp()}"
LOSS_FILE = f"{ADAPTER_PATH}/loss_history.json"
RESULT_PATH = f"{ROOT_DIR}/results/{get_hyp()}.json"
CONFIG_FILE = f"{ROOT_DIR}/sft_config.yaml"

# Inferencing batch
INFER_BATCH_SIZE = 30

# PROMPT
SYSTEM_PROMPT = """Use the provided [context] to answer the [question] as [answer], and write the part used from the [context] as [reference]. If you cannot answer the [question] based on the provided [context], answer "{target}" for [answer], and leave [reference] empty."""
TARGET_SENTENCE = "I cannot answer this question based on the provided context."

# Common helper funcs
def normalize_answer(s):
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def calculate_f1(truth, sentence):
    truth_toks = normalize_answer(truth).split()
    sentence_toks = normalize_answer(sentence).split()

    if len(truth_toks) == 0 or len(sentence_toks) == 0:
        return int(truth_toks == sentence_toks)

    common = collections.Counter(truth_toks) & collections.Counter(sentence_toks)
    num_same = sum(common.values())

    if num_same == 0: return 0.0
    precision = 1.0 * num_same / len(sentence_toks)
    recall = 1.0 * num_same / len(truth_toks)
    return (2 * precision * recall) / (precision + recall)

def get_squad2_f1_score(gold_answers, sentence):
    if not gold_answers: 
        return calculate_f1(TARGET_SENTENCE, sentence) # SQuAD2 unanswerable case
    return max(calculate_f1(a, sentence) for a in gold_answers)

def format_output(response):
    pattern = r"\[answer\](.*?)\[reference\](.*)"
    
    match = re.search(pattern, response, re.DOTALL)
    if match:
        answer = match.group(1).strip()
        reference = match.group(2).strip()
        return answer, reference
    else:
        return None, None

# ==========================================
# 데이터 전처리 및 MLX 학습 (SFT) 로직
# ==========================================
def prepare_data():
    os.makedirs(DATA_DIR, exist_ok=True)

    raw_dataset = load_dataset("squad_v2", split="train")
    # TODO: filter too long data
    answerable = raw_dataset.filter(lambda x: len(x["answers"]["text"]) > 0)
    unanswerable = raw_dataset.filter(lambda x: len(x["answers"]["text"]) == 0)

    answerable_size = int((TRAINING_DATASET_SIZE + TEST_DATASET_SIZE) * (1 - DATA_COMPOSITION_RATIO))
    unanswerable_size = TRAINING_DATASET_SIZE + TEST_DATASET_SIZE - answerable_size

    selected_answerable = answerable.shuffle(seed=SEED).select(range(answerable_size))
    selected_unanswerable = unanswerable.shuffle(seed=SEED).select(range(unanswerable_size))

    concat_dataset = concatenate_datasets([selected_answerable, selected_unanswerable]).shuffle(seed=SEED)
    split_data = concat_dataset.train_test_split(test_size=TEST_DATASET_SIZE / (TRAINING_DATASET_SIZE + TEST_DATASET_SIZE), seed=SEED)

    tokenizer_hf = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    def format_to_jsonl(dataset_split, file_name):
        file_path = os.path.join(DATA_DIR, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            for example in dataset_split:
                if len(example["answers"]["text"]) == 0:
                    ans_text = TARGET_SENTENCE
                    ref_text = ''
                else:
                    ans_text = example["answers"]["text"][0]
                    ref_text = ans_text

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT.format(target=TARGET_SENTENCE)},
                    {"role": "user", "content": f"[context]: {example['context']}\n[question]: {example['question']}"},
                    {"role": "assistant", "content": f"[answer]: {ans_text}\n[reference]: {ref_text}"}
                ]
                
                # Chat template 적용
                formatted_text = tokenizer_hf.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
                )
                
                # 원본 메타데이터도 평가를 위해 함께 저장
                json_record = {
                    "text": formatted_text,
                    "context": example["context"],
                    "question": example["question"],
                    "gold_answers": example["answers"]["text"] if len(example["answers"]["text"]) > 0 else []
                }
                f.write(json.dumps(json_record, ensure_ascii=False) + "\n")

    format_to_jsonl(split_data["train"], "train.jsonl")
    format_to_jsonl(split_data["test"], "valid.jsonl")
    print(f"✅ 데이터 변환 완료. (저장위치: {DATA_DIR})")

def prepare_hyp_params():
    config = {
        # model, data & training
        "model": MODEL_ID,
        "data": DATA_DIR,
        "train_type": "lora",
        "train_mode": "sft",
        # training schedule
        "batch_size": BATCH_SIZE,
        "grad_accumulation_steps": GRAD_ACCUMULATION_STEPS,
        "epochs": NUM_EPOCHS,
        "iters": ALL_ITERS,
        "learning_rate": LEARNING_RATE,
        # model architecture
        "num_layers": NUM_LAYERS,
        "max_seq_length": MAX_SEQ_LENGTH,
        # lora params
        "lora_parameters": {"rank": LORA_R, "dropout": LORA_DROPOUT, "scale": LORA_SCALE},
        # optimizer
        "optimizer": OPTIMIZER,
        "lr_schedule": {
            "name": LR_SCHEDULE,
            "warmup": int(WARMUP_RATIO * ALL_ITERS),
            "warmup_init": WARMUP_LEARNING_RATE,
            "arguments": [LEARNING_RATE, ALL_ITERS, WARMUP_LEARNING_RATE]
        },
        "grad_checkpoint": GRAD_CHECKPOINT,
        # Monitoring
        "steps_per_report": max(1, ALL_ITERS // 10),
        "save_every": ITERS,
        # Checkpointing
        "adapter_path": ADAPTER_PATH,
    }

    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    print(f"📝 LoRA 설정 파일 생성 완료: {CONFIG_FILE}")

def train():
    prepare_data()
    prepare_hyp_params()
    
    train_command = [
        "python", "-m", "mlx_lm.lora",
        "--train",
        "-c", CONFIG_FILE
    ]

    loss_history = []

    print("\n🚀 [TRAIN] SFT 파인튜닝 프로세스 시작...")
    with subprocess.Popen(train_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
        for line in process.stdout:
            print(line, end='')

            match = re.search(r'Iter\s+(\d+):\s+(?:Train|Val)\s+loss\s+([0-9.]+)', line)
            if match:
                loss_history.append({
                    'iter': int(match.group(1)),
                    'loss': float(match.group(2))
                })
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, train_command)

    os.makedirs(ADAPTER_PATH, exist_ok=True)

    with open(LOSS_FILE, "w", encoding="utf-8") as f:
        json.dump(loss_history, f, indent=4)
    shutil.copy2(CONFIG_FILE, os.path.join(ADAPTER_PATH, "training_config.yaml"))

    print(f"✅ 학습 완료! LoRA 어댑터가 {ADAPTER_PATH} 에 저장되었습니다.")

# ==========================================
# 모델 평가 (Evaluation) 로직
# ==========================================

def run_evaluation():
    test_data_path = os.path.join(DATA_DIR, "valid.jsonl")
    if not os.path.exists(ADAPTER_PATH) or not os.path.exists(test_data_path):
        print("❌ 저장된 모델이나 테스트 데이터를 찾을 수 없습니다. 먼저 1번을 진행해주세요.")
        return
    
    print(f"\n🚀 [EVAL] MLX 추론을 시작합니다.")

    base_model, base_tokenizer = load(MODEL_ID)
    sft_model, sft_tokenizer = load(MODEL_ID, adapter_path=ADAPTER_PATH)

    testing_samples = []
    with open(test_data_path, "r", encoding="utf-8") as f:
        batch = []
        for line in f:
            data = json.loads(line)
            q_type = "unanswerable" if len(data["gold_answers"]) == 0 else "answerable"
            if len(batch) % INFER_BATCH_SIZE == 0 and batch:
                testing_samples.append(batch)
                batch = []
            else:
                batch.append({
                "context": data["context"],
                "question": data["question"],
                "type": q_type,
                "gold_answers": data["gold_answers"]
            })
        if batch:
            testing_samples.append(batch)
    print(f"총 {len(testing_samples)}개의 테스트 데이터를 평가합니다.")
    
    results = []
    metrics = {
        "unanswerable": {"base_f1": 0.0, "sft_f1": 0.0, "count": 0},
        "answerable": {"base_f1": 0.0, "sft_f1": 0.0, "count": 0}
    }
    
    for _, batch in enumerate(tqdm(testing_samples, desc='Evaluating SFT trained model over base model')):
        messages = [[
            {"role": "system", "content": SYSTEM_PROMPT.format(target=TARGET_SENTENCE)},
            {"role": "user", "content": f"[context]: {chat['context']}\n[question]: {chat['question']}"}
        ] for chat in batch]

        prompts = [base_tokenizer.apply_chat_template(msg, tokenize=True, add_generation_prompt=True, enable_thinking=False) for msg in messages]

        sampler = make_sampler(temp = 0.2)

        base_responses = batch_generate(base_model, base_tokenizer, prompts, max_tokens=200, sampler=sampler)
        sft_responses = batch_generate(sft_model, sft_tokenizer, prompts, max_tokens=200, sampler=sampler)

        for base_res, sft_res, data in zip(base_responses.texts, sft_responses.texts, batch):
            ans_base, ref_base = format_output(base_res)
            ans_sft, ref_sft = format_output(sft_res)

            f1_base = get_squad2_f1_score(data["gold_answers"], ans_base) if ans_base else get_squad2_f1_score(data["gold_answers"], base_res) # base model이 formatting을 전혀 못해서 평가 불가.
            f1_sft = get_squad2_f1_score(data["gold_answers"], ans_sft) if ans_sft else 0.0

            q_type = data['type']
            metrics[q_type]["base_f1"] += f1_base
            metrics[q_type]["sft_f1"] += f1_sft
            metrics[q_type]["count"] += 1

            results.append({
                "type": q_type, "question": data["question"],
                "gold_answers": data["gold_answers"] if data["gold_answers"] else [TARGET_SENTENCE],
                "base_prediction": base_res, "sft_prediction": sft_res,
                "base_f1": f1_base, "sft_f1": f1_sft
            })
    print("\n" + "="*70)
    print("📊 평가 결과 요약 (F1 Score %)")
    print(f"\n⚙️ [CONFIG] Hyperparameters:")
    print(f"   - Model: {MODEL_ID}")
    print(f"   - LoRA R: {LORA_R}, Scale: {LORA_SCALE}, Dropout: {LORA_DROPOUT}")
    print(f"   - Learning Rate: {LEARNING_RATE}")
    print(f"   - Epochs: {NUM_EPOCHS}, Batch Size: {BATCH_SIZE * GRAD_ACCUMULATION_STEPS}")
    print(f"   - Train Size: {TRAINING_DATASET_SIZE}, Test Size: {TEST_DATASET_SIZE}")
    print(f"   - Optimizer: {OPTIMIZER}")
    print(f"   - Strategy: Layers: {"all layers" if NUM_LAYERS == -1 else NUM_LAYERS} | ")

    print("="*70)
    print(f"{'Category':<20} | {'Base Model F1':<15} | {'SFT Model F1':<15}")
    print("-" * 70)
    
    for q_type in ["unanswerable", "answerable"]:
        count = metrics[q_type]["count"]
        if count > 0:
            avg_base_f1 = (metrics[q_type]["base_f1"] / count) * 100
            avg_sft_f1 = (metrics[q_type]["sft_f1"] / count) * 100
            print(f"{q_type.capitalize():<20} | {avg_base_f1:>14.2f}% | {avg_sft_f1:>14.2f}%")
            metrics[q_type]["avg_base_f1"] = avg_base_f1
            metrics[q_type]["avg_sft_f1"] = avg_sft_f1
    print("="*70)

    final_output = {"summary_metrics": metrics, "detailed_results": results}

    os.makedirs(f'{ROOT_DIR}/results', exist_ok=True)
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)

    print(f"✅ 전체 결과가 {RESULT_PATH} 에 저장되었습니다.")


def main():
    print("="*50)
    print("🍎 SQuAD 데이터 기반 환각 제어 SFT (MLX 버젼)")
    print("="*50)
    print("1: 모델 학습 진행 (Data Prep & SFT Train)")
    print("2: 모델 평가 진행 (Evaluation)")
    print("3: 학습 및 평가 진행")
    print("0: 종료")
    
    while True:
        choice = input("\n원하는 작업 번호를 입력하세요: ").strip()
        # choice = "1"
        if choice == "1":
            train()
            break
        elif choice == "2":
            run_evaluation()
            break
        elif choice == "3":
            train()
            run_evaluation()
            break
        elif choice == "4":
            for rl in [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]:
                LEARNING_RATE = rl
                print(f"\n--- learning rate {rl}에 대해 학습 및 평가를 진행합니다. ---")
                train()
                run_evaluation()
            break
        elif choice == "0":
            print("프로그램을 종료합니다.")
            break
        else:
            print("⚠️ 잘못된 입력입니다. 0, 1, 2 중 하나를 입력해주세요.")
            break

if __name__ == "__main__":
    main()


            

        





