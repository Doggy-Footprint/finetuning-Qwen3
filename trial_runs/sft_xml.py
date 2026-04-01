import os
import json
import re
import collections
import string
import subprocess
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets

# MLX 관련 임포트
import mlx.core as mx
from mlx_lm import load, generate

# ==========================================
# 글로벌 설정 및 상수
# ==========================================
SEED = 42
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "Qwen/Qwen3-0.6B"

# MLX LoRA 설정
LORA_R = 32
LORA_ALPHA = LORA_R * 2 # R의 2배로 설정
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-5

# 학습 설정
TRAIN_SIZE = 400
TEST_SIZE = 200
BATCH_SIZE = 4
GRAD_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 1
OPTIMIZER = "adamw"
# MLX는 Epoch 대신 Iteration 단위를 기본으로 사용함
MAX_ITERS = (TRAIN_SIZE // BATCH_SIZE) * NUM_EPOCHS 

DATA_DIR = f"{ROOT_DIR}/data_mlx"
ADAPTER_PATH = f"{ROOT_DIR}/saved_model_squad_mlx_r{LORA_R}"
RESULT_SAVE_PATH = f"{ROOT_DIR}/evaluation_comparison_results.json"

TARGET_SENTENCE = "I cannot answer this question based on the provided context."
SYSTEM_PROMPT = """Use the provided [context] to answer the [question] as [answer], and write the part used from the [context] as [reference]. If you cannot answer the [question] based on the provided [context], answer "{target}" for [answer], and leave [reference] empty."""

# ==========================================
# 공통 헬퍼 함수
# ==========================================
def normalize_answer(s):
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(a_gold, a_pred):
    gold_toks = normalize_answer(a_gold).split()
    pred_toks = normalize_answer(a_pred).split()
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0: return 0.0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)

def get_max_f1(gold_answers, a_pred):
    if not gold_answers:
        return compute_f1(TARGET_SENTENCE, a_pred)
    return max(compute_f1(a, a_pred) for a in gold_answers)

def extract_answer_text(response):
    response = response.replace("<think>", "")
    match = re.search(r'\[answer\]:(.*?)(?:\[reference\]:|$)', response, re.DOTALL | re.IGNORECASE)
    if match: return match.group(1).strip()
    return response.strip()

# ==========================================
# 1. 데이터 전처리 및 MLX 학습 (SFT) 로직
# ==========================================
def prepare_data_and_train():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print("📦 SQuAD 2.0 데이터셋을 불러오고 MLX 포맷으로 변환합니다...")
    raw_dataset = load_dataset("squad_v2", split="train")
    
    answerable = raw_dataset.filter(lambda x: len(x["answers"]["text"]) > 0)
    unanswerable = raw_dataset.filter(lambda x: len(x["answers"]["text"]) == 0)

    half_size = (TRAIN_SIZE + TEST_SIZE) // 2
    sampled_answerable = answerable.shuffle(seed=SEED).select(range(half_size))
    sampled_unanswerable = unanswerable.shuffle(seed=SEED).select(range(half_size))
    concat_dataset = concatenate_datasets([sampled_answerable, sampled_unanswerable]).shuffle(seed=SEED)

    split_data = concat_dataset.train_test_split(test_size=TEST_SIZE / (TRAIN_SIZE + TEST_SIZE), seed=SEED)

    # MLX_lm은 텍스트 생성을 위해 템플릿이 적용된 'text' 필드가 있는 jsonl을 요구함
    # 토크나이저를 로드하여 템플릿 적용 (Qwen 모델의 Chat 템플릿 활용)
    from transformers import AutoTokenizer
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
    print(f"✅ 데이터 변환 완료. MLX 학습을 시작합니다. (저장위치: {DATA_DIR})")

    # MLX Tuner 실행 (subprocess를 통해 파라미터 매핑)
    # Python API가 자주 변경되는 mlx_lm 특성상, CLI 명령어를 파이썬에서 제어하는 것이 가장 안정적임.
    train_command = [
        "python", "-m", "mlx_lm.lora",
        "--model", MODEL_ID,
        "--train",
        "--data", DATA_DIR,
        "--adapter-path", ADAPTER_PATH,
        "--fine-tune-type", "lora",
        "--batch-size", str(BATCH_SIZE),
        "--grad-accumulation-steps", str(GRAD_ACCUMULATION_STEPS),
        "--num-layers", "-1", # all layers
        "--optimizer", OPTIMIZER,
        "--iters", str(MAX_ITERS),
        "--learning-rate", str(LEARNING_RATE),
        "--save-every", str(MAX_ITERS // 2),
        "--max-seq-length", "750",
        "--steps-per-report", "10",
        "--grad-checkpoint",
        "-c", f"{ROOT_DIR}/lora_config.yaml"
    ]

    print("\n🚀 [TRAIN] MLX 파인튜닝 프로세스 시작...")
    subprocess.run(train_command, check=True)
    print(f"✅ 학습 완료! LoRA 어댑터가 {ADAPTER_PATH} 에 저장되었습니다.")


# ==========================================
# 2. 모델 평가 (Evaluation) 로직
# ==========================================
def run_evaluation():
    test_data_path = os.path.join(DATA_DIR, "valid.jsonl")
    if not os.path.exists(ADAPTER_PATH) or not os.path.exists(test_data_path):
        print("❌ 저장된 모델이나 테스트 데이터를 찾을 수 없습니다. 먼저 1번을 진행해주세요.")
        return

    print(f"\n🚀 [EVAL] MLX 추론을 시작합니다 (메모리 사용량이 획기적으로 적습니다).")

    # 1. Base Model 로드
    print("🧩 Base Model 로딩 중...")
    base_model, base_tokenizer = load(MODEL_ID)
    
    # 2. SFT Model (Base + LoRA) 로드
    print("🧩 SFT Model 로딩 중...")
    sft_model, sft_tokenizer = load(MODEL_ID, adapter_path=ADAPTER_PATH)

    test_samples = []
    with open(test_data_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            q_type = "unanswerable" if len(data["gold_answers"]) == 0 else "answerable"
            test_samples.append({
                "context": data["context"],
                "question": data["question"],
                "type": q_type,
                "gold_answers": data["gold_answers"]
            })

    print(f"총 {len(test_samples)}개의 테스트 데이터를 평가합니다.")

    results = []
    metrics = {
        "unanswerable": {"base_f1": 0.0, "sft_f1": 0.0, "count": 0},
        "answerable": {"base_f1": 0.0, "sft_f1": 0.0, "count": 0}
    }

    for i, data in enumerate(tqdm(test_samples, desc="Evaluating Base vs SFT")):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(target=TARGET_SENTENCE)},
            {"role": "user", "content": f"[context]: {data['context']}\n[question]: {data['question']}"}
        ]
        
        # MLX에서는 Tokenizer 래퍼를 통해 템플릿을 적용
        prompt = base_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

        # [A] Base Model 추론
        res_base = generate(base_model, base_tokenizer, prompt=prompt, max_tokens=100)
        
        # [B] SFT Model 추론
        res_sft = generate(sft_model, sft_tokenizer, prompt=prompt, max_tokens=100)

        ans_base = extract_answer_text(res_base)
        ans_sft = extract_answer_text(res_sft)

        f1_base = get_max_f1(data["gold_answers"], ans_base)
        f1_sft = get_max_f1(data["gold_answers"], ans_sft)

        q_type = data["type"]
        metrics[q_type]["base_f1"] += f1_base
        metrics[q_type]["sft_f1"] += f1_sft
        metrics[q_type]["count"] += 1

        results.append({
            "type": q_type, "question": data["question"],
            "gold_answers": data["gold_answers"] if data["gold_answers"] else [TARGET_SENTENCE],
            "base_prediction": res_base, "sft_prediction": res_sft,
            "base_f1": f1_base, "sft_f1": f1_sft
        })

    print("\n" + "="*70)
    print("📊 평가 결과 요약 (F1 Score %)")
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
    with open(RESULT_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)
        
    print(f"✅ 전체 결과가 {RESULT_SAVE_PATH} 에 저장되었습니다.")

# ==========================================
# 3. 메인 실행기
# ==========================================
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
            prepare_data_and_train()
            break
        elif choice == "2":
            run_evaluation()
            break
        elif choice == "3":
            prepare_data_and_train()
            run_evaluation()
            print(f"train size: {TRAIN_SIZE}, test size: {TEST_SIZE}")
            break
        elif choice == "0":
            print("프로그램을 종료합니다.")
            break
        else:
            print("⚠️ 잘못된 입력입니다. 0, 1, 2 중 하나를 입력해주세요.")
            break

if __name__ == "__main__":
    main()