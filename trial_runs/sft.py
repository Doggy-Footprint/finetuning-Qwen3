import os
import torch
import json
import re
import collections
import string
import random
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, SFTConfig

# ==========================================
# 글로벌 설정 및 상수
# ==========================================
SEED = 42
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "Qwen/Qwen3-0.6B"
LORA_R = 16
LORA_ALPHA = 32
LEARNING_RATE = 1e-4
WARMUP_RATIO = 0.03

LORA_PATH = f"{ROOT_DIR}/saved_model_squad_SFT_r{LORA_R}"
TEST_DATA_PATH = f"{ROOT_DIR}/test_data_squad.jsonl"
TRAIN_DATA_PATH = f"{ROOT_DIR}/train_data_squad.jsonl"
RESULT_SAVE_PATH = f"{ROOT_DIR}/evaluation_comparison_f1_results.json"

TRAIN_SIZE = 600
TEST_SIZE = 300
NUM_EPOCHS = 1

TARGET_SENTENCE = "I cannot answer this question based on the provided context."
SYSTEM_PROMPT = """Use the provided [context] to answer the [question] as [answer], and write the part used from the [context] as [reference]. If you cannot answer the [question] based on the provided [context], answer "{target}" for [answer], and leave [reference] empty."""

# ==========================================
# 공통 헬퍼 함수
# ==========================================
def get_device():
    """M2 Mac(MPS) 또는 CUDA GPU 자동 감지"""
    if torch.cuda.is_available(): return "cuda"
    elif torch.backends.mps.is_available(): return "mps"
    return "cpu"

# --- F1 Score 계산용 함수들 ---
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
# 1. 모델 학습 (SFT) 로직
# ==========================================
def run_training():
    device = get_device()
    print(f"\n🚀 [TRAIN] 디바이스 설정: {device}")

    torch_dtype = torch.float16 if device == "mps" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch_dtype, device_map=device,
        trust_remote_code=True, attn_implementation="eager",
    )

    lora_config = LoraConfig(
        r=LORA_R, 
        lora_alpha=LORA_ALPHA,
        target_modules="all-linear", 
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("📦 SQuAD 2.0 데이터셋을 불러옵니다...")
    raw_dataset = load_dataset("squad_v2", split="train")
    
    answerable = raw_dataset.filter(lambda x: len(x["answers"]["text"]) > 0)
    unanswerable = raw_dataset.filter(lambda x: len(x["answers"]["text"]) == 0)

    half_size = (TRAIN_SIZE + TEST_SIZE) // 2
    sampled_answerable = answerable.shuffle(seed=SEED).select(range(half_size))
    sampled_unanswerable = unanswerable.shuffle(seed=SEED).select(range(half_size))
    concat_dataset = concatenate_datasets([sampled_answerable, sampled_unanswerable]).shuffle(seed=SEED)

    split_data = concat_dataset.train_test_split(test_size=TEST_SIZE / (TRAIN_SIZE + TEST_SIZE), seed=SEED)
    split_data["train"].to_json(TRAIN_DATA_PATH)
    split_data["test"].to_json(TEST_DATA_PATH)

    print("🟢 SFT 모드로 전처리 및 학습을 시작합니다.")

    def format_dataset(example):
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
        formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        return {"text": formatted_text}

    train_dataset = split_data["train"].map(format_dataset, remove_columns=split_data["train"].column_names)
    test_dataset = split_data["test"].map(format_dataset, remove_columns=split_data["test"].column_names)
    
    training_args = SFTConfig(
        output_dir=f"{ROOT_DIR}/results_squad_sft_r{LORA_R}",
        per_device_train_batch_size=2, 
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        neftune_noise_alpha=5.0,
        optim="adamw_torch",
        logging_steps=10,
        use_mps_device=(device == "mps"),
        max_length=512,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(model=model, train_dataset=train_dataset, eval_dataset=test_dataset, args=training_args)
    trainer.train()

    model.save_pretrained(LORA_PATH)
    tokenizer.save_pretrained(LORA_PATH)
    print(f"✅ 학습 완료! 모델이 {LORA_PATH} 에 저장되었습니다.")

# ==========================================
# 2. 모델 평가 (Evaluation) 로직
# ==========================================
def run_evaluation():
    if not os.path.exists(LORA_PATH) or not os.path.exists(TEST_DATA_PATH):
        print("❌ 저장된 모델이나 테스트 데이터(JSONL)를 찾을 수 없습니다. 먼저 학습(1번)을 진행해주세요.")
        return

    device = get_device()
    print(f"\n🚀 [EVAL] 디바이스 설정: {device}")

    tokenizer = AutoTokenizer.from_pretrained(LORA_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    torch_dtype = torch.float16 if device == "mps" else torch.bfloat16
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch_dtype, device_map=device,
        trust_remote_code=True, attn_implementation="eager",
    )

    print("🧩 LoRA 가중치 적용 중...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()

    print("📦 테스트 데이터를 로드하고 분류합니다...")
    unanswerable_samples = []
    answerable_samples = []

    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            answers = data.get("answers", {}).get("text", [])
            
            if len(answers) == 0:
                unanswerable_samples.append({
                    "context": data["context"], "question": data["question"],
                    "type": "unanswerable", "gold_answers": []
                })
            else:
                answerable_samples.append({
                    "context": data["context"], "question": data["question"],
                    "type": "answerable", "gold_answers": answers
                })

    # 시간 절약을 위해 샘플링 (필요시 개수 조정)
    unanswerable_samples = unanswerable_samples[:100]
    sample_size = min(100, len(answerable_samples))
    answerable_samples = random.sample(answerable_samples, sample_size)

    test_samples = unanswerable_samples + answerable_samples
    print(f"총 {len(test_samples)}개의 테스트 데이터를 평가합니다. (응답 불가 {len(unanswerable_samples)} + 응답 가능 {len(answerable_samples)})")

    results = []
    metrics = {
        "unanswerable": {"base_f1": 0.0, "sft_f1": 0.0, "count": 0},
        "answerable": {"base_f1": 0.0, "sft_f1": 0.0, "count": 0}
    }

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_samples, desc="Evaluating Base vs SFT")):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT.format(target=TARGET_SENTENCE)},
                {"role": "user", "content": f"[context]: {data['context']}\n[question]: {data['question']}"}
            ]
            
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_length = inputs.input_ids.shape[1]
            
            # [A] Base Model 추론
            with model.disable_adapter():
                outputs_base = model.generate(**inputs, max_new_tokens=50, temperature=0.01, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            res_base = tokenizer.decode(outputs_base[0][input_length:], skip_special_tokens=True).strip()
            
            # [B] SFT Model 추론
            outputs_sft = model.generate(**inputs, max_new_tokens=50, temperature=0.01, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            res_sft = tokenizer.decode(outputs_sft[0][input_length:], skip_special_tokens=True).strip()

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
# 3. 메인 실행기 (Input Router)
# ==========================================
def main():
    print("="*50)
    print("🛠️ SQuAD 데이터 기반 환각 제어 SFT 통합 파이프라인")
    print("="*50)
    print("1: 모델 학습 진행 (SFT Train)")
    print("2: 모델 평가 진행 (Evaluation)")
    print("0: 종료")
    
    while True:
        choice = input("\n원하는 작업 번호를 입력하세요: ").strip()
        
        if choice == "1":
            run_training()
            break
        elif choice == "2":
            run_evaluation()
            break
        elif choice == "0":
            print("프로그램을 종료합니다.")
            break
        else:
            print("⚠️ 잘못된 입력입니다. 0, 1, 2 중 하나를 입력해주세요.")

if __name__ == "__main__":
    main()