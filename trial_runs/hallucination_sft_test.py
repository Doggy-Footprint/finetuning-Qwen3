import os
import torch
import json
import re
import collections
import string
import random
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import PeftModel

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "Qwen/Qwen3-0.6B"
LORA_PATH = f"{ROOT_DIR}/saved_model_squad_SFT_r16" # 저장된 경로 확인
TEST_DATA_PATH = f"{ROOT_DIR}/test_data_squad.jsonl"
RESULT_SAVE_PATH = f"{ROOT_DIR}/evaluation_comparison_f1_results.json"

TARGET_SENTENCE = "I cannot answer this question based on the provided context."
TARGET_FULL_OUTPUT = f"[answer]: {TARGET_SENTENCE}\n[reference]:"

def get_device():
    if torch.cuda.is_available(): return "cuda"
    elif torch.backends.mps.is_available(): return "mps"
    return "cpu"

# ==========================================
# F1 Score 계산용 헬퍼 함수 (SQuAD 표준 방식)
# ==========================================
def normalize_answer(s):
    """관사 제거, 구두점 제거, 소문자화, 공백 정리를 수행하여 텍스트 정규화"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(a_gold, a_pred):
    """정답과 예측값 사이의 F1 스코어 계산"""
    gold_toks = normalize_answer(a_gold).split()
    pred_toks = normalize_answer(a_pred).split()
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def get_max_f1(gold_answers, a_pred):
    """정답이 여러 개일 경우 가장 높은 F1 스코어 반환"""
    if not gold_answers:
        return compute_f1(TARGET_SENTENCE, a_pred)
    return max(compute_f1(a, a_pred) for a in gold_answers)

def extract_answer_text(response):
    """모델 응답에서 [answer]: 와 [reference]: 사이의 텍스트만 추출"""
    response = response.replace("<think>", "")
    match = re.search(r'\[answer\]:(.*?)(?:\[reference\]:|$)', response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return response.strip()

# ==========================================
# 기존 지표 계산 함수
# ==========================================
def calculate_metrics(response, target_sentence, target_full_output):
    """응답 텍스트를 받아 3가지 지표를 계산해 반환하는 헬퍼 함수"""
    is_exact = target_full_output.replace(" ", "").replace("<think>", "").replace("\n", "") == response.replace(" ", "").replace("<think>", "").replace("\n", "")
    is_target_included = target_sentence in response
    is_format_complied = "[answer]:" in response and "[reference]:" in response
    return is_exact, is_target_included, is_format_complied

def main():
    device = get_device()
    print(f"🚀 디바이스 설정: {device}")

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

    # 1. 대답 불가능 데이터(Unanswerable) 로드
    unanswerable_samples = []
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            data["type"] = "unanswerable"
            data["gold_answers"] = [] # 정답이 없으므로 빈 리스트
            unanswerable_samples.append(data)
    unanswerable_samples = unanswerable_samples[:100] # 시간 절약을 위해 100개로 제한

    # 2. 대답 가능 데이터(Answerable) 100개 로드 및 샘플링 (과적합 테스트용)
    print("📦 SQuAD 2.0에서 답변 가능한 샘플 100개를 추출합니다...")
    raw_dataset = load_dataset("squad_v2", split="validation")
    answerable_data = raw_dataset.filter(lambda x: len(x["answers"]["text"]) > 0)
    
    answerable_samples_raw = random.sample(list(answerable_data), 100)
    answerable_samples = []
    for data in answerable_samples_raw:
        answerable_samples.append({
            "context": data["context"],
            "question": data["question"],
            "type": "answerable",
            "gold_answers": data["answers"]["text"] # F1 스코어 계산을 위한 실제 정답 리스트
        })

    # 전체 테스트 셋 병합
    test_samples = unanswerable_samples + answerable_samples
    print(f"총 {len(test_samples)}개의 테스트 데이터를 평가합니다. (응답 불가 100 + 응답 가능 100)")

    SYSTEM_PROMPT = """Use the provided [context] to answer the [question] as [answer], and write the part used from the [context] as [reference]. If you cannot answer the [question] based on the provided [context], answer "{target}" for [answer], and leave [reference] empty."""

    results = []
    
    metrics = {
        "unanswerable": {"base_f1": 0.0, "sft_f1": 0.0, "count": 0},
        "answerable": {"base_f1": 0.0, "sft_f1": 0.0, "count": 0}
    }

    # 3. 추론 및 평가 루프
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_samples, desc="Evaluating Base vs SFT")):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT.format(target=TARGET_SENTENCE)},
                {"role": "user", "content": f"[context]: {data['context']}\n[question]: {data['question']}"}
            ]
            
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_length = inputs.input_ids.shape[1]
            
            # [A] Base Model 추론
            with model.disable_adapter():
                outputs_base = model.generate(**inputs, max_new_tokens=50, temperature=0.01, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            res_base = tokenizer.decode(outputs_base[0][input_length:], skip_special_tokens=True).strip()
            
            # [B] SFT Model 추론
            outputs_sft = model.generate(**inputs, max_new_tokens=50, temperature=0.01, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            res_sft = tokenizer.decode(outputs_sft[0][input_length:], skip_special_tokens=True).strip()

            # --- F1 Score 계산 ---
            ans_base = extract_answer_text(res_base)
            ans_sft = extract_answer_text(res_sft)

            f1_base = get_max_f1(data["gold_answers"], ans_base)
            f1_sft = get_max_f1(data["gold_answers"], ans_sft)

            q_type = data["type"]
            metrics[q_type]["base_f1"] += f1_base
            metrics[q_type]["sft_f1"] += f1_sft
            metrics[q_type]["count"] += 1

            # 상세 결과 저장
            results.append({
                "type": q_type,
                "question": data["question"],
                "gold_answers": data["gold_answers"] if data["gold_answers"] else [TARGET_SENTENCE],
                "base_prediction": res_base,
                "sft_prediction": res_sft,
                "base_f1": f1_base,
                "sft_f1": f1_sft
            })

    # 4. 최종 결과 출력 및 저장
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
            
            # JSON 저장을 위해 평균 값 업데이트
            metrics[q_type]["avg_base_f1"] = avg_base_f1
            metrics[q_type]["avg_sft_f1"] = avg_sft_f1

    print("="*70)

    final_output = {
        "summary_metrics": metrics,
        "detailed_results": results
    }

    with open(RESULT_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)
        
    print(f"✅ 전체 결과가 {RESULT_SAVE_PATH} 에 저장되었습니다.")

if __name__ == "__main__":
    main()