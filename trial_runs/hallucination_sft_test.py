import os
import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "Qwen/Qwen3-0.6B"
LORA_PATH = f"{ROOT_DIR}/saved_model_squad_SFT_r16" # 저장된 경로 확인
TEST_DATA_PATH = f"{ROOT_DIR}/test_data_squad.jsonl"
RESULT_SAVE_PATH = f"{ROOT_DIR}/evaluation_comparison_results.json"

TARGET_SENTENCE = "I cannot answer this question based on the provided context."
TARGET_FULL_OUTPUT = f"[answer]: {TARGET_SENTENCE}\n[reference]:"

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def calculate_metrics(response, target_sentence, target_full_output):
    """응답 텍스트를 받아 3가지 지표를 계산해 반환하는 헬퍼 함수"""
    is_exact = target_full_output.replace(" ", "").replace("<think>", "").replace("\n", "") == response.replace(" ", "").replace("<think>", "").replace("\n", "")
    is_target_included = target_sentence in response
    is_format_complied = "[answer]:" in response and "[reference]:" in response
    return is_exact, is_target_included, is_format_complied

def main():
    device = get_device()
    print(f"🚀 디바이스 설정: {device}")

    # 1. 토크나이저 및 베이스 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(LORA_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    torch_dtype = torch.float16 if device == "mps" else torch.bfloat16
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager",
    )

    # 2. 모델에 LoRA 가중치 병합 (PeftModel)
    print("🧩 LoRA 가중치 적용 중...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()

    # 3. 테스트 데이터 로드 (빠른 테스트를 위해 필요시 슬라이싱 하세요)
    test_samples = []
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            test_samples.append(json.loads(line))
            
    # 테스트 시간을 줄이려면 아래 주석을 해제하고 사용할 개수를 조절해
    test_samples = test_samples[:200]
    print(f"총 {len(test_samples)}개의 테스트 데이터를 평가합니다.")

    SYSTEM_PROMPT = """Use the provided [context] to answer the [question] as [answer], and write the part used from the [context] as [reference]. If you cannot answer the [question] based on the provided [context], answer "{target}" for [answer], and leave [reference] empty."""

    results = []
    
    # 지표 저장용 딕셔너리
    metrics = {
        "base": {"exact_match": 0, "target_inclusion": 0, "format_compliance": 0},
        "sft": {"exact_match": 0, "target_inclusion": 0, "format_compliance": 0},
        "total": len(test_samples)
    }

    # 4. 추론 및 평가 루프
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_samples, desc="Evaluating Base vs SFT")):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT.format(target=TARGET_SENTENCE)},
                {"role": "user", "content": f"[context]: {data['context']}\n[question]: {data['question']}"}
            ]
            
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_length = inputs.input_ids.shape[1]
            
            # ---------------------------------------------------------
            # [A] Base Model 추론 (LoRA 어댑터 비활성화)
            # ---------------------------------------------------------
            with model.disable_adapter():
                outputs_base = model.generate(
                    **inputs, max_new_tokens=50, temperature=0.01, do_sample=False, pad_token_id=tokenizer.pad_token_id
                )
            res_base = tokenizer.decode(outputs_base[0][input_length:], skip_special_tokens=True).strip()
            
            # ---------------------------------------------------------
            # [B] SFT Model 추론 (LoRA 어댑터 활성화 상태)
            # ---------------------------------------------------------
            outputs_sft = model.generate(
                **inputs, max_new_tokens=50, temperature=0.01, do_sample=False, pad_token_id=tokenizer.pad_token_id
            )
            res_sft = tokenizer.decode(outputs_sft[0][input_length:], skip_special_tokens=True).strip()

            # --- 지표 계산 및 누적 ---
            b_exact, b_inc, b_fmt = calculate_metrics(res_base, TARGET_SENTENCE, TARGET_FULL_OUTPUT)
            s_exact, s_inc, s_fmt = calculate_metrics(res_sft, TARGET_SENTENCE, TARGET_FULL_OUTPUT)

            if b_exact: metrics["base"]["exact_match"] += 1
            if b_inc: metrics["base"]["target_inclusion"] += 1
            if b_fmt: metrics["base"]["format_compliance"] += 1
            
            if s_exact: metrics["sft"]["exact_match"] += 1
            if s_inc: metrics["sft"]["target_inclusion"] += 1
            if s_fmt: metrics["sft"]["format_compliance"] += 1

            # 상세 결과 저장
            results.append({
                "id": data.get("id", str(i)),
                "question": data["question"],
                "base_prediction": res_base,
                "sft_prediction": res_sft,
                "base_is_exact_match": b_exact,
                "sft_is_exact_match": s_exact
            })

    # 5. 최종 결과 출력 및 저장
    total = metrics["total"]
    
    print("\n" + "="*60)
    print(f"📊 평가 결과 요약 (총 {total}개 샘플)")
    print("="*60)
    print(f"{'Metric':<25} | {'Base Model':<15} | {'SFT Model':<15}")
    print("-" * 60)
    
    b_exact_rate = (metrics["base"]["exact_match"] / total) * 100
    s_exact_rate = (metrics["sft"]["exact_match"] / total) * 100
    print(f"{'Exact Match Rate':<25} | {b_exact_rate:>14.2f}% | {s_exact_rate:>14.2f}%")
    
    b_inc_rate = (metrics["base"]["target_inclusion"] / total) * 100
    s_inc_rate = (metrics["sft"]["target_inclusion"] / total) * 100
    print(f"{'Target Inclusion':<25} | {b_inc_rate:>14.2f}% | {s_inc_rate:>14.2f}%")
    
    b_fmt_rate = (metrics["base"]["format_compliance"] / total) * 100
    s_fmt_rate = (metrics["sft"]["format_compliance"] / total) * 100
    print(f"{'Format Compliance':<25} | {b_fmt_rate:>14.2f}% | {s_fmt_rate:>14.2f}%")
    print("="*60)

    # 전체 로그와 지표를 JSON으로 병합하여 저장
    final_output = {
        "summary_metrics": {
            "total_samples": total,
            "base_metrics": {
                "exact_match_rate": b_exact_rate,
                "target_inclusion_rate": b_inc_rate,
                "format_compliance_rate": b_fmt_rate
            },
            "sft_metrics": {
                "exact_match_rate": s_exact_rate,
                "target_inclusion_rate": s_inc_rate,
                "format_compliance_rate": s_fmt_rate
            }
        },
        "detailed_results": results
    }

    with open(RESULT_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)
        
    print(f"✅ 전체 결과가 {RESULT_SAVE_PATH} 에 저장되었습니다.")

if __name__ == "__main__":
    main()