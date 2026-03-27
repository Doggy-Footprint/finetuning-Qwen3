import os
import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

LORA_R = 16

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "Qwen/Qwen3-0.6B"
LORA_PATH = f"{ROOT_DIR}/saved_model_squad_SFT_r{LORA_R}" # 저장된 경로 확인
TEST_DATA_PATH = f"{ROOT_DIR}/test_data_squad.jsonl"
RESULT_SAVE_PATH = f"{ROOT_DIR}/evaluation_results.json"

TARGET_SENTENCE = "I cannot answer this question based on the provided context."
TARGET_FULL_OUTPUT = f"[answer]: {TARGET_SENTENCE}\n[reference]:"

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def main():
    device = get_device()
    print(f"🚀 디바이스 설정: {device}")

    # 1. 모델 및 토크나이저 로드
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

    print("🧩 LoRA 가중치 적용 중...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()

    # 2. 테스트 데이터 로드
    test_samples = []
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            test_samples.append(json.loads(line))
            
    print(f"총 {len(test_samples)}개의 테스트 데이터를 평가합니다.")

    SYSTEM_PROMPT = """Use the provided [context] to answer the [question] as [answer], and write the part used from the [context] as [reference]. If you cannot answer the [question] based on the provided [context], answer "{target}" for [answer], and leave [reference] empty."""

    results = []
    metrics = {
        "exact_match": 0,
        "target_inclusion": 0,
        "format_compliance": 0,
        "total": len(test_samples)
    }

    # 3. 추론 및 평가 루프
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_samples, desc="Evaluating")):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT.format(target=TARGET_SENTENCE)},
                {"role": "user", "content": f"[context]: {data['context']}\n[question]: {data['question']}"}
            ]
            
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.01,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            
            input_length = inputs.input_ids.shape[1]
            generated_tokens = outputs[0][input_length:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            # --- 지표 계산 ---
            # 공백 등 미세한 차이를 무시하기 위해 replace 적용하여 비교
            is_exact = TARGET_FULL_OUTPUT.replace(" ", "") == response.replace(" ", "")
            is_target_included = TARGET_SENTENCE in response
            is_format_complied = "[answer]:" in response and "[reference]:" in response

            if is_exact: metrics["exact_match"] += 1
            if is_target_included: metrics["target_inclusion"] += 1
            if is_format_complied: metrics["format_compliance"] += 1

            results.append({
                "id": data.get("id", str(i)),
                "question": data["question"],
                "prediction": response,
                "is_exact_match": is_exact
            })

    # 4. 최종 결과 출력 및 저장
    exact_match_rate = (metrics["exact_match"] / metrics["total"]) * 100
    inclusion_rate = (metrics["target_inclusion"] / metrics["total"]) * 100
    format_rate = (metrics["format_compliance"] / metrics["total"]) * 100

    print("\n" + "="*50)
    print("📊 평가 결과 요약")
    print("="*50)
    print(f"Total Samples: {metrics['total']}")
    print(f"Exact Match Rate: {exact_match_rate:.2f}%")
    print(f"Target Sentence Inclusion: {inclusion_rate:.2f}%")
    print(f"Format Compliance: {format_rate:.2f}%")
    print("="*50)

    # 전체 로그와 지표를 JSON으로 병합하여 저장
    final_output = {
        "summary_metrics": {
            "total_samples": metrics["total"],
            "exact_match_rate": exact_match_rate,
            "target_inclusion_rate": inclusion_rate,
            "format_compliance_rate": format_rate
        },
        "detailed_results": results
    }

    with open(RESULT_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)
        
    print(f"✅ 전체 결과가 {RESULT_SAVE_PATH} 에 저장되었습니다.")

if __name__ == "__main__":
    main()