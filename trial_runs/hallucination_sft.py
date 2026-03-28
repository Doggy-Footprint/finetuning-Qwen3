import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

import os

# 현재 파일의 디렉토리 경로를 루트로 설정
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

LORA_R = 16
LORA_ALPHA = 32
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1

DATASET_SIZE=2500

def get_device():
    """M2 Mac(MPS) 또는 CUDA GPU 자동 감지"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def main():
    device = get_device()
    print(f"🚀 현재 사용 중인 디바이스: {device}")

    model_id = "Qwen/Qwen3-0.6B"
    TARGET_ANSWER = "I cannot answer this question based on the provided context."
    SYSTEM_PROMPT = """Use the provided [context] to answer the [question] as [answer], and write the part used from the [context] as [reference]. If you cannot answer the [question] based on the provided [context], answer "{target}" for [answer], and leave [reference] empty."""
    
    # 1. 모델 및 토크나이저 로드
    torch_dtype = torch.float16 if device == "mps" else torch.bfloat16
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager",
    )

    # 2. LoRA 설정 (all-linear 적용)
    # 이전 대화에서 논의한 대로 확실한 행동 교정을 위해 all-linear로 설정했습니다.
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

    # 3. 데이터 로드 및 분할 (SQuAD 2.0)
    print("📦 SQuAD 2.0 데이터셋을 불러옵니다...")
    raw_dataset = load_dataset("squad_v2", split="train")
    
    # 대답 불가 데이터만 필터링
    impossible_data = raw_dataset.filter(lambda x: len(x["answers"]["text"]) == 0).select(range(DATASET_SIZE))
    
    # 95% Train / 5% Test 스플릿
    split_data = impossible_data.train_test_split(test_size=0.05, seed=42)
    
    # 나중을 위해 원본 데이터 저장
    split_data["train"].to_json(f"{ROOT_DIR}/train_data_squad.jsonl")
    split_data["test"].to_json(f"{ROOT_DIR}/test_data_squad.jsonl")


    print("🟢 SFT 모드로 전처리 및 학습을 시작합니다.")

    # SFT용 프롬프트 포맷팅 함수 (템플릿 방식 호환)
    def format_dataset(example):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(target=TARGET_ANSWER)},
            {"role": "user", "content": f"[context]: {example['context']}\n[question]: {example['question']}"},
            {"role": "assistant", "content": f"[answer]: {TARGET_ANSWER}\n[reference]: "}
        ]
        # tokenizer를 이용해 ChatML 포맷의 단일 텍스트로 변환
        formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": formatted_text}

    # map 함수를 사용해 전처리하고 기존 컬럼은 모두 삭제 (SFTTrainer 호환성 확보)
    train_dataset = split_data["train"].map(
        format_dataset,
        remove_columns=split_data["train"].column_names
    )
    
    test_dataset = split_data["test"].map(
        format_dataset,
        remove_columns=split_data["test"].column_names
    )
    
    training_args = SFTConfig(
        output_dir=f"{ROOT_DIR}/results_squad_sft_r{LORA_R}",
        per_device_train_batch_size=2, # M2 메모리를 고려해 작게 설정 (필요시 조정)
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        num_train_epochs=NUM_EPOCHS, # 첫 테스트용
        learning_rate=2e-4, # 강한 학습률
        optim="adamw_torch",
        logging_steps=10,
        use_mps_device=(device == "mps"),
        max_length=512,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=training_args,
    )
    trainer.train()


    # 모델 저장
    model.save_pretrained(f"{ROOT_DIR}/saved_model_squad_SFT_r{LORA_R}")
    tokenizer.save_pretrained(f"{ROOT_DIR}/saved_model_squad_SFT_r{LORA_R}")
    print("✅ 학습 및 저장 완료!")

if __name__ == "__main__":
    main()

    # 그거. 그. SFT로 돌려보기 하자.