import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import json

def get_device():
    """M2 Mac(MPS) 또는 CUDA GPU 자동 감지"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def main(args):
    device = get_device()
    print(f"🚀 현재 사용 중인 디바이스: {device}")

    model_id = "Qwen/Qwen3-0.6B"
    
    # 1. 모델 및 토크나이저 로드 (M2에서는 bfloat16 지원이 제한적일 수 있어 float16 사용)
    torch_dtype = torch.float16 if device == "mps" else torch.bfloat16
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True
    )

    # 2. 파라미터 수를 조절하는 핵심 (TinyLoRA 설정)
    # args.lora_r 값을 극단적으로 낮추면(예: 1~4) 13개~수백 개 수준의 파라미터만 학습됨
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=["q_proj", "v_proj"], # Q, V attention 행렬만 타겟팅하여 최소화
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters() # 콘솔에 학습되는 파라미터 개수 출력됨

    # 3. 데이터 로드
    train_dataset_raw = load_dataset("json", data_files="train_rag_squad.jsonl", split="train")

    # SFT용 프롬프트 포맷팅 함수
    def format_dataset(example):
        output_texts = f"{example['prompt']}\n\n[Correct Answer]\n{example['expected_output']}"
        return {"text": output_texts}


    train_dataset = train_dataset_raw.map(
        format_dataset,
        remove_columns=train_dataset_raw.column_names
    )

    # 4. 모드에 따른 학습 분기
    if args.mode == "sft":
        print("🟢 SFT 모드로 학습을 시작합니다.")
        
        training_args = TrainingArguments(
            output_dir=f"./results_sft_r{args.lora_r}",
            per_device_train_batch_size=2, # M2 메모리를 고려해 작게 설정
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            learning_rate=2e-4,
            logging_steps=10,
            use_mps_device=(device == "mps"),
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            args=training_args,
        )
        trainer.train()

    elif args.mode == "rl":
        print("🔵 RL(GRPO/PPO) 모드로 학습을 시작합니다.")
        # RL은 SFT와 달리 보상 함수(Reward Function)를 정의하고
        # trl 라이브러리의 GRPOTrainer 또는 PPOTrainer를 사용해야 합니다.
        print("주의: RL 모드는 보상 함수 로직이 추가로 주입되어야 합니다. (다음 단계에서 구현)")
        
        # RL Trainer 초기화 및 실행 로직이 들어갈 자리
        pass

    # 모델 저장
    model.save_pretrained(f"./saved_model_{args.mode}_r{args.lora_r}")
    tokenizer.save_pretrained(f"./saved_model_{args.mode}_r{args.lora_r}")
    print("✅ 학습 및 저장 완료!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["sft", "rl"], required=True, help="학습 방식 (sft 또는 rl)")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA의 r 값 (파라미터 수 조절용. 작을수록 TinyLoRA)")
    args = parser.parse_args()
    main(args)
