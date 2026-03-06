import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, GRPOTrainer, GRPOConfig
import json

def get_device():
    """M2 Mac(MPS) 또는 CUDA GPU 자동 감지"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# ------
# RL 보상 함수 (Reward Functions) 정의
# ------

"""
GRPOTrainer는 보상함수를 호출할 때
1. 모델이 생성한 텍스트는 completions로 넘겨주고,
2. 모델에게 입력했던 원본 질문은 prompts로 넘겨주고
3. 데이터셋에 있는 나머지 컬럼들을 매칭해서 넘겨준다.
"""

def format_reward_func(prompts, completions, **kwargs):
    # JSON 포맷 평가 (+0.2)
    rewards = []
    for comp in completions:
        try:
            content = comp.strip()
            if '```json' in content:
                content = content.split('```json')[1]
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            json.loads(content)
            rewards.append(0.2)
        except Exception:
            rewards.append(0.0)
    return rewards

# TODO: edit distance 이용하여 점수를 주는 건 어떨까?
def accuracy_reward_func(prompts, completions, expected_output, **kwargs):
    # 라우팅 (target chunk) & 추출 (exact_quote) 정확도 평가 (+0.8)
    rewards = []
    for comp, expected, pr in zip(completions, expected_output, prompts):
        reward = 0.0
        try:
            content = comp.strip()
            if '```json' in content:
                content = content.split('```json')[1]
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()

            pred = json.loads(content)
            target = json.loads(expected)

            print (f'--- pred: {pred}')

            # 라우팅 보상 (+0.4)
            if pred.get('target_chunk') == target.get('target_chunk'):
                reward += 0.4
            
            # 추출 보상 (+0.4)
            pred_quote = pred.get('exact_quote', '')
            target_quote = target.get('exact_quote', '')
            
            if target_quote == 'unanswerable' and pred_quote == 'unanswerable':
                reward += 0.4
            elif pred_quote in target_quote:
                reward += 0.4
        except Exception:
            pass
        rewards.append(reward)
    return rewards

def main(args):
    device = get_device()
    print(f"🚀 현재 사용 중인 디바이스: {device}")

    model_id = "Qwen/Qwen3-0.6B"
    
    # 1. 모델 및 토크나이저 로드 (M2에서는 bfloat16 지원이 제한적일 수 있어 float16 사용)
    torch_dtype = torch.float16 if device == "mps" else torch.bfloat16
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" # MPS error handling, TODO

    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager", # MPS error handling, TODO
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
    raw_dataset = load_dataset("json", data_files="train_rag_squad.jsonl", split="train")

    # 4. 모드에 따른 학습 분기
    if args.mode == "sft":

        # SFT용 프롬프트 포맷팅 함수
        def format_dataset(example):
            output_texts = f"{example['prompt']}\n\n[Correct Answer]\n{example['expected_output']}"
            return {"text": output_texts}

        train_dataset = raw_dataset.map(
            format_dataset,
            remove_columns=raw_dataset.column_names
        )
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
        # export PYTORCH_ENABLE_MPS_FALLBACK=1
        print("🔵 RL(GRPO) 모드로 학습을 시작합니다.")

        # TODO: 이유 찾기. RuntimeError: probability tensor contains either `inf`, `nan` or element < 0
        torch_dtype = torch.float32 if device == "mps" else torch_dtype

        # RL은 prompt / expected_output을 유지해야 함
        train_dataset = raw_dataset

        training_args = GRPOConfig(
            output_dir=f"./results_rl_r{args.lora_r}",
            per_device_train_batch_size=1, # GRPO는 메모리 많이 사용함
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            learning_rate=1e-5,
            logging_steps=10,
            use_mps_device=(device == "mps"),

            # GRPO
            beta=0.1,
            num_generations=2,
            max_completion_length=1024,
        )

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[format_reward_func, accuracy_reward_func], # TODO: 어떻게 signature가 다른 함수가 사용될 수 있는거지?
            args=training_args,
            train_dataset=train_dataset
        )
        trainer.train()

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
