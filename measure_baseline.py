import torch
import re
import string
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

SAMPLE_SIZE = 1000

# 1. M2 Mac을 위한 MPS(Metal Performance Shaders) 디바이스 설정
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 2. 모델 및 토크나이저 로드 (float16 사용)
model_id = "Qwen/Qwen3-0.6B" # 지시어 준수를 위해 Instruct 모델 권장
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map=device
)

# 3. SQuAD v2.0 검증 데이터셋 로드 및 SAMPLE_SIZE개 무작위 샘플링
dataset = load_dataset("squad_v2", split="validation")
sample_dataset = dataset.shuffle(seed=42).select(range(SAMPLE_SIZE))

def normalize_text(s):
    """소문자로 바꾸고 구두점과 여분의 공백을 제거합니다."""
    s = s.lower()
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def generate_answer(context, question):
    # 베이스라인 평가를 위한 프롬프트 구성
    prompt = f"""Read the following Text and answer the Question. If you cannot find the Answer to the Question in the Text, answer with 'I don't know'.

[Text]
{context}

[Question]
{question}

[Answer]
"""
    
    messages = [
        {"role": "system", "content": "You are an AI assistant that accurately extracts information from the given Text."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(device)
    
    # 추론 실행 (최대 생성 토큰 제한으로 속도 및 OOM 방지)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.2, # 사실 기반 추출이므로 낮은 temperature 설정
            do_sample=False
        )
        
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()

# 4. SAMPLE_SIZE개 샘플에 대한 추론 및 통계 계산 루프
samples = []
total_correct = 0
answerable_correct = 0
total_answerable = 0
unanswerable_correct = 0
total_unanswerable = 0

for data in tqdm(sample_dataset, desc="Evaluating"):
    context = data['context']
    question = data['question']
    
    # SQuAD v2.0은 정답이 없는 경우 answers['text']가 빈 배열입니다.
    true_answers = data['answers']['text']
    is_impossible = not true_answers
    if not is_impossible: continue # only use unanswerable cases

    model_answer = generate_answer(context, question)
    model_answer_normalized = normalize_text(model_answer)
    
    # 통계 업데이트
    if is_impossible:
        total_unanswerable += 1
        # 답변 불가능 질문에 "i don't know"가 포함되어 있으면 정답으로 처리
        if normalize_text("i don't know") in model_answer_normalized:
            unanswerable_correct += 1
    else:
        total_answerable += 1
        # 답변 가능 질문에 모델 답변이 정답 목록에 있으면 정답으로 처리
        true_answers_normalized = [normalize_text(ans) for ans in true_answers]
        if model_answer_normalized in true_answers_normalized:
            answerable_correct += 1

    # 첫 3개 샘플 저장
    if len(samples) < 3:
        samples.append({
            "question": question,
            "true_answers": true_answers,
            "model_answer": model_answer,
            "is_impossible": is_impossible
        })

# 결과 확인 (저장된 3개 샘플 출력)
for sample in samples:
    print(f"\n[Q] {sample['question']}")
    print(f"정답: {sample['true_answers']} (대답 불가: {sample['is_impossible']})")
    print(f"모델 답변: {sample['model_answer']}")
    print("-" * 50)

# 5. 최종 통계 출력
total_questions = total_answerable + total_unanswerable
total_correct = answerable_correct + unanswerable_correct

print("\n--- 통계 ---")
print(f"전체 정답률: {total_correct / total_questions:.2%} ({total_correct}/{total_questions})")
if total_answerable > 0:
    print(f"답변 가능 질문 정답률: {answerable_correct / total_answerable:.2%} ({answerable_correct}/{total_answerable})")
else:
    print("답변 가능 질문이 없습니다.")
if total_unanswerable > 0:
    print(f"답변 불가능 질문 정답률: {unanswerable_correct / total_unanswerable:.2%} ({unanswerable_correct}/{total_unanswerable})")
else:
    print("답변 불가능 질문이 없습니다.")
