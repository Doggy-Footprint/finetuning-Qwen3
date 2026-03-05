import json
import random
from datasets import load_dataset

# SQuAD v2.0 데이터 로딩
print('SQuAD v2.0 데이터 로딩')
dataset = load_dataset('squad_v2', split='train')

# 전체 문맥 풀 생성 (가짜 지문 무작위 추출용)
all_contexts = list(set(dataset['context']))

def create_rag_prompt(example):
    question = example['question']
    true_context = example['context']
    answers = example['answers']['text']

    # 정답 여부 확인
    is_answerable = len(answers) > 0
    exact_quote = answers[0] if is_answerable else 'unanswerable'

    # 가짜 지문 2개 추출
    distractors = random.sample([c for c in all_contexts if c != true_context], 2)

    # 지문 섞기
    chunks = [true_context] + distractors
    random.shuffle(chunks)

    chunk_label = ['A', 'B', 'C']
    target_index = chunks.index(true_context)
    target_chunk_label = chunk_label[target_index] if is_answerable else 'none'

    # 프롬프트 작성
    prompt = f"""Pick the most plausible context for the given question,
and then print the exact quote from the context that answers the question in JSON format. 
("target_chunk", and "exact_quote" are the keys you need to print)
If there is no quote for the question, print "target_chucnk" and "exact_quote" as "none".
[Question] {question}\n
[Context A] {chunks[0]}
[Context B] {chunks[1]}
[Context C] {chunks[2]}
"""
    
    print(prompt)
    print('---')
    
    expected_output = {
        "target_chunk": target_chunk_label,
        "exact_quote": exact_quote
    }

    return {
        "prompt": prompt,
        "expected_output": json.dumps(expected_output, ensure_ascii=False)
    }

# 샘플 데이터 변환
print('데이터 처리 중')
sample_size = 1
rag_dataset = []

for i in range(sample_size):
    rag_dataset.append(create_rag_prompt(dataset[i]))

# 저장
print('데이터 저장 중')
output_file = 'rag_squad_dataset.jsonl'
with open(output_file, 'w', encoding='utf-8') as f:
    for item in rag_dataset:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print('데이터 저장 완료')

