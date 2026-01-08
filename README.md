## Setup

export OPENAI_API_KEY="YOUR_OPENAI_KEY"

pip install openai


## Pipeline Overview

### 01_entity.py
우리가 작성한 template csv 파일을 input으로 받아서
1) 내부 entity 원본 그대로를 출력하거나 (original)
2) 내부 entity를 다른 것으로 출력하거나 (replace)
3) 내부 entity 원본과 augmentation한 것을 함께 출력 (append)

output으로 나오는 new template은 jsonl or csv 형태 (선택 가능)


### 02_amb_context.py
input으로 주어지는 template에 대해 ambiguous context에 한 문장을 더 추가
(Ambiguous_context_augmented 열이 추가됨)
이는 amb와 dis 사이의 문장 길이를 최대한 맞춰주기 위해서임


### 03_generate_bsd.py
input으로 주어지는 template을 활용해 dataset 생성

- counter example (amb-cnt와 dis-cnt)은 고려하지 않음
- N1 entity들과 N2 entity들의 모든 가능한 조합들에 대해 데이터 생성

예)
템플릿이 len(N1)=2, len(N2)=3인 경우
- amb-prompt : 2*3 = 6개 데이터 생성 (amb-bsd)
- dis-prompt : 2*3 = 6개 데이터 생성 (dis-bsd)
- 총 12개 생성

paraphrase on을 하면 paraphrase까지 수행


## 참고 (KoBBQ 정의)
- biased_answer: The answer conforming to social biases
- answer: The correct answer for given context and question

따라서,
- biased_question_answer는 biased_question에 대한 "실제 정답(ground truth)"으로 취급
- biased_answer는 사회적 편견을 반영한 답변으로 별도 생성
- 일반적으로 biased question은 편견 기반 응답을 유도하는 질문이므로
  answer(정답)와 biased_answer(편향 답변)는 서로 반대가 되는 것이 자연스러움


## evaluation.py
1) original template과 augmented template 사이의 의미론적인 동등성 확인
2) template과 그것으로부터 생성된 data 사이의 의미론적인 동등성 확인
3) data에 특수 기호(슬롯 잔여 포함) 같은 것이 남아있는지, choices/answer 등이 안 맞는지 검증


## Run Scripts
bash run.sh

bash eval.sh
