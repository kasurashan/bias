# template vs generated 데이터 의미동등성 평가
python evaluation.py eval1 --templates template_amb_context_added.jsonl --generated final_data.jsonl --out eval1.jsonl

# template vs paraphrase template 규칙 준수 평가
python evaluation.py eval2 --original out_append.jsonl --paraphrased template_amb_context_added.jsonl --out eval2.jsonl

# 최종 데이터 포맷 검증
python evaluation.py eval3 --input final_data.jsonl --out eval3.jsonl
