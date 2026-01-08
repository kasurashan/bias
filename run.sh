# # replace 모드(기존 N1/N2 항목을 같은 개수로 교체)
python 01_entity.py --input input.csv --output out_replace.jsonl --mode replace --model gpt-4.1
# # # append 모드(기존 유지 + 2개씩 추가) / k=1~2 권장 / k=3부터는 개수가 정확히 안 지켜짐
python 01_entity.py --input input.csv --output out_append.jsonl --mode append --append-k 2 --model gpt-4.1
# # # original 모드(그대로 내보내기)
python 01_entity.py --input input.csv --output out_original.jsonl --mode original --model gpt-4.1



python 02_amb_context.py --input out_append.jsonl --output template_amb_context_added.jsonl --model gpt-4.1

# (추천) 최소 수정(문장 어색함 개선) + amb에서만 augmented 반영
python 03_generate_bsd.py --input template_amb_context_added.jsonl --output final_data.jsonl --modes amb,dis --paraphrase off --model gpt-4.1

# # 더 적극 패러프레이즈
# python 03_generate_bsd.py --input templates.jsonl --output out.jsonl --modes amb,dis --paraphrase on --model gpt-4.1

# # 중간에 끊겼으면 이어서
# python 03_generate_bsd.py --input templates.jsonl --output out.jsonl --modes amb,dis --paraphrase off --resume --model gpt-4.1
