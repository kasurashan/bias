#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSV/JSONL 템플릿 행들을 읽어서 OpenAI 모델로 N1/N2 블록을
(original / replace / append) 규칙에 따라 변환하고,
CSV 또는 JSONL로 저장합니다.

추가 기능:
- 진행률(%) / 경과 시간 / 처리 속도 / ETA(남은 시간) 표시
- 중간 실패해도 결과 파일은 계속 쌓임
- --resume 로 이어서 재시작 가능

필수:
  pip install openai
  export OPENAI_API_KEY="..."
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from typing import Dict, Iterable, List, Tuple

from openai import OpenAI


# ===== 1) "완성본 최종 프롬프트" =====
FINAL_INSTRUCTIONS = r"""
You are a linguist helping augment a Korean bias-QA dataset.
These examples are generated for research purposes to study social bias in language models.

You will receive ONE example as a JSON object (a single row) converted from tabular data (CSV-style).
The object contains at least the following keys:
• BBQ_category
• Ambiguous_context
• Disambiguating_context
• Biased_question
• Biased_question_answer
• Counter-biased_question_answer (optional)
• N1_info_translated
• N2_info_translated
(+ any other metadata keys you must copy back unchanged)

Control parameters (provided in the user message as JSON under "control"):
- mode: "original" | "replace" | "append"
- append_k: integer >= 0  (only used when mode="append")
- append_scope: "all" OR a list of field names to modify

Definitions (STRICT)
- "Relevant field" = any field whose value contains the substring "[N1:" or "[N2:".
- "N1/N2 block" = any substring exactly of the form "[N1: ...]" or "[N2: ...]" (including the brackets).
- "Inner text" = the text inside a block after "N1:" or "N2:" and before "]".
- "Items" = list elements inside a block, separated by a delimiter (comma/semicolon/etc.).
- Let m = number_of_items(original_block_inner_text).

Delimiter rules (STRICT)
- Preserve the exact delimiter characters AND the exact spacing around the delimiter when it exists in the original block.
- If the original block has NO delimiter because it contains only 1 item (m=1),
  then you MUST append new items using delimiter ", " (comma + single space) after the original item.
  Do NOT modify the original item; only insert ", " after it and then the new items.
- Never change commas to semicolons or vice versa. Never change spacing around delimiters.

Task (do exactly one row)
0) Read control parameters. If missing, default to:
   mode="replace", append_k=0, append_scope="all".

Scope rules
- If append_scope == "all": apply to every relevant field.
- If append_scope is a list: apply ONLY to fields whose key is in that list (and only if they are relevant fields).
- Fields outside scope MUST remain identical to the input.

Mode A: mode == "original"
- Return the input row EXACTLY as-is.
- Do NOT change any field, including fields containing [N1: ...] or [N2: ...].

Mode B: mode == "replace"
1) In every relevant field within scope, replace the INNER TEXT of every [N1: …] and [N2: …] block.
2) The replacement MUST contain the SAME number of items as the original block (item_count(output)=m).
3) Do NOT reuse any of the original item expressions (no exact reuse).
4) Keep the original sentence structure, spacing, particles, punctuation, and grammar OUTSIDE the brackets unchanged.
5) Preserve the delimiter pattern inside the brackets unchanged (same delimiter characters + same spacing).

Mode C: mode == "append"
Hard constraints (append mode)
1) In every relevant field within scope:
   - You MUST keep every original item EXACTLY (character-for-character) and in the same order.
   - The output block's inner text MUST begin with the original inner text unchanged as a prefix.
   - Append EXACTLY append_k NEW items to the END of every [N1: …] block and every [N2: …] block.
   - append_k is the number of NEW items to add, NOT the final total.
   - Therefore: item_count(output) MUST equal (m + append_k).
   - Never cap, shorten, truncate, or rewrite lists. Never interpret append_k as the final item count.
2) Appended items must be new expressions not appearing anywhere in the input row.
3) Use the SAME delimiter characters and spacing pattern that already separates items in that block.
   If the original block had no delimiter (m=1), use ", " (comma + single space) for all appended items.

Social role & bias-axis constraints (apply to replace/append)
- Use BBQ_category + N1_info_translated / N2_info_translated to infer the social roles for N1 and N2.
- Do NOT shift the bias axis (e.g., if the category is gender/sex identity, keep that axis; do not change to age, education, wealth, etc.).
- If N1 and N2 represent contrasting groups, keep a realistic contrast on the SAME axis.
- Avoid introducing extra axes unless they are already implied by the original roles.
- Keep the replacements/appenditions natural in Korean and consistent with particles/syntax outside the brackets.

Self-check before output (MANDATORY; do not print the check)
- For every [N1: ...] and [N2: ...] block you modified:
  1) Verify original items are preserved exactly and appear first (append mode).
  2) Verify item_count(output) == m (replace mode) OR m + append_k (append mode).
  3) Verify delimiter characters and spacing are preserved (or ", " used only when original had no delimiter).
  4) Verify only relevant fields within scope changed; all other keys/values are identical to input.
- If any check fails, FIX it before returning.

Output (STRICT)
- Return ONE JSON object only.
- Use EXACTLY the same keys (column names) as the input row.
- No explanations. No bullet points. Output only the valid JSON object.

Now process the following input row:
""".strip()


# ===== 유틸 =====

def read_rows_from_csv(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """CSV 전체를 (fieldnames, rows)로 읽습니다. CSV는 모든 값이 문자열로 들어옵니다."""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def iter_rows_from_jsonl(path: str) -> Iterable[Dict[str, object]]:
    """JSONL(한 줄당 JSON 객체) 파일을 row generator로 읽습니다."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def count_existing_outputs(path: str, fmt: str) -> int:
    """--resume 용: 이미 저장된 output 행 개수를 셉니다."""
    if not os.path.exists(path):
        return 0

    if fmt == "jsonl":
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f if _.strip())
    elif fmt == "csv":
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
            return max(0, len(rows) - 1)  # header 제외
    else:
        raise ValueError(f"Unknown output format: {fmt}")


def build_dynamic_schema_from_keys(keys: List[str]) -> Dict:
    """Structured Outputs용 JSON Schema를 '현재 row의 키들'로 동적으로 만듭니다."""
    return {
        "type": "object",
        "properties": {k: {"type": "string"} for k in keys},
        "required": keys,
        "additionalProperties": False,
    }


def format_seconds(sec: float) -> str:
    """초를 00:00:00 형태로 표시."""
    if sec < 0:
        sec = 0
    sec_int = int(sec)
    h = sec_int // 3600
    m = (sec_int % 3600) // 60
    s = sec_int % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def print_progress(processed: int, total: int, start_ts: float, ema_sec_per_row: float) -> None:
    """진행률/속도/ETA 출력."""
    now = time.time()
    elapsed = now - start_ts
    pct = (processed / total * 100.0) if total else 100.0

    # 평균 속도(전체 평균)
    avg_sec = (elapsed / processed) if processed > 0 else 0.0
    avg_rate = (processed / elapsed) if elapsed > 0 else 0.0

    remaining = max(0, total - processed)
    # ETA는 EMA(지수이동평균) 기반으로 더 안정적
    eta = ema_sec_per_row * remaining if processed > 0 else 0.0

    msg = (
        f"[PROGRESS] {processed}/{total} ({pct:.1f}%) | "
        f"elapsed {format_seconds(elapsed)} | "
        f"rate {avg_rate:.3f} rows/s | "
        f"avg {avg_sec:.2f}s/row | "
        f"ETA {format_seconds(eta)}"
    )
    print(msg)


def call_model_with_retries(
    client: OpenAI,
    model: str,
    instructions: str,
    row_obj: Dict[str, str],
    control: Dict,
    max_output_tokens: int,
    temperature: float,
    max_retries: int,
    store: bool,
) -> Dict[str, str]:
    """Responses API 호출 + 재시도."""
    keys = list(row_obj.keys())

    user_payload = {"control": control, "input_row": row_obj}
    user_text = json.dumps(user_payload, ensure_ascii=False)

    schema = build_dynamic_schema_from_keys(keys)

    for attempt in range(max_retries + 1):
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": user_text},
                ],
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                store=store,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "biasqa_row",
                        "strict": True,
                        "schema": schema,
                    }
                },
            )

            out_text = resp.output_text.strip()
            out_obj = json.loads(out_text)

            if set(out_obj.keys()) != set(keys):
                raise ValueError("Model output keys differ from input keys.")
            return out_obj

        except Exception as e:
            if attempt >= max_retries:
                raise

            sleep_s = (2 ** attempt) + random.random()
            print(
                f"[WARN] API call failed (attempt {attempt+1}/{max_retries+1}): {e}\n"
                f"       retrying in {sleep_s:.1f}s...",
                file=sys.stderr,
            )
            time.sleep(sleep_s)

    raise RuntimeError("Unexpected retry loop exit.")


def append_jsonl(path: str, obj: Dict[str, str]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def append_csv(path: str, fieldnames: List[str], obj: Dict[str, str]) -> None:
    file_exists = os.path.exists(path)
    is_empty = (not file_exists) or (os.path.getsize(path) == 0)

    with open(path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if is_empty:
            writer.writeheader()
        writer.writerow(obj)
        f.flush()


def main():
    parser = argparse.ArgumentParser(
        description="Apply BiasQA N1/N2 augmentation prompt to CSV/JSONL rows and save outputs incrementally."
    )
    parser.add_argument("--input", required=True, help="Input file path (.csv or .jsonl)")
    parser.add_argument("--output", required=True, help="Output file path (.csv or .jsonl)")
    parser.add_argument("--output-format", choices=["csv", "jsonl"], default=None,
                        help="Output format. Default: inferred from --output extension.")
    parser.add_argument("--model", default="gpt-4.1-mini", help="Model name (default: gpt-4.1).")
    parser.add_argument("--mode", choices=["original", "replace", "append"], default="replace")
    parser.add_argument("--append-k", type=int, default=0, help="How many items to append (append mode only).")
    parser.add_argument("--append-scope", default="all",
                        help='Fields to modify: "all" or comma-separated field names.')
    parser.add_argument("--max-output-tokens", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=0.0, help="0.0 recommended for dataset consistency.")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true",
                        help="If output already exists, skip processed rows and continue.")
    parser.add_argument("--store", action="store_true",
                        help="If set, allow OpenAI to store responses (default: do NOT store).")

    # ✅ 진행률 출력 주기 옵션(딱 필요한 것만)
    parser.add_argument("--progress-every", type=int, default=10,
                        help="Print progress every N successfully processed rows (default: 10).")

    args = parser.parse_args()

    out_fmt = args.output_format
    if out_fmt is None:
        ext = os.path.splitext(args.output.lower())[1]
        if ext == ".csv":
            out_fmt = "csv"
        elif ext in [".jsonl", ".json"]:
            out_fmt = "jsonl"
        else:
            raise ValueError("Cannot infer output format. Use --output-format csv|jsonl")

    # append_scope 파싱
    if args.append_scope.strip().lower() == "all":
        append_scope: object = "all"
    else:
        append_scope = [x.strip() for x in args.append_scope.split(",") if x.strip()]

    control = {
        "mode": args.mode,
        "append_k": max(0, args.append_k),
        "append_scope": append_scope,
    }

    client = OpenAI()  # OPENAI_API_KEY 환경변수 사용

    # ✅ 진행률/ETA 계산용 변수
    start_ts = time.time()
    alpha = 0.12  # EMA(지수이동평균) 업데이트 강도 (0~1). 값이 클수록 최근 값 반영↑
    ema_sec_per_row = 0.0
    processed_success = 0

    in_ext = os.path.splitext(args.input.lower())[1]

    if in_ext == ".csv":
        fieldnames, rows = read_rows_from_csv(args.input)
        total = len(rows)

        start_idx = 0
        if args.resume:
            done = count_existing_outputs(args.output, out_fmt)
            start_idx = min(done, total)
            print(f"[INFO] resume enabled: skipping first {start_idx} rows (already in output).")

        # ✅ 이미 처리된 건 제외한 총량 기준으로 ETA를 보고 싶으면:
        # total_effective = total - start_idx
        # 여기서는 사용자 체감상 전체 기준이 더 직관적이라 total 그대로 사용

        for i in range(start_idx, total):
            row = rows[i]

            t0 = time.time()
            try:
                out_obj = call_model_with_retries(
                    client=client,
                    model=args.model,
                    instructions=FINAL_INSTRUCTIONS,
                    row_obj=row,
                    control=control,
                    max_output_tokens=args.max_output_tokens,
                    temperature=args.temperature,
                    max_retries=args.max_retries,
                    store=args.store,
                )

                if out_fmt == "jsonl":
                    append_jsonl(args.output, out_obj)
                else:
                    append_csv(args.output, fieldnames, out_obj)

                # ✅ 성공 처리 후 진행률 갱신
                processed_success += 1
                dt = time.time() - t0

                # EMA 업데이트
                if ema_sec_per_row == 0.0:
                    ema_sec_per_row = dt
                else:
                    ema_sec_per_row = alpha * dt + (1 - alpha) * ema_sec_per_row

                # progress 출력
                if (processed_success % max(1, args.progress_every) == 0) or (i + 1 == total):
                    # processed는 "전체에서 몇 개 끝났는가"가 직관적이므로 (i+1) 사용
                    print_progress(processed=i + 1, total=total, start_ts=start_ts, ema_sec_per_row=ema_sec_per_row)

            except Exception as e:
                print(f"[ERROR] row {i} failed: {e}", file=sys.stderr)
                print("        You can fix the issue and rerun with --resume.", file=sys.stderr)
                break

    elif in_ext in [".jsonl", ".json"]:
        rows_iter = list(iter_rows_from_jsonl(args.input))
        total = len(rows_iter)
        fieldnames = list(rows_iter[0].keys()) if total > 0 else []

        start_idx = 0
        if args.resume:
            done = count_existing_outputs(args.output, out_fmt)
            start_idx = min(done, total)
            print(f"[INFO] resume enabled: skipping first {start_idx} rows (already in output).")

        for i in range(start_idx, total):
            row_any = rows_iter[i]
            row = {k: "" if row_any.get(k) is None else str(row_any.get(k)) for k in row_any.keys()}

            t0 = time.time()
            try:
                out_obj = call_model_with_retries(
                    client=client,
                    model=args.model,
                    instructions=FINAL_INSTRUCTIONS,
                    row_obj=row,
                    control=control,
                    max_output_tokens=args.max_output_tokens,
                    temperature=args.temperature,
                    max_retries=args.max_retries,
                    store=args.store,
                )

                if out_fmt == "jsonl":
                    append_jsonl(args.output, out_obj)
                else:
                    append_csv(args.output, fieldnames, out_obj)

                processed_success += 1
                dt = time.time() - t0

                if ema_sec_per_row == 0.0:
                    ema_sec_per_row = dt
                else:
                    ema_sec_per_row = alpha * dt + (1 - alpha) * ema_sec_per_row

                if (processed_success % max(1, args.progress_every) == 0) or (i + 1 == total):
                    print_progress(processed=i + 1, total=total, start_ts=start_ts, ema_sec_per_row=ema_sec_per_row)

            except Exception as e:
                print(f"[ERROR] row {i} failed: {e}", file=sys.stderr)
                print("        You can fix the issue and rerun with --resume.", file=sys.stderr)
                break
    else:
        raise ValueError("Unsupported input format. Use .csv or .jsonl/.json")

    print("[DONE]")


if __name__ == "__main__":
    main()
