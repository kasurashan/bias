#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
augment_amb_sentence.py

목표:
- 입력 템플릿(row)마다 Ambiguous_context의 "마지막 문장" 흐름을 이어가는
  '중립적 추가 문장 1개'를 생성하여 Ambiguous_context_augmented 컬럼으로 저장
- 원본 row의 다른 컬럼은 "그대로 유지"
- 출력: CSV 또는 JSONL
- 중간 에러 시에도 이전 결과는 파일에 이미 저장됨(append)
- --resume 로 이어서 처리 가능
- 진행률/ETA 표시

필수:
  pip install openai
  export OPENAI_API_KEY="..."

OpenAI API:
- Responses API 사용 (client.responses.create)
- Structured Outputs(JSON Schema)로 {"Ambiguous_context_augmented": "..."} 형태 강제
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


# =========================
# 1) "최종 프롬프트" (가볍게 다듬은 버전)
#    - 원문 프롬프트의 기능은 유지하되, N1/N2 금지와 '마지막 문장 요소 반향'의 충돌을 줄임
# =========================

FINAL_PROMPT = r"""
You are a linguist helping augment a Korean bias-QA dataset.
These examples are generated for research purposes to study social bias in language models.

You will receive one example as JSON with at least:
• Category
• Ambiguous_context
• Disambiguating_context
• Biased_question
• Biased_question_answer
(+ any other metadata columns you must not change)

Task:
Generate exactly ONE extra Korean sentence that should naturally follow the FINAL sentence of Ambiguous_context.

CRITICAL: Match the register/style of Ambiguous_context exactly.
- First, infer the dominant sentence-ending style used in Ambiguous_context (e.g., 합니다체/해요체/해라체/이다체).
- Then write the new sentence using ONLY that same style.
- Do NOT switch politeness level or formality (e.g., do not change from “~합니다” to “~했다/~이다”, or vice versa).
- Prefer to mirror the FINAL sentence’s ending and pacing.

Requirements:
- Treat Ambiguous_context as one coherent scene; add ONE neutral, observer-style detail.
- Continue the flow of the FINAL sentence:
  * keep the same tense and narrative voice
  * echo (reuse or refer to) at least one core element of that FINAL sentence
    (location / action / object / situation), BUT do NOT echo any N1/N2 slot content.
  * the two sentences must read smoothly back-to-back (not abrupt).
- Choose ONE approach:
  (a preferred) a neutral observation-based fact that could logically follow, OR
  (b) overall atmosphere (lighting/sound/emotional tone) that fits the scene.
- You may begin with a transition adverb, but vary it (do not overuse the same phrase).
- MUST NOT mention Category or Disambiguating_context.
- MUST NOT reuse any content from any [N1: …] or [N2: …] slot (do not mention those entities).
- Must be grammatically complete, natural Korean, about 15–30 Korean characters in length.
  Count Korean characters (Hangul syllables) excluding spaces/punctuation.
- Must end with a period.

Output MUST be a single JSON object with exactly one key:
  "Ambiguous_context_augmented": string
This string must be ONLY the generated sentence (not the full context).

No explanations. Output only valid JSON.
""".strip()


# =========================
# 2) 입력 로드 (CSV / JSONL / JSON)
# =========================

def load_rows(path: str) -> Tuple[str, List[Dict[str, Any]], Optional[List[str]]]:
    """
    Returns:
      input_kind: "csv" | "jsonl" | "json"
      rows: list of row dicts
      fieldnames: CSV일 때 원본 컬럼 순서 (그 외 None)
    """
    ext = os.path.splitext(path.lower())[1]
    if ext == ".csv":
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            rows = [dict(r) for r in reader]
        return "csv", rows, fieldnames

    if ext == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return "jsonl", rows, None

    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return "json", obj, None
        if isinstance(obj, dict):
            return "json", [obj], None
        raise ValueError("JSON root must be object or array.")

    raise ValueError("Unsupported input format. Use .csv / .jsonl / .json")


# =========================
# 3) 출력 append (CSV / JSONL)
# =========================

def count_existing_outputs(path: str, fmt: str) -> int:
    """--resume용: 이미 생성된 output 레코드 수"""
    if not os.path.exists(path):
        return 0
    if fmt == "jsonl":
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    if fmt == "csv":
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
            return max(0, len(rows) - 1)  # header 제외
    raise ValueError("Unknown output format.")


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def append_csv(path: str, fieldnames: List[str], obj: Dict[str, Any]) -> None:
    file_exists = os.path.exists(path)
    is_empty = (not file_exists) or (os.path.getsize(path) == 0)
    with open(path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if is_empty:
            writer.writeheader()
        writer.writerow(obj)
        f.flush()


# =========================
# 4) 제약 검증 (길이/마침표/N1N2 내용 포함 여부)
# =========================

BLOCK_RE = re.compile(r"\[(N1|N2):\s*([^\]]*?)\]")

def extract_slot_items(text: str) -> List[str]:
    """
    [N1: ...], [N2: ...] 블록 안의 아이템들을 모두 추출.
    예: "[N1: 남자 중학생과, 남학생]" -> ["남자 중학생과", "남학생"]
    """
    items: List[str] = []
    if not text:
        return items
    for m in BLOCK_RE.finditer(text):
        inner = m.group(2).strip()
        for it in inner.split(","):
            it = it.strip()
            if it:
                items.append(it)
    return items


def hangul_syllable_count(s: str) -> int:
    """가-힣(한글 음절)만 카운트. 공백/구두점 제외."""
    return len(re.findall(r"[가-힣]", s))


def validate_generated_sentence(
    sentence: str,
    forbidden_items: List[str],
    min_hangul: int = 15,
    max_hangul: int = 30,
) -> None:
    """
    실패하면 ValueError.
    """
    s = (sentence or "").strip()

    if not s.endswith("."):
        raise ValueError("Generated sentence must end with a period '.'")

    # 길이(한글 음절만)
    n = hangul_syllable_count(s)
    if n < min_hangul or n > max_hangul:
        raise ValueError(f"Hangul length out of range: {n} (expected {min_hangul}-{max_hangul})")

    # 슬롯 내용 재사용 금지(간단 substring 체크)
    # 너무 짧은 토큰은 오탐 위험이 크므로 3글자 이상만 검사
    for it in forbidden_items:
        it2 = it.strip()
        if len(it2) >= 3 and it2 in s:
            raise ValueError(f"Sentence reuses N1/N2 slot content: '{it2}'")

    # 블록 표기 자체도 나오면 안 됨
    if "[N1:" in s or "[N2:" in s:
        raise ValueError("Sentence must not contain [N1: / [N2: blocks.")


# =========================
# 5) OpenAI 호출 (Structured Outputs)
# =========================

def call_model_for_aug_sentence(
    client: OpenAI,
    model: str,
    row: Dict[str, Any],
    temperature: float,
    max_output_tokens: int,
    max_retries: int,
    store: bool,
) -> str:
    """
    모델로부터 {"Ambiguous_context_augmented": "..."} 받기.
    검증 실패 시 last_error를 payload에 추가하여 재시도.
    """
    schema = {
        "type": "object",
        "properties": {"Ambiguous_context_augmented": {"type": "string"}},
        "required": ["Ambiguous_context_augmented"],
        "additionalProperties": False,
    }

    # 금지 아이템: Ambiguous_context + Disambiguating_context 내부의 N1/N2 슬롯들
    amb_key = "Ambiguous_context" if "Ambiguous_context" in row else "ambiguous_context"
    dis_key = "Disambiguating_context" if "Disambiguating_context" in row else "disambiguated_context"

    amb_text = str(row.get(amb_key, ""))
    dis_text = str(row.get(dis_key, ""))

    forbidden_items = extract_slot_items(amb_text) + extract_slot_items(dis_text)

    payload: Dict[str, Any] = {
        "row": row,
        # 모델이 "금지 대상"을 더 명확히 이해하도록 힌트로 전달 (출력엔 영향 X)
        "constraints_hint": {
            "forbidden_slot_items": forbidden_items,
            "hangul_length_range": [15, 30],
        },
    }

    user_text = json.dumps(payload, ensure_ascii=False)

    for attempt in range(max_retries + 1):
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": FINAL_PROMPT},
                    {"role": "user", "content": user_text},
                ],
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                store=store,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "amb_aug",
                        "strict": True,
                        "schema": schema,
                    }
                },
            )

            out_obj = json.loads(resp.output_text)
            sent = out_obj["Ambiguous_context_augmented"].strip()

            # 로컬 검증
            validate_generated_sentence(sent, forbidden_items)

            return sent

        except Exception as e:
            if attempt >= max_retries:
                raise

            payload = dict(payload)
            payload["last_error"] = str(e)
            user_text = json.dumps(payload, ensure_ascii=False)

            sleep_s = (2 ** attempt) + random.random()
            print(
                f"[WARN] row augmentation failed (attempt {attempt+1}/{max_retries+1}): {e}\n"
                f"       retrying in {sleep_s:.1f}s...",
                file=sys.stderr,
            )
            time.sleep(sleep_s)

    raise RuntimeError("Unexpected retry loop exit.")


# =========================
# 6) 진행률/ETA
# =========================

def format_seconds(sec: float) -> str:
    sec = max(0, sec)
    s = int(sec)
    h = s // 3600
    m = (s % 3600) // 60
    s2 = s % 60
    return f"{h:02d}:{m:02d}:{s2:02d}"


def print_progress(done: int, total: int, start_ts: float, ema_sec: float) -> None:
    elapsed = time.time() - start_ts
    pct = (done / total * 100.0) if total else 100.0
    rate = (done / elapsed) if elapsed > 0 else 0.0
    eta = ema_sec * (total - done) if done > 0 else 0.0
    print(
        f"[PROGRESS] {done}/{total} ({pct:.1f}%) | "
        f"elapsed {format_seconds(elapsed)} | rate {rate:.3f} rows/s | ETA {format_seconds(eta)}"
    )


# =========================
# 7) main
# =========================

def main():
    ap = argparse.ArgumentParser(description="Augment Ambiguous_context with one neutral follow-up sentence.")
    ap.add_argument("--input", required=True, help="Input templates (.csv / .jsonl / .json)")
    ap.add_argument("--output", required=True, help="Output (.csv or .jsonl)")
    ap.add_argument("--output-format", choices=["csv", "jsonl"], default=None,
                    help="If omitted, inferred from output extension")
    ap.add_argument("--model", default="gpt-4.1-mini", help="Default: gpt-4.1-mini")
    ap.add_argument("--temperature", type=float, default=0.2, help="Lower is more consistent (default 0.2)")
    ap.add_argument("--max-output-tokens", type=int, default=120, help="Sentence is short; 120 is enough")
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--resume", action="store_true", help="Continue from existing output file")
    ap.add_argument("--store", action="store_true",
                    help="Allow OpenAI to store responses (default off). See data controls docs.")
    ap.add_argument("--progress-every", type=int, default=10, help="Print progress every N rows")

    args = ap.parse_args()

    out_fmt = args.output_format
    if out_fmt is None:
        ext = os.path.splitext(args.output.lower())[1]
        if ext == ".csv":
            out_fmt = "csv"
        elif ext in [".jsonl", ".json"]:
            out_fmt = "jsonl"
        else:
            raise ValueError("Cannot infer output format; use --output-format csv|jsonl")

    in_kind, rows, csv_fieldnames = load_rows(args.input)
    total = len(rows)

    # resume: output에 이미 저장된 행 수만큼 input에서 스킵
    start_idx = 0
    if args.resume:
        done = count_existing_outputs(args.output, out_fmt)
        start_idx = min(done, total)
        print(f"[INFO] resume enabled: skipping first {start_idx} rows (already in output).")

    # CSV 출력 컬럼 순서: 원본 + Ambiguous_context_augmented
    if out_fmt == "csv":
        if in_kind == "csv" and csv_fieldnames:
            out_fieldnames = list(csv_fieldnames)
        else:
            # json 입력인 경우, 첫 row의 key 순서를 기본으로
            out_fieldnames = list(rows[0].keys()) if rows else []
        if "Ambiguous_context_augmented" not in out_fieldnames:
            out_fieldnames.append("Ambiguous_context_augmented")
    else:
        out_fieldnames = None

    client = OpenAI()

    start_ts = time.time()
    alpha = 0.12
    ema_sec = 0.0
    processed = 0

    for i in range(start_idx, total):
        row = rows[i]
        t0 = time.time()

        try:
            aug_sent = call_model_for_aug_sentence(
                client=client,
                model=args.model,
                row=row,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                max_retries=args.max_retries,
                store=args.store,
            )

            out_row = dict(row)  # 원본 유지
            out_row["Ambiguous_context_augmented"] = aug_sent

            if out_fmt == "jsonl":
                append_jsonl(args.output, out_row)
            else:
                assert out_fieldnames is not None
                append_csv(args.output, out_fieldnames, out_row)

            processed += 1
            dt = time.time() - t0
            ema_sec = dt if ema_sec == 0.0 else alpha * dt + (1 - alpha) * ema_sec

            if (processed % max(1, args.progress_every) == 0) or (i + 1 == total):
                print_progress(done=i + 1, total=total, start_ts=start_ts, ema_sec=ema_sec)

        except Exception as e:
            print(f"[ERROR] row {i} failed: {e}", file=sys.stderr)
            print("        Previous results are saved. Fix and rerun with --resume.", file=sys.stderr)
            sys.exit(1)

    print("[DONE]")


if __name__ == "__main__":
    main()
