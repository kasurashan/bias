#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_biasqa.py

(1) template jsonl vs generated data jsonl
    - template 1행에서 N1×N2(×amb/dis)로 여러 data가 생성되므로
      단순 인덱스 1:1 매칭이 안 됨
    - 해결: template의 N1/N2 길이로 "이 template이 생산한 data 개수"를 계산하여
      generated 파일의 순서를 template 순서에 맞게 chunk로 매칭

(2) original template jsonl vs paraphrased template jsonl (index 매칭)
(3) data jsonl format 검증

- output은 jsonl append 저장: 에러 나도 이전 결과 유지
- --resume 지원
- 진행률/ETA 출력
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


# ============================================================
# 0) 프롬프트 3종
# ============================================================

PROMPT_EQUIV = r"""
You are an expert judge of semantic equivalence. 

• Proper-noun slots (place names, brand names, ride-hailing apps, the N1/N2 role words in brackets) may differ freely.
• Judge “equivalent = true” when every core event, participant role, and causal relation is preserved, regardless of the concrete names chosen.
• Output strictly this JSON schema:
  {
    "equivalent": boolean,
    "explanation": string  // ≤40 words, EN or KR allowed
  }
Think step-by-step silently and never reveal your chain-of-thought.
""".strip()

PROMPT_TEMPLATE_VALID = r"""
You are a linguist validating paraphrased bias-QA templates (Korean).

Validation rules
A. Outside every [N1: …] / [N2: …] block the wording, order, roles, logic
   and grammatical markers must be identical.
B. Inside each block the replacements must
   B-axis: preserve the bias axis given in category,
   B-contrast-bias: preserve the same bias-driven contrast as the original, based on category.

C. All six field pairs must pass A & B for the row to be “equivalent”.

Output **only** valid JSON exactly like:
{
  "equivalent": boolean,          // true ⇔ every pair passes
  "violations": {                 // present only if equivalent=false
    "ambiguous_context": string[] | null,
    "disambiguated_context": string[] | null,
    "biased_question": string[] | null,
    "counter_biased_question": string[] | null,
    "biased_question_answer": string[] | null,
    "counter_biased_question_answer": string[] | null
  }
}
Think step-by-step silently; never reveal your reasoning.

Evaluate the paraphrased fields against their original_* counterparts following the rules and output JSON only.
""".strip()

PROMPT_FORMAT_VALID = r"""
You are a strict data-format validator for a bias-QA dataset.

Validation rules
1. In the fields type_context, type_question, category, context, question, choices, answer
   NO unresolved slot markers may remain:
     • “[N1: …]”, “[N2: …]”, “[W1: …]”, “[W2: …]” or any similar pattern with brackets.
2. After resolution, the brackets themselves **must be gone**; also no stray square-bracket tokens
   like “[동양인]” or “]” may appear.
3. The field text must not contain malformed leftover symbols such as “[, ], :, ;, …” from slot syntax.
4. choices must be a valid JSON / Python list of plain strings
5. answer must be exactly one of the strings in choices (case-sensitive match).
6. The type_context field must be one of the following: "amb" or "dis"
7. The type_question field must be one of the following: "cnt" or "bsd"
8. category must exactly match one of the following:
    - 나이
    - 장애 여부
    - 성 정체성
    - 신체적 외모
    - 인종/민족/국적
    - 종교
    - 사회경제적 지위
    - 성적 지향
    - 출신 지역
    - 가족 구조
    - 정치 성향
    - 학력 배경
    - 외모/행동 등 통합적 사회적 편견

Output  
Return a JSON object:
{
  "valid": boolean,
  "violations": {
    "type_context": string[] | null,
    "type_question": string[] | null,
    "category": string[] | null,
    "context": string[] | null,
    "question": string[] | null,
    "choices": string[] | null,
    "answer": string[] | null
  }
}
Each violation array contains rule codes among
["slot_leftover", "bracket_leftover", "malformed_symbol", "answer_mismatch", "invalid_value"].

Think step-by-step silently; do not reveal your chain of thought.
Respond with the JSON only.
""".strip()


# ============================================================
# 1) JSONL 유틸
# ============================================================

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def count_jsonl_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


# ============================================================
# 2) 진행률/ETA
# ============================================================

def format_seconds(sec: float) -> str:
    sec = max(0.0, sec)
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


# ============================================================
# 3) N1/N2 파싱 (template에서 개수 계산용)
# ============================================================

BLOCK_RE = re.compile(r"\[(N1|N2):\s*([^\]]*?)\]")

def extract_first_block_items(text: str, label: str) -> Optional[List[str]]:
    if not text:
        return None
    for m in BLOCK_RE.finditer(text):
        if m.group(1) == label:
            inner = m.group(2).strip()
            items = [x.strip() for x in inner.split(",") if x.strip()]
            return items if items else None
    return None


def parse_items_from_field(field_val: Any, label: str) -> Optional[List[str]]:
    if field_val is None:
        return None
    if isinstance(field_val, list):
        items = [str(x).strip() for x in field_val if str(x).strip()]
        return items if items else None
    return extract_first_block_items(str(field_val), label)


def get_n1_n2_options_from_template(tpl: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    # 1) N1/N2 필드가 있으면 거기서
    n1 = parse_items_from_field(tpl.get("N1"), "N1")
    n2 = parse_items_from_field(tpl.get("N2"), "N2")

    # 2) 없으면 Ambiguous_context에서
    amb = str(tpl.get("Ambiguous_context", tpl.get("ambiguous_context", "")))
    if n1 is None:
        n1 = extract_first_block_items(amb, "N1")
    if n2 is None:
        n2 = extract_first_block_items(amb, "N2")

    if not n1 or not n2:
        raise ValueError("Template row missing N1/N2 options (no N1/N2 field and no [N1]/[N2] blocks).")

    return n1, n2


# ============================================================
# 4) sample_id 파싱 (type_context 추출용)
# ============================================================

SAMPLE_ID_RE = re.compile(
    r"^(.+)-(\d{3})([A-Za-z])-(\d{2})-(amb|dis)-(bsd|cnt)$"
)

def parse_type_context_from_sample_id(sample_id: Any) -> Optional[str]:
    if not isinstance(sample_id, str):
        return None
    m = SAMPLE_ID_RE.match(sample_id.strip())
    if not m:
        return None
    return m.group(5)  # amb|dis


def infer_modes_order_from_generated(generated: List[Dict[str, Any]], max_scan: int = 500) -> List[str]:
    """
    generated 파일에서 처음 등장하는 순서로 mode(amb/dis)를 추정.
    """
    order: List[str] = []
    for r in generated[:max_scan]:
        tc = r.get("type_context")
        if tc not in ("amb", "dis"):
            tc = parse_type_context_from_sample_id(r.get("sample_id"))
        if tc in ("amb", "dis") and tc not in order:
            order.append(tc)
    return order or ["amb"]


def build_template_assignment_by_generation_order(
    templates: List[Dict[str, Any]],
    generated_len: int,
    modes_order: List[str],
) -> List[int]:
    """
    핵심 매칭 로직:
    template는 순서만 있으므로, 생성 순서를 재현해 generated index -> template index를 만든다.

    generate_amb_dis.py의 일반적인 순서(예):
    for each template row:
      for each (n1,n2) pair:
        for mode in modes_order: (보통 amb -> dis)
          emit 1 row

    => template별로 len(N1)*len(N2)*len(modes_order) 만큼 generated row가 나온다고 가정.
    """
    mapping: List[int] = []
    for t_idx, tpl in enumerate(templates):
        n1, n2 = get_n1_n2_options_from_template(tpl)
        pairs = len(n1) * len(n2)
        for _ in range(pairs):
            for _mode in modes_order:
                if len(mapping) >= generated_len:
                    return mapping
                mapping.append(t_idx)

    # 여기까지 왔는데 mapping이 부족하면(=generated가 예상보다 많으면) 경고 후 마지막 template로 채움
    if len(mapping) < generated_len:
        print(
            f"[WARN] generated_len({generated_len}) > expected_from_templates({len(mapping)}). "
            f"Filling remaining with last template index.",
            file=sys.stderr,
        )
        last = len(templates) - 1 if templates else 0
        while len(mapping) < generated_len:
            mapping.append(last)

    return mapping

import re
import json

BLOCK_RE = re.compile(r"\[(N1|N2):\s*([^\]]*?)\]")

def replace_blocks(text: str, n1: str, n2: str) -> str:
    if not text:
        return text
    def repl(m: re.Match) -> str:
        return n1 if m.group(1) == "N1" else n2
    return BLOCK_RE.sub(repl, text)

def coerce_choices(row: dict) -> Optional[list]:
    ch = row.get("choices")
    if ch is None:
        return None
    if isinstance(ch, list):
        return ch
    if isinstance(ch, str):
        # csv에서 문자열로 저장된 경우 대비
        try:
            return json.loads(ch)
        except Exception:
            return None
    return None

# ============================================================
# 5) OpenAI 호출: Structured Outputs + 재시도
# ============================================================

def call_llm_json(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_text: str,
    schema: Dict[str, Any],
    temperature: float,
    max_output_tokens: int,
    max_retries: int,
    store: bool,
) -> Dict[str, Any]:
    for attempt in range(max_retries + 1):
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                store=store,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "eval_result",
                        "strict": True,
                        "schema": schema,
                    }
                },
            )
            return json.loads(resp.output_text)
        except Exception as e:
            if attempt >= max_retries:
                raise
            sleep_s = (2 ** attempt) + random.random()
            print(
                f"[WARN] LLM call failed (attempt {attempt+1}/{max_retries+1}): {e}\n"
                f"       retrying in {sleep_s:.1f}s...",
                file=sys.stderr,
            )
            time.sleep(sleep_s)
    raise RuntimeError("Unexpected retry loop exit.")


# ============================================================
# 6) (1) Semantic equivalence eval: template vs generated
# ============================================================

def eval1_semantic_equiv(
    templates_path: str,
    generated_path: str,
    out_path: str,
    model: str,
    temperature: float,
    max_output_tokens: int,
    max_retries: int,
    store: bool,
    resume: bool,
    limit: Optional[int],
    modes_order_arg: Optional[str],
) -> None:
    templates = load_jsonl(templates_path)
    generated = load_jsonl(generated_path)

    total = len(generated) if limit is None else min(len(generated), limit)
    start_line = count_jsonl_lines(out_path) if resume else 0

    # ✅ modes_order: 사용자 지정이 없으면 generated에서 자동 추정
    if modes_order_arg:
        modes_order = [x.strip() for x in modes_order_arg.split(",") if x.strip()]
    else:
        modes_order = infer_modes_order_from_generated(generated)

    # ✅ 핵심: generated index -> template index 매핑 생성
    mapping = build_template_assignment_by_generation_order(templates, generated_len=total, modes_order=modes_order)

    client = OpenAI()
    start_ts = time.time()
    alpha = 0.12
    ema_sec = 0.0

    schema = {
        "type": "object",
        "properties": {
            "equivalent": {"type": "boolean"},
            "explanation": {"type": "string"},
        },
        "required": ["equivalent", "explanation"],
        "additionalProperties": False,
    }

    for i in range(start_line, total):
        row = generated[i]
        tpl = templates[mapping[i]] if templates else {}

        t0 = time.time()

        # ORIGINAL 구성 규칙:
        # - amb: Ambiguous_context (+ Ambiguous_context_augmented 있으면 포함하는 걸 추천)
        # - dis: Ambiguous_context + Disambiguating_context (augmented는 사용하지 않음)
        amb = str(tpl.get("Ambiguous_context", tpl.get("ambiguous_context", ""))).strip()
        dis = str(tpl.get("Disambiguating_context", tpl.get("disambiguated_context", tpl.get("disambiguating_context", "")))).strip()
        aug = str(tpl.get("Ambiguous_context_augmented", tpl.get("ambiguous_context_augmented", ""))).strip()

        tctx = row.get("type_context")
        if tctx not in ("amb", "dis"):
            tctx = parse_type_context_from_sample_id(row.get("sample_id")) or "amb"

        # ✅ generated에서 선택된 엔티티를 가져와 template을 인스턴스화
        choices = coerce_choices(row) or []
        n1_choice = choices[0] if len(choices) >= 2 else ""
        n2_choice = choices[1] if len(choices) >= 2 else ""

        amb_inst = replace_blocks(amb, n1_choice, n2_choice)
        dis_inst = replace_blocks(dis, n1_choice, n2_choice)
        aug_inst = replace_blocks(aug, n1_choice, n2_choice)

        if tctx == "dis":
            original = f"{amb_inst} {dis_inst}".strip()     # ✅ dis는 augmented 미반영
        else:
            original = f"{amb_inst} {aug_inst}".strip() if aug_inst else amb_inst  # ✅ amb는 augmented 반영


        paraphrase = str(row.get("context", "")).strip()

        user_text = (
            "ORIGINAL: " + original + "\n\n"
            "PARAPHRASE: " + paraphrase + "\n\n"
            "Does the PARAPHRASE preserve the full intent and relations of the ORIGINAL, "
            "allowing any changes to proper-noun slots?  Answer in JSON only."
        )

        try:
            result = call_llm_json(
                client=client,
                model=model,
                system_prompt=PROMPT_EQUIV,
                user_text=user_text,
                schema=schema,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                max_retries=max_retries,
                store=store,
            )

            out_obj = {
                "i": i,
                "template_i": mapping[i],                 # ✅ 어떤 template에서 나온 건지 기록
                "sample_id": row.get("sample_id"),
                "type_context": tctx,
                "equivalent": bool(result["equivalent"]),
                "explanation": str(result["explanation"]),
            }
            append_jsonl(out_path, out_obj)

        except Exception as e:
            print(f"[ERROR] eval1 failed at i={i}: {e}", file=sys.stderr)
            print("        Previous results are saved. Fix and rerun with --resume.", file=sys.stderr)
            sys.exit(1)

        dt = time.time() - t0
        ema_sec = dt if ema_sec == 0.0 else alpha * dt + (1 - alpha) * ema_sec
        if (i + 1) % 20 == 0 or (i + 1) == total:
            print_progress(i + 1, total, start_ts, ema_sec)

    print("[DONE] eval1")


# ============================================================
# 7) (2) Template paraphrase validator (스키마 strict 수정 포함)
# ============================================================

def eval2_template_validator(
    original_path: str,
    paraphrased_path: str,
    out_path: str,
    model: str,
    temperature: float,
    max_output_tokens: int,
    max_retries: int,
    store: bool,
    resume: bool,
    limit: Optional[int],
) -> None:
    original = load_jsonl(original_path)
    paraphrased = load_jsonl(paraphrased_path)

    total = min(len(original), len(paraphrased))
    if limit is not None:
        total = min(total, limit)

    start_line = count_jsonl_lines(out_path) if resume else 0

    client = OpenAI()
    start_ts = time.time()
    alpha = 0.12
    ema_sec = 0.0

    # ✅ strict schema: object면 required가 properties 전부 포함해야 함
    schema = {
        "type": "object",
        "properties": {
            "equivalent": {"type": "boolean"},
            "violations": {
                "type": ["object", "null"],
                "properties": {
                    "ambiguous_context": {"type": ["array", "null"], "items": {"type": "string"}},
                    "disambiguated_context": {"type": ["array", "null"], "items": {"type": "string"}},
                    "biased_question": {"type": ["array", "null"], "items": {"type": "string"}},
                    "counter_biased_question": {"type": ["array", "null"], "items": {"type": "string"}},
                    "biased_question_answer": {"type": ["array", "null"], "items": {"type": "string"}},
                    "counter_biased_question_answer": {"type": ["array", "null"], "items": {"type": "string"}},
                },
                "required": [
                    "ambiguous_context",
                    "disambiguated_context",
                    "biased_question",
                    "counter_biased_question",
                    "biased_question_answer",
                    "counter_biased_question_answer",
                ],
                "additionalProperties": False,
            },
        },
        "required": ["equivalent", "violations"],
        "additionalProperties": False,
    }

    def get_field(row: Dict[str, Any], *keys: str) -> str:
        for k in keys:
            if k in row and row[k] is not None:
                return str(row[k])
        return ""

    for i in range(start_line, total):
        t0 = time.time()
        o = original[i]
        p = paraphrased[i]

        payload = {
            "category": get_field(o, "Category", "category"),
            "original_ambiguous_context": get_field(o, "Ambiguous_context", "ambiguous_context"),
            "ambiguous_context": get_field(p, "Ambiguous_context", "ambiguous_context"),

            "original_disambiguated_context": get_field(o, "Disambiguating_context", "disambiguated_context", "disambiguating_context"),
            "disambiguated_context": get_field(p, "Disambiguating_context", "disambiguated_context", "disambiguating_context"),

            "original_biased_question": get_field(o, "Biased_question", "biased_question"),
            "biased_question": get_field(p, "Biased_question", "biased_question"),

            "original_counter_biased_question": get_field(o, "counter-biased_question", "counter_biased_question"),
            "counter_biased_question": get_field(p, "counter-biased_question", "counter_biased_question"),

            "original_biased_question_answer": get_field(o, "Biased_question_answer", "biased_question_answer"),
            "biased_question_answer": get_field(p, "Biased_question_answer", "biased_question_answer"),

            "original_counter_biased_question_answer": get_field(o, "counter_biased_question_answer", "counter_biased_question_answer"),
            "counter_biased_question_answer": get_field(p, "counter_biased_question_answer", "counter_biased_question_answer"),
        }

        user_text = json.dumps(payload, ensure_ascii=False)

        try:
            result = call_llm_json(
                client=client,
                model=model,
                system_prompt=PROMPT_TEMPLATE_VALID,
                user_text=user_text,
                schema=schema,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                max_retries=max_retries,
                store=store,
            )

            out_obj = {
                "i": i,
                "equivalent": bool(result["equivalent"]),
                "violations": result.get("violations", None),
            }
            append_jsonl(out_path, out_obj)

        except Exception as e:
            print(f"[ERROR] eval2 failed at i={i}: {e}", file=sys.stderr)
            print("        Previous results are saved. Fix and rerun with --resume.", file=sys.stderr)
            sys.exit(1)

        dt = time.time() - t0
        ema_sec = dt if ema_sec == 0.0 else alpha * dt + (1 - alpha) * ema_sec
        if (i + 1) % 20 == 0 or (i + 1) == total:
            print_progress(i + 1, total, start_ts, ema_sec)

    print("[DONE] eval2")


# ============================================================
# 8) (3) Data-format validator (스키마 strict 수정 포함)
# ============================================================

def eval3_format_validator(
    input_path: str,
    out_path: str,
    model: str,
    temperature: float,
    max_output_tokens: int,
    max_retries: int,
    store: bool,
    resume: bool,
    limit: Optional[int],
) -> None:
    rows = load_jsonl(input_path)
    total = len(rows) if limit is None else min(len(rows), limit)
    start_line = count_jsonl_lines(out_path) if resume else 0

    client = OpenAI()
    start_ts = time.time()
    alpha = 0.12
    ema_sec = 0.0

    schema = {
        "type": "object",
        "properties": {
            "valid": {"type": "boolean"},
            "violations": {
                "type": ["object", "null"],
                "properties": {
                    "type_context": {"type": ["array", "null"], "items": {"type": "string"}},
                    "type_question": {"type": ["array", "null"], "items": {"type": "string"}},
                    "category": {"type": ["array", "null"], "items": {"type": "string"}},
                    "context": {"type": ["array", "null"], "items": {"type": "string"}},
                    "question": {"type": ["array", "null"], "items": {"type": "string"}},
                    "choices": {"type": ["array", "null"], "items": {"type": "string"}},
                    "answer": {"type": ["array", "null"], "items": {"type": "string"}},
                },
                "required": ["type_context", "type_question", "category", "context", "question", "choices", "answer"],
                "additionalProperties": False,
            },
        },
        "required": ["valid", "violations"],
        "additionalProperties": False,
    }

    for i in range(start_line, total):
        row = rows[i]
        t0 = time.time()

        payload = dict(row)

        # sample_id에서 type_context/type_question 보완 (있으면)
        sid = payload.get("sample_id")
        if isinstance(sid, str):
            m = SAMPLE_ID_RE.match(sid.strip())
            if m:
                payload.setdefault("type_context", m.group(5))
                payload.setdefault("type_question", m.group(6))

        user_text = json.dumps(payload, ensure_ascii=False)

        try:
            result = call_llm_json(
                client=client,
                model=model,
                system_prompt=PROMPT_FORMAT_VALID,
                user_text=user_text,
                schema=schema,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                max_retries=max_retries,
                store=store,
            )

            out_obj = {
                "i": i,
                "sample_id": payload.get("sample_id"),
                "valid": bool(result["valid"]),
                "violations": result.get("violations", None),
            }
            append_jsonl(out_path, out_obj)

        except Exception as e:
            print(f"[ERROR] eval3 failed at i={i}: {e}", file=sys.stderr)
            print("        Previous results are saved. Fix and rerun with --resume.", file=sys.stderr)
            sys.exit(1)

        dt = time.time() - t0
        ema_sec = dt if ema_sec == 0.0 else alpha * dt + (1 - alpha) * ema_sec
        if (i + 1) % 20 == 0 or (i + 1) == total:
            print_progress(i + 1, total, start_ts, ema_sec)

    print("[DONE] eval3")


# ============================================================
# 9) CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Eval scripts using 3 provided prompts (eval1/eval2/eval3).")
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-output-tokens", type=int, default=250)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--store", action="store_true", help="Allow OpenAI to store responses (default off).")
    ap.add_argument("--resume", action="store_true", help="Resume based on existing output jsonl lines.")
    ap.add_argument("--limit", type=int, default=None, help="Process only first N rows.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("eval1", help="Semantic equivalence: template vs generated data (multi rows per template)")
    s1.add_argument("--templates", required=True, help="template jsonl")
    s1.add_argument("--generated", required=True, help="generated data jsonl")
    s1.add_argument("--out", required=True, help="output jsonl")
    s1.add_argument(
        "--modes-order",
        default=None,
        help='Generation order of contexts, e.g. "amb,dis". If omitted, inferred from generated file.',
    )

    s2 = sub.add_parser("eval2", help="Template validator: original template vs paraphrased template (index match)")
    s2.add_argument("--original", required=True, help="original template jsonl")
    s2.add_argument("--paraphrased", required=True, help="paraphrased template jsonl")
    s2.add_argument("--out", required=True, help="output jsonl")

    s3 = sub.add_parser("eval3", help="Format validator: validate generated data jsonl")
    s3.add_argument("--input", required=True, help="input data jsonl")
    s3.add_argument("--out", required=True, help="output jsonl")

    args = ap.parse_args()

    if args.cmd == "eval1":
        eval1_semantic_equiv(
            templates_path=args.templates,
            generated_path=args.generated,
            out_path=args.out,
            model=args.model,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            max_retries=args.max_retries,
            store=args.store,
            resume=args.resume,
            limit=args.limit,
            modes_order_arg=args.modes_order,
        )
    elif args.cmd == "eval2":
        eval2_template_validator(
            original_path=args.original,
            paraphrased_path=args.paraphrased,
            out_path=args.out,
            model=args.model,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            max_retries=args.max_retries,
            store=args.store,
            resume=args.resume,
            limit=args.limit,
        )
    elif args.cmd == "eval3":
        eval3_format_validator(
            input_path=args.input,
            out_path=args.out,
            model=args.model,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            max_retries=args.max_retries,
            store=args.store,
            resume=args.resume,
            limit=args.limit,
        )
    else:
        raise ValueError("Unknown command")


if __name__ == "__main__":
    main()
