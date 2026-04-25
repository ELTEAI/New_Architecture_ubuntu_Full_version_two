"""
Locol
Evaluate command-level semantic accuracy from ASR results.

Input (default):
    <voice_command_dataset>/results/asr_results.jsonl

Output (default, under this package directory):
    command_level_semantic_accuracy/semantic_accuracy_results.jsonl
    command_level_semantic_accuracy/semantic_accuracy_summary.txt

This script uses OpenAI API as a semantic evaluator to judge whether
the ASR transcript preserves the robot command intent of the reference command.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

# This folder: .../voice_command_dataset/command_level_semantic_accuracy
PKG_DIR = Path(__file__).resolve().parent
# Parent: .../voice_command_dataset (where results/asr_results.jsonl lives)
VOICE_DATASET_DIR = PKG_DIR.parent
TESTS_DIR = VOICE_DATASET_DIR.parent
DEFAULT_ENV_PATH = TESTS_DIR / ".env"

DEFAULT_INPUT = VOICE_DATASET_DIR / "results" / "asr_results.jsonl"
DEFAULT_OUTPUT = PKG_DIR / "semantic_accuracy_results.jsonl"
DEFAULT_SUMMARY = PKG_DIR / "semantic_accuracy_summary.txt"

EVAL_MODEL = "gpt-4o-mini"

# Merged from evaluate_one; reused when --skip-existing and input text unchanged
_SEMANTIC_RESULT_KEYS = (
    "semantic_match",
    "semantic_reason",
    "normalized_reference_intent",
    "normalized_asr_intent",
    "semantic_eval_latency",
    "semantic_eval_model",
)


def _eval_cache_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    """Reference + ASR + error 不变时可复用上次 LLM 判定。"""
    return (
        str(row.get("uid", "")),
        str(row.get("reference_text", "")),
        str(row.get("asr_text", "")),
        str(row.get("error", "") or ""),
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_prev_eval_by_uid(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for r in load_jsonl(path):
        uid = r.get("uid")
        if uid:
            out[str(uid)] = r
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in ("1", "true", "yes", "y")


def _norm_for_match(s: str) -> str:
    """与 WER/近匹配一致：小写、去标点、压空白。用于判定 invalid 时「表面文本是否被 ASR 保留」。"""
    t = str(s).lower().strip()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t


def _reconcile_invalid_parsed(row: dict[str, Any], parsed: dict[str, Any]) -> None:
    """
    模型在 category=invalid 时易把「仍应拒识」判成 false，但 reason 又写「未改变无效性」。

    若规范化后参考与 ASR 完全一致，则 ASR 未扭曲指令，应记为 semantic_match=true。
    """
    if str(row.get("category", "")).lower() != "invalid":
        return
    if str(row.get("error", "") or "").strip():
        return
    a = _norm_for_match(str(row.get("reference_text", "")))
    b = _norm_for_match(str(row.get("asr_text", "")))
    if not a or a != b:
        return
    if not parsed.get("semantic_match", False):
        parsed["semantic_match"] = True
        parsed["reason"] = (
            "Invalid/ambiguous command preserved: ASR matches reference after normalization."
        )


def build_eval_prompt(row: dict[str, Any]) -> str:
    reference = row.get("reference_text", "")
    asr_text = row.get("asr_text", "")
    category = row.get("category", "")
    expected_route = row.get("expected_route", "")
    expected_intent = row.get("expected_intent", "")
    expected_steps = row.get("expected_steps", "")

    return f"""
You are evaluating ASR output for a quadruped robot voice-command system.

Task:
Determine whether the ASR transcript preserves the same robot command intent as the reference command.

Important:
- Judge semantic equivalence, not exact wording.
- Minor wording differences are acceptable if the robot should execute the same command.
- Direction, action type, duration, emergency intent, and rejection intent must be preserved.
- If the ASR transcript changes left/right, forward/backward, stop/non-stop, duration, action type, or emergency intent, mark it as not semantically equivalent.
- For invalid/ambiguous commands (category invalid): set "semantic_match" to TRUE if the user would still be handled the same way
  (still unparseable, still should be rejected, or still ambiguous) — including when the text is almost the same. Only set FALSE if
  the ASR would turn a bad command into a valid executable robot command, or it clearly changes the intended rejection/stop policy.
- For invalid commands, "semantic_match" does NOT mean the command is "valid"; it only means the ASR did not wrongly fix or distort the command meaning.
- For emergency commands, the ASR transcript is correct if it still clearly means stop, halt, freeze, abort, cancel motion, or emergency stop.

Reference command:
{reference}

ASR transcript:
{asr_text}

Metadata:
category = {category}
expected_route = {expected_route}
expected_intent = {expected_intent}
expected_steps = {expected_steps}

Return only valid JSON with this schema:
{{
  "semantic_match": true or false,
  "reason": "brief reason",
  "normalized_reference_intent": "short intent phrase",
  "normalized_asr_intent": "short intent phrase"
}}
""".strip()


def evaluate_one(client: OpenAI, row: dict[str, Any], model: str) -> dict[str, Any]:
    prompt = build_eval_prompt(row)

    t0 = time.perf_counter()

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an evaluator for robot command semantic equivalence. Return JSON only. "
                    "For category=invalid, semantic_match=true means the ASR still warrants the same handling "
                    "(e.g. still reject), not that the command is a valid action."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        response_format={"type": "json_object"},
    )

    t1 = time.perf_counter()
    content = response.choices[0].message.content or "{}"

    try:
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            parsed = {"semantic_match": False, "reason": "Evaluator JSON was not an object", "normalized_reference_intent": "", "normalized_asr_intent": ""}
        else:
            _reconcile_invalid_parsed(row, parsed)
    except json.JSONDecodeError:
        parsed = {
            "semantic_match": False,
            "reason": f"Evaluator returned invalid JSON: {content}",
            "normalized_reference_intent": "",
            "normalized_asr_intent": "",
        }

    return {
        "semantic_match": normalize_bool(parsed.get("semantic_match")),
        "semantic_reason": str(parsed.get("reason", "")),
        "normalized_reference_intent": str(parsed.get("normalized_reference_intent", "")),
        "normalized_asr_intent": str(parsed.get("normalized_asr_intent", "")),
        "semantic_eval_latency": round(t1 - t0, 6),
        "semantic_eval_model": model,
    }


def summarize(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No rows.\n"

    total = len(rows)
    usable = [r for r in rows if not str(r.get("error", "") or "").strip()]
    n_usable = len(usable)

    match_count = sum(1 for r in usable if normalize_bool(r.get("semantic_match")))
    acc = match_count / n_usable if n_usable else 0.0

    lines: list[str] = []
    lines.append("======== Command-Level Semantic Accuracy Summary ========")
    lines.append(f"Total rows: {total}")
    lines.append(f"Rows without ASR/API error: {n_usable}")
    lines.append(f"Semantic matches: {match_count}/{n_usable}")
    lines.append(f"Command-level semantic accuracy: {acc * 100:.2f}%")
    lines.append("")

    # By category
    cats = sorted(set(str(r.get("category", "")) for r in usable))
    lines.append("===== By category =====")
    lines.append(f"{'category':<20} n    match%")
    lines.append("-" * 38)
    for c in cats:
        group = [r for r in usable if str(r.get("category", "")) == c]
        n = len(group)
        m = sum(1 for r in group if normalize_bool(r.get("semantic_match")))
        pct = m / n * 100 if n else 0.0
        lines.append(f"{c:<20} {n:<4} {pct:6.2f}")

    lines.append("")

    # By voice
    voices = sorted(set(str(r.get("voice", "")) for r in usable))
    lines.append("===== By voice =====")
    lines.append(f"{'voice':<20} n    match%")
    lines.append("-" * 38)
    for v in voices:
        group = [r for r in usable if str(r.get("voice", "")) == v]
        n = len(group)
        m = sum(1 for r in group if normalize_bool(r.get("semantic_match")))
        pct = m / n * 100 if n else 0.0
        lines.append(f"{v:<20} {n:<4} {pct:6.2f}")

    # List failures
    failures = [r for r in usable if not normalize_bool(r.get("semantic_match"))]
    lines.append("")
    lines.append("===== Semantic mismatch cases =====")
    if not failures:
        lines.append("None")
    else:
        for r in failures:
            lines.append(
                f"- {r.get('sample_id')} / {r.get('voice')} / {r.get('category')}: "
                f"REF='{r.get('reference_text')}' | ASR='{r.get('asr_text')}' | "
                f"reason={r.get('semantic_reason')}"
            )

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate command-level semantic accuracy.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"ASR jsonl (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output jsonl (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help=f"Summary txt (default: {DEFAULT_SUMMARY})",
    )
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument("--model", type=str, default=EVAL_MODEL)
    parser.add_argument("--limit", type=int, default=0, help="For debugging. 0 means all rows.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="If --output exists, reuse per-uid evaluation when ref/asr/error unchanged and model matches",
    )
    args = parser.parse_args()

    if args.env.exists():
        load_dotenv(args.env)
    else:
        load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print(f"[Error] OPENAI_API_KEY not found. Checked env path: {args.env}", file=sys.stderr)
        return 1

    client = OpenAI()

    if not args.input.exists():
        print(f"[Error] Input not found: {args.input}", file=sys.stderr)
        return 1

    rows = load_jsonl(args.input)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    print(f"[Info] Loaded {len(rows)} rows from {args.input}")
    print(f"[Info] Evaluator model: {args.model}")
    print(f"[Info] Default output directory: {PKG_DIR}")

    prev_by_uid = _load_prev_eval_by_uid(args.output) if args.skip_existing else {}
    if args.skip_existing and prev_by_uid:
        print(f"[Info] --skip-existing: {len(prev_by_uid)} prior uid entries in {args.output}")

    results: list[dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        ref = row.get("reference_text", "")
        hyp = row.get("asr_text", "")
        err = str(row.get("error", "") or "").strip()
        uid = str(row.get("uid") or "")

        print(f"[{idx}/{len(rows)}] {row.get('sample_id')} / {row.get('voice')}")

        if args.skip_existing and uid and uid in prev_by_uid:
            old = prev_by_uid[uid]
            if (
                _eval_cache_key(row) == _eval_cache_key(old)
                and str(old.get("semantic_eval_model", "")) == str(args.model)
            ):
                eval_result = {k: old[k] for k in _SEMANTIC_RESULT_KEYS if k in old}
                if all(k in eval_result for k in _SEMANTIC_RESULT_KEYS):
                    merged = {**row, **eval_result}
                    results.append(merged)
                    print("  [Skip] Reused prior evaluation (same uid, ref/asr/error, model)")
                    print(
                        f"  REF: {ref}\n"
                        f"  ASR: {hyp}\n"
                        f"  semantic_match={merged['semantic_match']} | reason={merged['semantic_reason']}"
                    )
                    continue

        if err:
            eval_result = {
                "semantic_match": False,
                "semantic_reason": f"Skipped due to ASR/API error: {err}",
                "normalized_reference_intent": "",
                "normalized_asr_intent": "",
                "semantic_eval_latency": 0.0,
                "semantic_eval_model": args.model,
            }
        elif not str(hyp).strip():
            eval_result = {
                "semantic_match": False,
                "semantic_reason": "Empty ASR transcript.",
                "normalized_reference_intent": "",
                "normalized_asr_intent": "",
                "semantic_eval_latency": 0.0,
                "semantic_eval_model": args.model,
            }
        else:
            try:
                eval_result = evaluate_one(client, row, args.model)
            except Exception as e:  # noqa: BLE001
                eval_result = {
                    "semantic_match": False,
                    "semantic_reason": f"Evaluator API error: {e}",
                    "normalized_reference_intent": "",
                    "normalized_asr_intent": "",
                    "semantic_eval_latency": 0.0,
                    "semantic_eval_model": args.model,
                }

        merged = {
            **row,
            **eval_result,
        }
        results.append(merged)

        print(
            f"  REF: {ref}\n"
            f"  ASR: {hyp}\n"
            f"  semantic_match={merged['semantic_match']} | reason={merged['semantic_reason']}"
        )

    write_jsonl(args.output, results)

    summary_text = summarize(results)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(summary_text, encoding="utf-8")

    print("")
    print(summary_text)
    print(f"[Done] Results written to: {args.output}")
    print(f"[Done] Summary written to: {args.summary}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())