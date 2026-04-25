"""
Locol
Evaluate emergency routing from semantic_accuracy_results.jsonl (no LLM, no ASR).

Uses rule-based `emergency_router(asr_text)` to classify each row as
  predicted_route in {"emergency", "planner"}.

Input (default):
    ../command_level_semantic_accuracy/semantic_accuracy_results.jsonl

Output (default, this directory):
    emergency_routing_results.jsonl
    emergency_routing_summary.txt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# .../voice_command_dataset/emergency_routing_accuracy
PKG_DIR = Path(__file__).resolve().parent
VOICE_DATASET_DIR = PKG_DIR.parent

DEFAULT_INPUT = (
    VOICE_DATASET_DIR
    / "command_level_semantic_accuracy"
    / "semantic_accuracy_results.jsonl"
)
DEFAULT_OUTPUT = PKG_DIR / "emergency_routing_results.jsonl"
DEFAULT_SUMMARY = PKG_DIR / "emergency_routing_summary.txt"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("’", "'")
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def emergency_router(asr_text: str) -> tuple[str, str]:
    text = normalize_text(asr_text)

    # Multi-step markers indicate that "stop" may be a planned step,
    # not necessarily an emergency interrupt.
    multi_step_markers = [
        "and then",
        "then",
        "first",
        "after",
        "before",
    ]
    has_multi_step_marker = any(marker in text for marker in multi_step_markers)

    strong_emergency_patterns = [
        r"\bstop immediately\b",
        r"\bemergency stop\b",
        r"\bstop moving now\b",
        r"\bstop moving\b",
        r"\bhalt\b",
        r"\bfreeze\b",
        r"\bdo not move\b",
        r"\bdon't move\b",
        r"\bcancel all actions\b",
        r"\babort\b",
        r"\babort the current task\b",
        r"\bshut down movement\b",
        r"\bshutdown movement\b",
    ]

    for pattern in strong_emergency_patterns:
        if re.search(pattern, text):
            return "emergency", pattern

    # A short standalone stop utterance is an emergency interrupt.
    if text in {"stop", "stop now"}:
        return "emergency", "standalone_stop"

    # If the utterance is a multi-step instruction, route it to the planner.
    # Example: "walk forward for two seconds and then stop"
    if has_multi_step_marker:
        return "planner", "multi_step_instruction"

    # Remaining simple stop-like utterances are treated as emergency.
    if re.search(r"\bstop\b", text):
        return "emergency", r"\bstop\b"

    return "planner", ""


def _route_is_correct(expected_route: str, predicted_route: str) -> bool:
    """
    True iff routing decision matches ground truth for this test:

    - expected emergency -> must predict emergency
    - expected non-emergency (planner, reject, etc.) -> must NOT predict emergency
    """
    if expected_route == "emergency":
        return predicted_route == "emergency"
    return predicted_route != "emergency"


def summarize(rows: list[dict[str, Any]]) -> str:
    total = len(rows)

    emergency_rows = [r for r in rows if r.get("expected_route") == "emergency"]
    non_emergency_rows = [r for r in rows if r.get("expected_route") != "emergency"]

    tp = sum(1 for r in emergency_rows if r.get("predicted_route") == "emergency")
    fn = len(emergency_rows) - tp

    fp = sum(1 for r in non_emergency_rows if r.get("predicted_route") == "emergency")
    tn = len(non_emergency_rows) - fp

    emergency_acc = tp / len(emergency_rows) if emergency_rows else 0.0
    fnr = fn / len(emergency_rows) if emergency_rows else 0.0
    fpr = fp / len(non_emergency_rows) if non_emergency_rows else 0.0

    overall_route_acc = sum(1 for r in rows if r.get("route_correct") is True) / total if total else 0.0

    lines: list[str] = []
    lines.append("======== Emergency Routing Summary ========")
    lines.append(f"Total rows: {total}")
    lines.append(f"Ground-truth emergency rows: {len(emergency_rows)}")
    lines.append(f"Ground-truth non-emergency rows: {len(non_emergency_rows)}")
    lines.append("")
    lines.append(
        f"Emergency detection accuracy (recall on emergency set): {emergency_acc * 100:.2f}%  ({tp}/{len(emergency_rows) if emergency_rows else 0})"
    )
    lines.append(
        f"Emergency false negative rate: {fnr * 100:.2f}%  ({fn}/{len(emergency_rows) if emergency_rows else 0})"
    )
    lines.append(
        f"Emergency false positive rate: {fpr * 100:.2f}%  ({fp}/{len(non_emergency_rows) if non_emergency_rows else 0})"
    )
    lines.append(f"Overall route correctness: {overall_route_acc * 100:.2f}%")
    lines.append("")

    lines.append("===== By category =====")
    lines.append(f"{'category':<20} n    emergency_pred%   route_correct%")
    lines.append("-" * 62)

    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_cat[str(r.get("category", ""))].append(r)

    for cat in sorted(by_cat.keys()):
        group = by_cat[cat]
        n = len(group)
        pred_em = sum(1 for r in group if r.get("predicted_route") == "emergency")
        correct = sum(1 for r in group if r.get("route_correct") is True)
        lines.append(
            f"{cat:<20} {n:<4} {pred_em / n * 100:13.2f}   {correct / n * 100:13.2f}"
        )

    # Emergency-only: by voice
    lines.append("")
    lines.append("===== By voice (emergency ground-truth rows only) =====")
    em_by_voice: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in emergency_rows:
        em_by_voice[str(r.get("voice", ""))].append(r)
    if not emergency_rows:
        lines.append("(no emergency ground-truth rows)")
    else:
        lines.append(f"{'voice':<12} n    tp% (detected as emergency)")
        lines.append("-" * 42)
        for v in sorted(em_by_voice.keys()):
            g = em_by_voice[v]
            n = len(g)
            tpk = sum(1 for r in g if r.get("predicted_route") == "emergency")
            lines.append(f"{v:<12} {n:<4} {tpk / n * 100:6.1f}")

    false_negatives = [r for r in emergency_rows if r.get("predicted_route") != "emergency"]
    lines.append("")
    lines.append("===== Emergency false negatives (missed emergency) =====")
    if not false_negatives:
        lines.append("None")
    else:
        for r in false_negatives:
            lines.append(
                f"- {r.get('sample_id')} / {r.get('voice')}: "
                f"ASR='{r.get('asr_text')}' | REF='{r.get('reference_text')}'"
            )

    false_positives = [r for r in non_emergency_rows if r.get("predicted_route") == "emergency"]
    lines.append("")
    lines.append("===== Emergency false positives (planner spurious to emergency) =====")
    if not false_positives:
        lines.append("None")
    else:
        for r in false_positives:
            lines.append(
                f"- {r.get('sample_id')} / {r.get('voice')} / {r.get('category')}: "
                f"ASR='{r.get('asr_text')}' | REF='{r.get('reference_text')}' | "
                f"matched_rule={r.get('matched_rule')}"
            )

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rule-based emergency vs planner routing on ASR text from semantic_accuracy_results.jsonl",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"semantic_accuracy_results.jsonl (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Per-row results jsonl (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help=f"Text summary (default: {DEFAULT_SUMMARY})",
    )
    parser.add_argument("--limit", type=int, default=0, help="Process first N rows only; 0 = all")
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"[Error] Input not found: {args.input}", file=sys.stderr)
        return 1

    rows = load_jsonl(args.input)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    if not rows:
        print("[Error] No rows loaded.", file=sys.stderr)
        return 1

    print(f"[Info] Input:  {args.input.resolve()}")
    print(f"[Info] Output: {args.output.resolve()}")
    print(f"[Info] Loaded {len(rows)} rows")

    results: list[dict[str, Any]] = []
    for r in rows:
        asr_text = str(r.get("asr_text", "") or "")
        expected_route = str(r.get("expected_route", "") or "")

        predicted_route, matched_rule = emergency_router(asr_text)
        route_correct = _route_is_correct(expected_route, predicted_route)

        results.append(
            {
                **r,
                "predicted_route": predicted_route,
                "matched_rule": matched_rule,
                "route_correct": route_correct,
            }
        )

    write_jsonl(args.output, results)

    summary_text = summarize(results)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(summary_text, encoding="utf-8")

    print("")
    print(summary_text)
    print(f"[Done] {args.output}")
    print(f"[Done] {args.summary}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
