from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# .../Tests/voice_command_dataset/planning_latency
BASE_DIR = Path(__file__).resolve().parent
VOICE_DATASET_DIR = BASE_DIR.parent

# 上游：plan_validity_rate/evaluate_plan_validity.py 的默认输出
DEFAULT_INPUT = VOICE_DATASET_DIR / "plan_validity_rate" / "plan_validity_results.jsonl"
# 本目录输出
DEFAULT_OUTPUT = BASE_DIR / "planning_latency_summary.txt"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * p
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def summarize_group(name: str, rows: list[dict[str, Any]]) -> list[str]:
    latencies = [
        float(r.get("planning_latency", 0.0))
        for r in rows
        if r.get("tested_for_plan_validity") is True
    ]

    lines = []
    lines.append(f"--- {name} ---")
    lines.append(f"N: {len(latencies)}")

    if not latencies:
        lines.append("No latency data.")
        return lines

    lines.append(f"Mean planning latency: {statistics.mean(latencies):.4f} s")
    lines.append(f"Median planning latency: {statistics.median(latencies):.4f} s")
    lines.append(f"P95 planning latency: {percentile(latencies, 0.95):.4f} s")
    lines.append(f"Min planning latency: {min(latencies):.4f} s")
    lines.append(f"Max planning latency: {max(latencies):.4f} s")

    if len(latencies) >= 2:
        lines.append(f"Std planning latency: {statistics.stdev(latencies):.4f} s")

    return lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description="从 plan_validity_results.jsonl 汇总 transcript→plan 的 planning_latency。",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="默认: ../plan_validity_rate/plan_validity_results.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="默认: 本目录下 planning_latency_summary.txt",
    )
    args = parser.parse_args()
    input_path: Path = args.input
    output_path: Path = args.output

    if not input_path.is_file():
        print(
            f"未找到输入: {input_path}\n"
            f"请先运行 plan_validity_rate/evaluate_plan_validity.py 或指定 --input。",
            file=sys.stderr,
        )
        return 2

    rows = load_jsonl(input_path)

    tested = [
        r for r in rows
        if r.get("tested_for_plan_validity") is True
        and str(r.get("category", "")) in {"normal_action", "combo_action"}
    ]

    lines = []
    lines.append("======== Planning Latency Summary ========")
    lines.append(f"Source file: {input_path.resolve()}")
    lines.append(f"Planner-tested rows: {len(tested)}")
    lines.append("")

    lines.extend(summarize_group("Overall", tested))
    lines.append("")

    by_cat = defaultdict(list)
    for r in tested:
        by_cat[str(r.get("category", ""))].append(r)

    for cat in sorted(by_cat.keys()):
        lines.extend(summarize_group(f"Category: {cat}", by_cat[cat]))
        lines.append("")

    text = "\n".join(lines)
    print(text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    print(f"\n[Done] Written to {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())