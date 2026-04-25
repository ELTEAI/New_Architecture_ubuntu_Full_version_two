from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent

VLLM_FILE = THIS_DIR / "ttft_vllm_results.jsonl"
TRANSFORMERS_FILE = THIS_DIR / "ttft_transformers_results.jsonl"
OUTPUT = THIS_DIR / "ttft_comparison_summary.txt"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def values(rows: list[dict[str, Any]], key: str) -> list[float]:
    out = []
    for r in rows:
        if r.get("error"):
            continue
        v = r.get(key)
        if v is None:
            continue
        try:
            out.append(float(v))
        except Exception:
            pass
    return out


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def summarize_backend(name: str, rows: list[dict[str, Any]]) -> list[str]:
    ttfts = values(rows, "ttft")
    totals = values(rows, "total_latency")
    tps = values(rows, "tokens_per_second_rough")
    error_count = sum(1 for r in rows if r.get("error"))

    lines = []
    lines.append(f"--- {name} ---")
    lines.append(f"N: {len(rows)}")
    lines.append(f"Errors: {error_count}")
    lines.append(f"Mean TTFT: {statistics.mean(ttfts):.4f} s")
    lines.append(f"Median TTFT: {statistics.median(ttfts):.4f} s")
    lines.append(f"P95 TTFT: {percentile(ttfts, 0.95):.4f} s")
    lines.append(f"Mean total latency: {statistics.mean(totals):.4f} s")
    lines.append(f"Median total latency: {statistics.median(totals):.4f} s")
    lines.append(f"P95 total latency: {percentile(totals, 0.95):.4f} s")
    lines.append(f"Mean tokens/s rough: {statistics.mean(tps):.2f}")
    return lines


def summarize_by_category(name: str, rows: list[dict[str, Any]]) -> list[str]:
    groups = defaultdict(list)
    for r in rows:
        groups[str(r.get("category", ""))].append(r)

    lines = []
    lines.append(f"===== {name} by category =====")
    lines.append(f"{'category':<18} n    mean_ttft  p95_ttft  mean_total")
    lines.append("-" * 60)

    for cat in sorted(groups.keys()):
        g = groups[cat]
        ttfts = values(g, "ttft")
        totals = values(g, "total_latency")
        lines.append(
            f"{cat:<18} {len(g):<4} "
            f"{statistics.mean(ttfts):9.4f} "
            f"{percentile(ttfts, 0.95):9.4f} "
            f"{statistics.mean(totals):10.4f}"
        )

    return lines


def main() -> None:
    vllm = load_jsonl(VLLM_FILE)
    trans = load_jsonl(TRANSFORMERS_FILE)

    lines = []
    lines.append("======== TTFT Comparison Summary ========")
    lines.append("")
    lines.extend(summarize_backend("vLLM", vllm))
    lines.append("")
    lines.extend(summarize_backend("Transformers", trans))
    lines.append("")

    v_ttft = values(vllm, "ttft")
    t_ttft = values(trans, "ttft")
    if v_ttft and t_ttft:
        speedup = statistics.mean(t_ttft) / statistics.mean(v_ttft)
        lines.append(f"Mean TTFT speedup (Transformers / vLLM): {speedup:.2f}x")

    lines.append("")
    lines.extend(summarize_by_category("vLLM", vllm))
    lines.append("")
    lines.extend(summarize_by_category("Transformers", trans))

    text = "\n".join(lines)
    print(text)
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"\n[Done] Written to {OUTPUT}")


if __name__ == "__main__":
    main()