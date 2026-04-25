"""
Locol
Summarize ASR metrics from run_asr_test output (asr_results.jsonl or .csv).

Default input: <this_dir>/asr_results.jsonl. No pandas required.
Writes a text report to <this_dir>/asr_summary.txt (English). Stdout mirrors the report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# Script lives in .../voice_command_dataset/results/
RESULTS_DIR = Path(__file__).resolve().parent

def _as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in ("1", "true", "yes", "y")


def _as_float(x: Any) -> float:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0.0
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def _normalize_json_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **r,
            "asr_success": _as_bool(r.get("asr_success")),
            "wer": _as_float(r.get("wer")),
            "asr_latency": _as_float(r.get("asr_latency")),
        }
        for r in rows
    ]


def load_results(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        return _normalize_json_rows(load_jsonl(path))
    if suffix == ".csv":
        return _normalize_rows(load_csv(path))
    raise ValueError(f"Unsupported format: {path} (use .jsonl or .csv)")


def _normalize_rows(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in raw:
        out.append(
            {
                **r,
                "asr_success": _as_bool(r.get("asr_success")),
                "expected_valid": _as_bool(r.get("expected_valid")),
                "wer": _as_float(r.get("wer")),
                "asr_latency": _as_float(r.get("asr_latency")),
            }
        )
    return out


def _percentile_nearest_r7(xs: list[float], p: float) -> float:
    """p in [0, 100], R-7 like."""
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    if n == 1:
        return s[0]
    k = (n - 1) * (p / 100.0)
    f = math.floor(k)
    c = min(f + 1, n - 1)
    if f == c:
        return s[int(k)]
    return s[f] + (s[c] - s[f]) * (k - f)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    m = len(s) // 2
    if len(s) % 2:
        return s[m]
    return (s[m - 1] + s[m]) / 2.0


def _stdev_sample(xs: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return math.sqrt(var)


def _split_ok(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with_err: list[dict[str, Any]] = []
    transcribed: list[dict[str, Any]] = []
    for r in rows:
        err = str(r.get("error", "") or "").strip()
        if err:
            with_err.append(r)
        else:
            transcribed.append(r)
    return with_err, transcribed


def _summary_block(
    label: str,
    rows: list[dict[str, Any]],
    out: list[str],
) -> None:
    if not rows:
        out.append(f"\n--- {label} (0 rows) ---\n")
        return
    wers = [_as_float(r.get("wer")) for r in rows]
    lats = [_as_float(r.get("asr_latency")) for r in rows]
    ok_flags = [_as_bool(r.get("asr_success")) for r in rows]
    n = len(rows)
    n_pass = sum(1 for x in ok_flags if x)
    out.append(f"\n--- {label} (n={n}) ---\n")
    out.append(
        f"  Near-match success rate (asr_success):  {n_pass / n * 100:.2f}%  ({n_pass}/{n})\n"
    )
    out.append(f"  Mean WER:                  {_mean(wers):.4f}\n")
    out.append(f"  Mean latency:              {_mean(lats):.3f} s\n")
    out.append(f"  Median latency:            {_median(lats):.3f} s\n")
    out.append(f"  P95 latency:               {_percentile_nearest_r7(lats, 95):.3f} s\n")
    if n >= 2:
        out.append(f"  Latency sample stdev:      {_stdev_sample(lats):.3f} s\n")


def _table_grouped(
    title: str,
    rows: Iterable[dict[str, Any]],
    key: str,
    out: list[str],
) -> None:
    from collections import defaultdict

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        k = str(r.get(key, "")) or "(empty)"
        groups[k].append(r)

    out.append(f"\n===== {title} =====\n")
    out.append(f"{'group':<22}  n    ok%      mean_WER  mean_s   p95_s\n")
    out.append("-" * 64 + "\n")
    for gname in sorted(groups.keys()):
        g = groups[gname]
        n = len(g)
        wers = [_as_float(x.get("wer")) for x in g]
        lats = [_as_float(x.get("asr_latency")) for x in g]
        oks = sum(1 for x in g if _as_bool(x.get("asr_success")))
        mean_wer = _mean(wers)
        mean_lat = _mean(lats)
        p95 = _percentile_nearest_r7(lats, 95)
        ok_pct = oks / n * 100 if n else 0.0
        out.append(
            f"{gname:<22}  {n:<4} {ok_pct:5.1f}   {mean_wer:8.4f}  {mean_lat:6.3f}  {p95:6.3f}\n"
        )


def _per_sample_rollup(rows: list[dict[str, Any]], out: list[str]) -> None:
    from collections import defaultdict

    by_sid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if str(r.get("error", "") or "").strip():
            continue
        sid = str(r.get("sample_id", ""))
        if not sid:
            continue
        by_sid[sid].append(r)

    if not by_sid:
        return
    all_voices_ok = 0
    for _, lst in by_sid.items():
        if len(lst) < 1:
            continue
        if all(_as_bool(x.get("asr_success")) for x in lst):
            all_voices_ok += 1
    n_s = len(by_sid)
    out.append("\n----- Per sample_id (rows without API error) -----\n")
    out.append(f"  Distinct sample_id count: {n_s}\n")
    out.append(
        f"  Samples with all recorded voices asr_success: "
        f"{all_voices_ok} / {n_s}  ({(all_voices_ok / n_s * 100) if n_s else 0:.1f}%)\n"
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize ASR results (jsonl or csv). Writes asr_summary.txt to results/ by default.",
    )
    parser.add_argument(
        "input_path",
        type=Path,
        nargs="?",
        default=None,
        help="asr_results.jsonl or asr_results.csv (default: <results_dir>/asr_results.jsonl)",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Prefer asr_results.csv as default when no path given",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=f"Text report path (default: {RESULTS_DIR / 'asr_summary.txt'})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print one JSON line to stdout; also write asr_summary.json next to the report",
    )
    args = parser.parse_args()

    default_in = RESULTS_DIR / (
        "asr_results.csv" if args.csv else "asr_results.jsonl"
    )
    in_path = (args.input_path or default_in).resolve()
    if args.input_path is None and not in_path.is_file():
        alt = RESULTS_DIR / "asr_results.csv"
        if not args.csv and alt.is_file():
            in_path = alt

    out_report = (args.output or (RESULTS_DIR / "asr_summary.txt")).resolve()
    out_json = out_report.with_name("asr_summary.json")

    try:
        rows = load_results(in_path)
    except (OSError, FileNotFoundError, ValueError) as e:
        print(f"[Error] {e}", file=sys.stderr)
        return 1

    if not rows:
        print("[Error] No data rows in input", file=sys.stderr)
        return 1

    failed, ok_rows = _split_ok(rows)
    n_fail, n_all = len(failed), len(rows)

    if args.json:
        wers = [_as_float(r.get("wer")) for r in ok_rows]
        lats = [_as_float(r.get("asr_latency")) for r in ok_rows]
        oks = sum(1 for r in ok_rows if _as_bool(r.get("asr_success")))
        n_okc = len(ok_rows)
        payload = {
            "input": str(in_path),
            "total_rows": n_all,
            "api_error_rows": n_fail,
            "transcribed_rows": n_okc,
            "asr_success_rate": round((oks / n_okc) if n_okc else 0.0, 6),
            "mean_wer_transcribed": round(_mean(wers) if wers else 0.0, 6),
            "mean_latency_s": round(_mean(lats) if lats else 0.0, 6),
            "p95_latency_s": round(_percentile_nearest_r7(lats, 95) if lats else 0.0, 6),
        }
        line = json.dumps(payload, ensure_ascii=False)
        print(line)
        _write_text(out_json, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        return 0

    lines: list[str] = []
    lines.append("\n======== ASR Summary ========\n")
    lines.append(f"Source file: {in_path}\n")
    lines.append(
        f"Total rows: {n_all}  |  API/path errors: {n_fail}  |  Transcribed (no error): {len(ok_rows)}\n"
    )

    if failed:
        lines.append(
            "\n[Note] Some rows have errors. The 'transcribed only' block better reflects ASR quality;\n"
        )
        lines.append(
            "       rows with failures are counted with WER=1 etc. in the 'all rows' block.\n"
        )

    _summary_block("All rows (WER=1 when API or path error)", rows, lines)
    if failed and ok_rows:
        _summary_block("Transcribed only (empty error field)", ok_rows, lines)
    if failed and not ok_rows:
        _summary_block("Error rows only", failed, lines)

    _per_sample_rollup(rows, lines)

    _table_grouped("By category", ok_rows or rows, "category", lines)
    _table_grouped("By voice", ok_rows or rows, "voice", lines)

    lines.append(f"\n[Done] Report written to: {out_report}\n")

    text = "".join(lines)
    _write_text(out_report, text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
