"""
Locol
从 prompts.csv 与 audio/ 中已存在的 wav 构建 manifest.jsonl。

每一行一条 JSON 记录，对应 (样本 id × 声线) 中「文件已存在」的组合，
把 CSV 中的期望意图/步骤等元数据与音频路径写在一起，便于 ASR 或端到端流程消费。

与 generate_tts_dataset.py 使用相同的 VOICES 与目录约定；生成全部音频后再运行本脚本。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# 与 generate_tts_dataset.py 保持一致（本脚本所在目录为数据集根）
DATASET_DIR = Path(__file__).resolve().parent

VOICES = ["alloy", "nova", "echo"]

REQUIRED_COLUMNS = (
    "id",
    "category",
    "text",
    "language",
    "expected_route",
    "expected_valid",
    "expected_intent",
    "expected_steps",
)


def _parse_bool(s: str) -> bool:
    return s.strip().lower() in ("1", "true", "yes", "y")


def _check_csv_columns(fieldnames: list[str] | None) -> str | None:
    if not fieldnames:
        return "CSV 无表头"
    missing = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
    if missing:
        return f"表头缺列: {missing}"
    return None


def build_records(
    rows: list[dict[str, str]],
    missing_out: list[str],
    dataset_dir: Path,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row in rows:
        sid = (row.get("id") or "").strip()
        if not sid:
            continue
        for voice in VOICES:
            rel_wav = Path("audio") / f"{sid}_{voice}.wav"
            audio_path = dataset_dir / rel_wav
            if not audio_path.is_file():
                missing_out.append(rel_wav.as_posix())
                continue
            uid = f"{sid}_{voice}"
            item: dict[str, object] = {
                "uid": uid,
                "sample_id": sid,
                "category": row["category"],
                "voice": voice,
                "reference_text": row["text"],
                "language": row["language"],
                "expected_route": row["expected_route"],
                "expected_valid": _parse_bool(row["expected_valid"]),
                "expected_intent": row["expected_intent"],
                "expected_steps": row["expected_steps"],
                # 相对 DATASET_DIR，便于整包移动后仍可用
                "audio_relpath": rel_wav.as_posix(),
                "audio_path": audio_path.resolve().as_posix(),
            }
            records.append(item)
    # 稳定顺序：先 sample_id 再 voice
    records.sort(key=lambda r: (str(r["sample_id"]), str(r["voice"])))
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="从 prompts + audio 生成 manifest.jsonl")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="数据集根目录（默认：本脚本所在目录，与 generate_tts_dataset 一致）",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="manifest 输出路径（默认：<dataset-dir>/manifest.jsonl）",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="存在缺失 wav 时仍写出 manifest 并以 0 退出（默认：任一条缺失则非 0）",
    )
    args = parser.parse_args()

    dataset_dir = (args.dataset_dir or DATASET_DIR).resolve()
    prompts_csv = dataset_dir / "prompts.csv"
    out_path = (args.out if args.out is not None else dataset_dir / "manifest.jsonl").resolve()

    if not prompts_csv.is_file():
        print(f"[Error] 未找到: {prompts_csv}", file=sys.stderr)
        return 1

    with prompts_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        err = _check_csv_columns(reader.fieldnames)
        if err:
            print(f"[Error] {prompts_csv}: {err}", file=sys.stderr)
            return 1
        rows = list(reader)

    if not rows:
        print(f"[Error] {prompts_csv} 无数据行", file=sys.stderr)
        return 1

    n_ids = len([r for r in rows if (r.get("id") or "").strip()])
    expected = n_ids * len(VOICES)
    missing: list[str] = []
    records = build_records(rows, missing, dataset_dir)

    n = len(records)
    if missing and not args.allow_partial:
        print(
            f"[Error] 缺失 {len(missing)}/{expected} 个音频（已写入 0 条，未生成 manifest）。"
            f" 示例: {missing[:3]}",
            file=sys.stderr,
        )
        for m in missing[:20]:
            print(f"  [Missing] {m}", file=sys.stderr)
        if len(missing) > 20:
            print(f"  ... 另有 {len(missing) - 20} 个未列出", file=sys.stderr)
        print("[Error] 先生成全部 wav 或使用 --allow-partial 只导出已有文件", file=sys.stderr)
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for item in records:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")

    if missing and args.allow_partial:
        print(
            f"[Warning] 缺失 {len(missing)} 个文件，已部分写入 {n} 条（--allow-partial）",
            file=sys.stderr,
        )
        for m in missing[:10]:
            print(f"  [Missing] {m}", file=sys.stderr)
        if len(missing) > 10:
            print(f"  ... 另有 {len(missing) - 10} 个", file=sys.stderr)

    print(f"[Done] {out_path}：{n} 条（期望 {expected} 条）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
