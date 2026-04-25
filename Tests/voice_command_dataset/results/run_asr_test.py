"""
对 manifest.jsonl 中的音频批量调用 OpenAI ASR (whisper-1)，
计算字词级 WER 与近匹配成功率，并写出 CSV / JSONL.

依赖：同目录上级 Tests/.env 中配置 OPENAI_API_KEY（与 generate_tts_dataset 一致）.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

# 与 generate_tts_dataset / build_manifest 一致
_TESTS_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = Path(__file__).resolve().parent

_DEFAULT_ASR_MODEL = "whisper-1"


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def levenshtein_distance(a: list[str], b: list[str]) -> int:
    """词级 Levenshtein，仅用两行滚动数组。"""
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev = cur
    return prev[n]


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    return levenshtein_distance(ref_words, hyp_words) / len(ref_words)


def exact_or_near_match(
    reference: str,
    hypothesis: str,
    wer_threshold: float = 0.25,
) -> bool:
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    if ref_norm == hyp_norm:
        return True

    return word_error_rate(reference, hypothesis) <= wer_threshold


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"未找到 {path}，请先运行 build_manifest.py")

    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def resolve_audio_path(item: dict[str, Any], dataset_dir: Path) -> Path:
    """
    优先使用 manifest 中的 audio_relpath（相对 dataset_dir，可移植）；
    再尝试已存在的 audio_path 绝对路径。
    """
    rel = item.get("audio_relpath")
    if rel:
        p = (dataset_dir / rel).resolve()
        if p.is_file():
            return p
    ap = item.get("audio_path")
    if ap:
        p2 = Path(ap)
        if p2.is_file():
            return p2.resolve()
    if rel:
        p = (dataset_dir / rel).resolve()
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"找不到音频: rel={rel!r} abspath={ap!r}（请确认已生成 wav 或路径未移动）",
    )


def whisper_language(item: dict[str, Any]) -> str:
    """Whisper 使用 ISO-639-1 两字母，如 en。"""
    raw = (item.get("language") or "en").strip().lower()
    if not raw:
        return "en"
    return raw.split("-")[0][:2]


def transcribe_audio(
    client: OpenAI,
    audio_path: Path,
    language: str,
    model: str,
) -> tuple[str, float, str]:
    """返回 (asr_text, latency_sec, error_message)。"""
    t0: float | None = None
    try:
        t0 = time.perf_counter()
        with audio_path.open("rb") as audio_file:
            result = client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                language=language,
            )
        t1 = time.perf_counter()
        return result.text.strip(), t1 - t0, ""
    except Exception as e:  # noqa: BLE001
        t1 = time.perf_counter()
        latency = t1 - t0 if t0 is not None else 0.0
        return "", latency, str(e)


def _result_row(
    item: dict[str, Any],
    audio_path_display: str,
    reference_text: str,
    asr_text: str,
    asr_latency: float,
    wer: float,
    asr_success: bool,
    error: str,
) -> dict[str, Any]:
    return {
        "uid": item.get("uid", ""),
        "sample_id": item.get("sample_id", ""),
        "category": item.get("category", ""),
        "voice": item.get("voice", ""),
        "audio_path": audio_path_display,
        "reference_text": reference_text,
        "asr_text": asr_text,
        "asr_latency": round(asr_latency, 6),
        "wer": round(wer, 6),
        "asr_success": asr_success,
        "expected_route": item.get("expected_route", ""),
        "expected_valid": item.get("expected_valid", ""),
        "expected_intent": item.get("expected_intent", ""),
        "expected_steps": item.get("expected_steps", ""),
        "error": error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="对 manifest 中音频进行 ASR 评估，输出 results/asr_results.*",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
        help="数据集根目录（默认：本脚本所在目录）",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="manifest.jsonl 路径（默认：<dataset-dir>/manifest.jsonl）",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="结果输出目录（默认：<dataset-dir>/results）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="只处理前 N 条，0 表示全部",
    )
    parser.add_argument(
        "--model",
        default=_DEFAULT_ASR_MODEL,
        help=f"ASR 模型名（默认: {_DEFAULT_ASR_MODEL}）",
    )
    parser.add_argument(
        "--wer-threshold",
        type=float,
        default=0.25,
        help="标为 asr_success 的 WER 上限（默认 0.25）",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    manifest_path = (args.manifest or (dataset_dir / "manifest.jsonl")).resolve()
    out_dir = (args.out_dir or (dataset_dir / "results")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "asr_results.csv"
    out_jsonl = out_dir / "asr_results.jsonl"

    load_dotenv(_TESTS_DIR / ".env")
    client = OpenAI()

    items = load_manifest(manifest_path)
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    print(
        f"[Info] 从 {manifest_path} 加载 {len(items)} 条；输出 -> {out_dir}",
        flush=True,
    )

    fieldnames = [
        "uid",
        "sample_id",
        "category",
        "voice",
        "audio_path",
        "reference_text",
        "asr_text",
        "asr_latency",
        "wer",
        "asr_success",
        "expected_route",
        "expected_valid",
        "expected_intent",
        "expected_steps",
        "error",
    ]

    n_ok = 0
    n_err = 0
    sum_latency_ok = 0.0
    sum_wer_ok = 0.0
    n_asr_pass = 0

    with out_csv.open("w", encoding="utf-8", newline="") as csv_f, out_jsonl.open(
        "w",
        encoding="utf-8",
    ) as jsonl_f:
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, item in enumerate(items, start=1):
            reference_text = str(item.get("reference_text", ""))
            language = whisper_language(item)

            try:
                ap = resolve_audio_path(item, dataset_dir)
            except FileNotFoundError as e:
                n_err += 1
                err_msg = str(e)
                row = _result_row(
                    item,
                    str(item.get("audio_path", item.get("audio_relpath", ""))),
                    reference_text,
                    "",
                    0.0,
                    1.0,
                    False,
                    err_msg,
                )
                print(f"[{idx}/{len(items)}] [Error] {err_msg}", flush=True)
                writer.writerow(row)
                jsonl_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            print(f"[{idx}/{len(items)}] ASR: {ap}", flush=True)
            asr_text, asr_latency, error = transcribe_audio(
                client, ap, language, args.model,
            )

            if error:
                n_err += 1
                wer = 1.0
                asr_success = False
                print(f"  [Error] {error}", flush=True)
            else:
                n_ok += 1
                sum_latency_ok += asr_latency
                wer = word_error_rate(reference_text, asr_text)
                sum_wer_ok += wer
                asr_success = exact_or_near_match(
                    reference_text,
                    asr_text,
                    args.wer_threshold,
                )
                if asr_success:
                    n_asr_pass += 1
                print(f"  REF: {reference_text}", flush=True)
                print(f"  ASR: {asr_text}", flush=True)
                print(
                    f"  WER: {wer:.3f} | latency: {asr_latency:.3f}s | success={asr_success}",
                    flush=True,
                )

            row = _result_row(
                item,
                str(ap),
                reference_text,
                asr_text,
                asr_latency,
                wer,
                asr_success,
                error,
            )
            writer.writerow(row)
            jsonl_f.write(json.dumps(row, ensure_ascii=False) + "\n")

    mean_lat = (sum_latency_ok / n_ok) if n_ok else 0.0
    mean_wer = (sum_wer_ok / n_ok) if n_ok else 0.0
    print(
        f"[Summary] 共 {len(items)} 条；ASR 请求成功 {n_ok}；失败/跳过 {n_err}；"
        f"近匹配 {n_asr_pass} 条；成功样本平均 WER {mean_wer:.4f}、平均延迟 {mean_lat:.3f}s",
        flush=True,
    )
    print(f"[Done] {out_csv}\n[Done] {out_jsonl}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
