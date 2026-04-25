"""
Benchmark vLLM TTFT (time to first token) on ASR 文本，输入默认来自 plan_validity 产物。

默认输入（与 evaluate_plan_validity 输出一致）:
    <voice_command_dataset>/plan_validity_rate/plan_validity_results.jsonl

默认输出（本目录）:
    <voice_command_dataset>/TTFT/ttft_vllm_results.jsonl

--model / --base-url / --api-key 未指定时，从仓库内 VLA_Agent_Core/config.yaml 的 llm 段读取
（与 VLABrainPlanner 一致），可直接运行: python benchmark_vllm_ttft.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

# .../Tests/voice_command_dataset/TTFT
THIS_DIR = Path(__file__).resolve().parent
VOICE_DIR = THIS_DIR.parent

# 与 <repo>/Tests/voice_command_dataset/plan_validity_rate/plan_validity_results.jsonl 等价
DEFAULT_INPUT = VOICE_DIR / "plan_validity_rate" / "plan_validity_results.jsonl"
# 输出在 <repo>/Tests/voice_command_dataset/TTFT/
DEFAULT_OUTPUT = THIS_DIR / "ttft_vllm_results.jsonl"

# .../New_Architecture（Tests/voice_command_dataset 的上两级）
_REPO_ROOT = VOICE_DIR.parent.parent
_VLA_CONFIG = _REPO_ROOT / "VLA_Agent_Core" / "config.yaml"


def _load_vla_llm_defaults() -> dict[str, str]:
    """与 agent_planner 相同来源：VLA_Agent_Core/config.yaml 的 llm 段。"""
    out = {
        "model_name": "Qwen3.5-4B",
        "base_url": "http://127.0.0.1:8000/v1",
        "api_key": "EMPTY",
    }
    if not _VLA_CONFIG.is_file():
        return out
    try:
        import yaml  # type: ignore[import-untyped]

        with _VLA_CONFIG.open("r", encoding="utf-8") as f:
            full = yaml.safe_load(f) or {}
        llm = full.get("llm") or {}
        if isinstance(llm.get("model_name"), str):
            out["model_name"] = llm["model_name"]
        if isinstance(llm.get("base_url"), str):
            out["base_url"] = llm["base_url"]
        if isinstance(llm.get("api_key"), str):
            out["api_key"] = llm["api_key"]
    except Exception:
        pass
    return out


SYSTEM_PROMPT = """
You are VLA_Agent_Core, a high-level planning and skill orchestration agent for a robot dog.

Convert the user's instruction into a JSON object with this exact schema:
{
  "sequence_name": "short_english_name",
  "actions": [
    {"action_id": integer, "duration": number}
  ]
}

Action schema:
- Mode 1 continuous motion, duration must be 0:
  0 forward, 2 back, 3 strafe left, 4 strafe right, 5 turn left, 6 turn right
- Mode 0 emergency stop, duration must be 0:
  1 full stop / idle
- Mode 2 blocking action, duration must be an integer in [2,5]:
  7 sit, 8 stand up, 9 stretch, 10 roll over, 11 pose, 12 greeting

Return JSON only. No markdown. No explanation.
""".strip()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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


def count_rough_tokens(text: str) -> int:
    # Rough fallback, enough for relative comparison.
    return max(1, len(text.split()))


def stream_chat_completion(
    client: OpenAI,
    model: str,
    instruction: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    t_request = time.perf_counter()
    first_token_time = None
    chunks = []

    stream = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ],
    )

    for event in stream:
        delta = event.choices[0].delta.content
        if delta:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            chunks.append(delta)

    t_done = time.perf_counter()

    output_text = "".join(chunks)
    ttft = (first_token_time - t_request) if first_token_time is not None else None
    total_latency = t_done - t_request
    output_tokens_rough = count_rough_tokens(output_text)
    decode_time = total_latency - ttft if ttft is not None else None
    tokens_per_second = (
        output_tokens_rough / decode_time
        if decode_time and decode_time > 0
        else 0.0
    )

    return {
        "output_text": output_text,
        "ttft": ttft,
        "total_latency": total_latency,
        "output_tokens_rough": output_tokens_rough,
        "tokens_per_second_rough": tokens_per_second,
    }


def main() -> int:
    _llm = _load_vla_llm_defaults()
    parser = argparse.ArgumentParser(
        description="测 vLLM 流式首字延迟（TTFT）。--model / --base-url / --api-key 默认与 VLA_Agent_Core/config.yaml 中 llm 一致。",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-url", type=str, default=_llm["base_url"])
    parser.add_argument("--api-key", type=str, default=_llm["api_key"])
    parser.add_argument(
        "--model",
        type=str,
        default=_llm["model_name"],
        help="默认读取自 VLA_Agent_Core/config.yaml 的 llm.model_name",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    if not args.input.is_file():
        print(
            f"未找到输入: {args.input}\n"
            f"请先运行 plan_validity_rate/evaluate_plan_validity.py 或指定 --input。",
            file=sys.stderr,
        )
        return 2

    rows = load_jsonl(args.input)
    rows = [
        r for r in rows
        if r.get("tested_for_plan_validity") is True
        and str(r.get("category", "")) in {"normal_action", "combo_action"}
    ]
    if args.limit > 0:
        rows = rows[: args.limit]

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    results = []
    for i, row in enumerate(rows, start=1):
        instruction = str(row.get("asr_text") or row.get("reference_text") or "").strip()
        print(f"[vLLM {i}/{len(rows)}] {row.get('id') or row.get('sample_id')} | {instruction}")

        try:
            metrics = stream_chat_completion(
                client=client,
                model=args.model,
                instruction=instruction,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            error = ""
        except Exception as e:
            metrics = {
                "output_text": "",
                "ttft": None,
                "total_latency": None,
                "output_tokens_rough": 0,
                "tokens_per_second_rough": 0.0,
            }
            error = repr(e)

        results.append(
            {
                "backend": "vllm",
                "id": row.get("id") or row.get("sample_id"),
                "category": row.get("category"),
                "voice": row.get("voice"),
                "instruction": instruction,
                "ttft": metrics["ttft"],
                "total_latency": metrics["total_latency"],
                "output_tokens_rough": metrics["output_tokens_rough"],
                "tokens_per_second_rough": metrics["tokens_per_second_rough"],
                "output_text": metrics["output_text"],
                "error": error,
            }
        )

    write_jsonl(args.output, results)
    print(f"[Done] Written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())