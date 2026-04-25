"""
Qwen3.5 需 transformers>=5与 vLLM 0.19 的 transformers<5 冲突
cd /home/ubuntu/New_Architecture
source venvs/qwen35/bin/activate
在本地用 HuggingFace Transformers 流式推理测 TTFT；默认权重目录（与 vLLM 同一份 Qwen3.5-4B 拷贝）:
    <repo>/VLA_Pipeline/models/Qwen3.5-4B

默认输入:
    <voice_command_dataset>/plan_validity_rate/plan_validity_results.jsonl
默认输出:
    <voice_command_dataset>/TTFT/ttft_transformers_results.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

THIS_DIR = Path(__file__).resolve().parent
VOICE_DIR = THIS_DIR.parent
_REPO_ROOT = VOICE_DIR.parent.parent
DEFAULT_INPUT = VOICE_DIR / "plan_validity_rate" / "plan_validity_results.jsonl"
DEFAULT_OUTPUT = THIS_DIR / "ttft_transformers_results.jsonl"
# 与仓库内 vLLM / pipeline 使用的 4B 权重路径一致
DEFAULT_MODEL_PATH = _REPO_ROOT / "VLA_Pipeline" / "models" / "Qwen3.5-4B"

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


def build_prompt(tokenizer: Any, instruction: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return f"{SYSTEM_PROMPT}\n\nUser: {instruction}\nAssistant:"


def count_rough_tokens(text: str) -> int:
    return max(1, len(text.split()))


def _resolve_dtype(name: str) -> Any:
    n = (name or "auto").lower().strip()
    if n in ("float16", "fp16"):
        return torch.float16
    if n in ("bfloat16", "bf16"):
        return torch.bfloat16
    if n in ("float32", "fp32"):
        return torch.float32
    return "auto"


def _ensure_tokenizer_config(tokenizer: Any) -> None:
    if getattr(tokenizer, "pad_token", None) is None and getattr(
        tokenizer, "eos_token", None
    ) is not None:
        tokenizer.pad_token = tokenizer.eos_token


def generate_streaming(
    model: Any,
    tokenizer: Any,
    instruction: str,
    max_new_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    prompt = build_prompt(tokenizer, instruction)
    inputs = tokenizer(prompt, return_tensors="pt")
    # 与 model 同设备，避免多卡时 input 在 meta/cpu 而权重在 GPU
    dev = next(model.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    gen: dict[str, Any] = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
    }
    if temperature and temperature > 0:
        gen["do_sample"] = True
        gen["temperature"] = float(temperature)
    else:
        gen["do_sample"] = False

    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        gen["eos_token_id"] = eos
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is not None:
        gen["pad_token_id"] = pad

    t_request = time.perf_counter()
    first_token_time: float | None = None
    chunks: list[str] = []

    def _run() -> None:
        with torch.inference_mode():
            model.generate(**gen)

    th = threading.Thread(target=_run)
    th.start()

    for text in streamer:
        if text:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            chunks.append(text)

    th.join()
    t_done = time.perf_counter()

    output_text = "".join(chunks)
    ttft = (first_token_time - t_request) if first_token_time is not None else None
    total_latency = t_done - t_request
    output_tokens_rough = count_rough_tokens(output_text)
    decode_time = total_latency - ttft if ttft is not None else None
    tps = (
        output_tokens_rough / decode_time
        if decode_time and decode_time > 0
        else 0.0
    )

    return {
        "output_text": output_text,
        "ttft": ttft,
        "total_latency": total_latency,
        "output_tokens_rough": output_tokens_rough,
        "tokens_per_second_rough": tps,
    }


def main() -> int:
    default_mp = str(DEFAULT_MODEL_PATH)
    p = argparse.ArgumentParser(
        description="本地 Transformers 流式测首字延迟（TTFT）。默认同目录权重: VLA_Pipeline/models/Qwen3.5-4B。",
    )
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument(
        "--model-path",
        type=str,
        default=default_mp,
        help="默认: <repo>/VLA_Pipeline/models/Qwen3.5-4B",
    )
    p.add_argument(
        "--attn-implementation",
        type=str,
        default="sdpa",
        choices=("sdpa", "eager", "flash_attention_2"),
        help="PyTorch2+ 下 sdpa 通常比 eager 快；需安装 flash-attn 才用 flash_attention_2。",
    )
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="auto | bfloat16 | float16 | float32",
    )
    p.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="传给 from_pretrained(device_map=...)。none/null 表示不传 device_map。",
    )
    args = p.parse_args()

    model_path = Path(args.model_path)
    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        print(
            f"未找到有效模型目录: {model_path}\n"
            f"需包含 config.json（Qwen3.5-4B 等）。可传 --model-path 指向本地权重。",
            file=sys.stderr,
        )
        return 2

    if not args.input.is_file():
        print(
            f"未找到输入: {args.input}\n"
            f"请先运行 plan_validity_rate/evaluate_plan_validity.py 或指定 --input。",
            file=sys.stderr,
        )
        return 2

    rows = load_jsonl(args.input)
    rows = [
        r
        for r in rows
        if r.get("tested_for_plan_validity") is True
        and str(r.get("category", "")) in {"normal_action", "combo_action"}
    ]
    if args.limit > 0:
        rows = rows[: args.limit]

    print(f"[Info] model_path = {model_path.resolve()}")
    print("[Info] loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True
    )
    _ensure_tokenizer_config(tokenizer)

    torch_dtype = _resolve_dtype(args.dtype)
    model_kw: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "attn_implementation": args.attn_implementation,
    }
    if (args.dtype or "").lower() != "auto":
        model_kw["torch_dtype"] = torch_dtype
    else:
        model_kw["torch_dtype"] = "auto"

    dm: str | None
    if args.device_map and args.device_map.lower() in ("none", "null", ""):
        dm = None
    else:
        dm = args.device_map

    print(
        f"[Info] loading model (device_map={dm!r}, "
        f"attn={args.attn_implementation!r}) ..."
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        device_map=dm,
        **model_kw,
    )
    model.eval()

    # 首次构图 / CUDA 多跑一轮，使后续样本次可比
    print("[Info] warm-up (short generation) ...")
    with torch.inference_mode():
        _ = generate_streaming(
            model=model,
            tokenizer=tokenizer,
            instruction="walk forward",
            max_new_tokens=32,
            temperature=args.temperature,
        )

    results: list[dict[str, Any]] = []
    for i, row in enumerate(rows, start=1):
        inst = str(row.get("asr_text") or row.get("reference_text") or "").strip()
        label = row.get("uid") or row.get("sample_id") or i
        print(f"[transformers {i}/{len(rows)}] {label} | {inst}")

        try:
            metrics = generate_streaming(
                model=model,
                tokenizer=tokenizer,
                instruction=inst,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            err = ""
        except Exception as e:
            metrics = {
                "output_text": "",
                "ttft": None,
                "total_latency": None,
                "output_tokens_rough": 0,
                "tokens_per_second_rough": 0.0,
            }
            err = repr(e)

        results.append(
            {
                "backend": "transformers",
                "model_path": str(model_path.resolve()),
                "uid": row.get("uid"),
                "id": row.get("id") or row.get("sample_id"),
                "category": row.get("category"),
                "voice": row.get("voice"),
                "instruction": inst,
                "ttft": metrics["ttft"],
                "total_latency": metrics["total_latency"],
                "output_tokens_rough": metrics["output_tokens_rough"],
                "tokens_per_second_rough": metrics["tokens_per_second_rough"],
                "output_text": metrics["output_text"],
                "error": err,
            }
        )

    write_jsonl(args.output, results)
    print(f"[Done] Written to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
