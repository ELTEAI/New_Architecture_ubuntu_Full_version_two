#!/usr/bin/env python3
"""
下载 Qwen3.5-4B 到 VLA_Pipeline/models/Qwen3.5-4B

用法:
  python scripts/download_qwen35_4b.py

可选环境变量:
  HF_TOKEN / HUGGING_FACE_HUB_TOKEN
  HF_ENDPOINT=https://hf-mirror.com   # 网络不稳时可用镜像
"""

from pathlib import Path
import os

from huggingface_hub import snapshot_download


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    target = root / "models" / "Qwen3.5-4B"
    target.mkdir(parents=True, exist_ok=True)

    repo_id = "Qwen/Qwen3.5-4B"
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    max_workers = int(os.getenv("HF_MAX_WORKERS", "1"))

    print(f"[DL] repo={repo_id}")
    print(f"[DL] target={target}")
    print(f"[DL] max_workers={max_workers}")

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        token=token,
        max_workers=max_workers,
    )

    print("[DL] done.")


if __name__ == "__main__":
    main()

