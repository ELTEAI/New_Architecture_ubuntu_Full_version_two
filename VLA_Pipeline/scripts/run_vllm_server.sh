#!/usr/bin/env bash
# 在 VLA_Pipeline 内部启动 vLLM（OpenAI 兼容接口）
# 用法：
#   cd /home/ubuntu/New_Architecture/VLA_Pipeline
#   ./scripts/run_vllm_server.sh
#
# 可通过环境变量覆盖：
#   MODEL_PATH                  默认 VLA_Pipeline/models/Qwen3.5-4B
#   VLLM_PORT                   默认 8000
#   MAX_LEN                     默认 8192
#   VLLM_PYTHON                 默认 python3
#   VLLM_LANGUAGE_MODEL_ONLY    默认 1
#   VLLM_TOOL_CALL_PARSER       默认 qwen3_coder
#   VLLM_DISABLE_TOOL_PARSER    设 1 可关闭 tool parser

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-$ROOT/models/Qwen3.5-4B}"
PORT="${VLLM_PORT:-8000}"
MAX_LEN="${MAX_LEN:-8192}"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "未找到模型目录: $MODEL_PATH" >&2
  echo "可先执行: python \"$ROOT/scripts/download_qwen35_4b.py\"" >&2
  exit 1
fi

EXTRA=(--trust-remote-code)
if [[ "${TRUST_REMOTE_CODE:-1}" == "0" ]]; then
  EXTRA=()
fi

SERVE_TAIL=(
  --served-model-name "${SERVED_NAME:-Qwen3.5-4B}"
  --max-model-len "$MAX_LEN"
  --host "${VLLM_HOST:-0.0.0.0}"
  --port "$PORT"
)

if [[ "${VLLM_LANGUAGE_MODEL_ONLY:-1}" != "0" ]]; then
  SERVE_TAIL+=(--language-model-only)
fi

if [[ "${VLLM_DISABLE_TOOL_PARSER:-0}" != "1" ]]; then
  SERVE_TAIL+=(
    --enable-auto-tool-choice
    --tool-call-parser "${VLLM_TOOL_CALL_PARSER:-qwen3_coder}"
  )
fi

SERVE_TAIL+=("${EXTRA[@]}")

PYTHON="${VLLM_PYTHON:-${PYTHON:-python3}}"

if command -v vllm >/dev/null 2>&1; then
  exec vllm serve "$MODEL_PATH" "${SERVE_TAIL[@]}"
fi

if "$PYTHON" -c "import vllm" >/dev/null 2>&1; then
  exec "$PYTHON" -m vllm.entrypoints.cli.main serve "$MODEL_PATH" "${SERVE_TAIL[@]}"
fi

echo "未找到 vllm：既没有「vllm」命令，当前「$PYTHON」也无法 import vllm。" >&2
echo "请先安装 vllm，并确保与 CUDA / 驱动版本匹配。" >&2
exit 1

