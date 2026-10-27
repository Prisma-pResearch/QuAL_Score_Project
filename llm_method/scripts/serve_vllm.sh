#!/usr/bin/env bash
# Start an OpenAI-compatible vLLM server for gpt-oss-120b.
#
# Tensor parallel size defaults to 1. Override VLLM_TP_SIZE for a
# multi-GPU machine.
#
#   VLLM_MODEL_PATH     model id or local path     (default: openai/gpt-oss-120b)
#   VLLM_SERVED_NAME    name the API serves        (default: gpt-oss-120b)
#   VLLM_HOST           bind host                  (default: 0.0.0.0)
#   VLLM_PORT           bind port                  (default: 8000)
#   VLLM_TP_SIZE        tensor parallel size       (default: 1)
#   VLLM_GPU_MEM_UTIL   GPU memory fraction        (default: 0.95)
#   VLLM_DTYPE          dtype                      (default: bfloat16)
#   VLLM_MAX_MODEL_LEN  context length             (default: 65536)

set -euo pipefail

MODEL_PATH="${VLLM_MODEL_PATH:-openai/gpt-oss-120b}"
SERVED_NAME="${VLLM_SERVED_NAME:-gpt-oss-120b}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
TP_SIZE="${VLLM_TP_SIZE:-1}"
GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.95}"
DTYPE="${VLLM_DTYPE:-bfloat16}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"

if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm is not on PATH. Install it in the environment you use to serve the model." >&2
  exit 1
fi

echo "Starting vLLM server:"
echo "  Model:           ${MODEL_PATH}"
echo "  Served as:       ${SERVED_NAME}"
echo "  Host/Port:       ${HOST}:${PORT}"
echo "  Tensor parallel: ${TP_SIZE}"
echo "  GPU mem util:    ${GPU_MEM_UTIL}"
echo "  Dtype:           ${DTYPE}"
echo "  Max model len:   ${MAX_MODEL_LEN}"

exec vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --dtype "${DTYPE}" \
  --max-model-len "${MAX_MODEL_LEN}"
