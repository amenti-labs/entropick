#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -f .env ]]; then
  echo "Missing deployments/openentropy/.env" >&2
  echo "Create it first with: cp .env.example .env" >&2
  exit 1
fi

set -a
source .env
set +a

: "${HF_MODEL:=Qwen/Qwen2.5-1.5B-Instruct}"
: "${VLLM_PORT:=8000}"

exec vllm serve "$HF_MODEL" \
  --port "$VLLM_PORT" \
  --logits-processors qr_sampler \
  "$@"
