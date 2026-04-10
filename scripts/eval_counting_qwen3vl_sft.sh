#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CHECKPOINT="${1:-outputs/counting_sft/best}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python -m livr.eval \
  --config configs/counting_qwen3vl_sft.yaml \
  --checkpoint "${CHECKPOINT}"
