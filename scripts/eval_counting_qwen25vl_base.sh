#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python -m livr.eval \
  --config configs/counting_qwen25vl_sft.yaml \
  --output-dir outputs/counting_qwen25vl_base_eval
