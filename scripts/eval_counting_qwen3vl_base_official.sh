#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python -m livr.eval_qwen3vl_base_official \
  --config configs/counting_qwen3vl_sft.yaml \
  --output-dir outputs/counting_base_official_eval
