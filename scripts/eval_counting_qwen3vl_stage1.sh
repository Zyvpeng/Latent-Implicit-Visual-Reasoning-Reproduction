#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CHECKPOINT="${1:-outputs/counting_livr_stage1/best}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python -m livr.eval \
  --config configs/counting_qwen3vl_livr_stage1.yaml \
  --checkpoint "${CHECKPOINT}" \
  --output-dir outputs/counting_livr_stage1_eval
