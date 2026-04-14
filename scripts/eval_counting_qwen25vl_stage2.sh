#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CHECKPOINT="${1:-outputs/counting_qwen25vl_livr_stage2/best}"
EXTRA_ARGS=()
if [[ "${SAVE_LATENT_ATTN:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--save-latent-attn)
  if [[ -n "${LATENT_ATTN_DIR:-}" ]]; then
    EXTRA_ARGS+=(--latent-attn-dir "${LATENT_ATTN_DIR}")
  fi
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -m livr.eval \
  --config configs/counting_qwen25vl_livr_stage2.yaml \
  --checkpoint "${CHECKPOINT}" \
  --output-dir outputs/counting_qwen25vl_livr_stage2_eval \
  "${EXTRA_ARGS[@]}"
