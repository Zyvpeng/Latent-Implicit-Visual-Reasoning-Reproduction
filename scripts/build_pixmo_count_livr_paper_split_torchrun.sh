#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29632}"
CACHE_DIR="${CACHE_DIR:-/tmp/livr_clip_cache/pixmo_count_livr_paper}"

torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  -m livr.build_pixmo_count_livr_paper_split \
  --input-dir data/pixmo_count_official \
  --output-dir data/pixmo_count_livr_paper \
  --clip-cache-dir "${CACHE_DIR}" \
  --train-size 1000 \
  --seed 42 \
  --min-count 2 \
  --max-count 10
