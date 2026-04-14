#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python -m livr.build_pixmo_count_clipfar_split \
  --input-dir data/pixmo_count_official \
  --output-dir data/pixmo_count_clipfar \
  --train-size 1000 \
  --seed 42 \
  --min-count 2 \
  --max-count 10 \
  --remove-near-duplicates
