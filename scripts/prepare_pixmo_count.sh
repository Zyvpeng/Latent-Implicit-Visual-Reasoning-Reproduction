#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python -m livr.prepare_pixmo_count --output-dir data/pixmo_count --train-size 1000 --seed 42 --min-count 2 --max-count 10
