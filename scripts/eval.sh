#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python -m livr.eval --config configs/localization_qwen3vl_livr.yaml --checkpoint outputs/localization_livr/epoch_0
