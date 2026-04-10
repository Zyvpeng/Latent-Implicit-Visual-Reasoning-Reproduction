#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python -m livr.train --config configs/localization_qwen3vl_sft.yaml
