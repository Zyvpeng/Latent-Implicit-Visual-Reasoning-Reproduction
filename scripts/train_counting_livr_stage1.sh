#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python -m livr.train --config configs/counting_qwen3vl_livr_stage1.yaml
