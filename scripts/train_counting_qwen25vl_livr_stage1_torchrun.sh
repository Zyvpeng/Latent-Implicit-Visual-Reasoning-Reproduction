#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
MASTER_PORT="${MASTER_PORT:-29511}"
torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port="${MASTER_PORT}" -m livr.train --config configs/counting_qwen25vl_livr_stage1.yaml
