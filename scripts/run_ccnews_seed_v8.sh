#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

CONFIG_PATH="${CONFIG_PATH:?set CONFIG_PATH=/abs/path/to/config.yaml}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/torchrun}"

echo "[ccnews-seed-v8] config=${CONFIG_PATH}"
echo "[ccnews-seed-v8] estimated runtime: ~3.5h-5.0h on 4x RTX 8000 for 0->6000"
echo "[ccnews-seed-v8] checkpoints: 2500 3000 3500 4000 4500 5000 5500 6000"

"${PYTHON_BIN}" --standalone --nproc_per_node=4 scripts/train_lm.py --config "${CONFIG_PATH}"
