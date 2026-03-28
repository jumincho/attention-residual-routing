#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

CONFIG_PATH="${CONFIG_PATH:?set CONFIG_PATH=/abs/path/to/config.yaml}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/torchrun}"

echo "[ccnews-standard-seed-v8] config=${CONFIG_PATH}"
echo "[ccnews-standard-seed-v8] estimated runtime: ~3h-4.5h on 4x RTX 8000 for 0->5000"

"${PYTHON_BIN}" --standalone --nproc_per_node=4 scripts/train_lm.py --config "${CONFIG_PATH}"
