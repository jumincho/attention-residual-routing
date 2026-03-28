#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/torchrun}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT/configs/scale_heterogeneity_v5/standard_24x512_ccnews.yaml}"

cd "$ROOT"
"${PYTHON_BIN}" --standalone --nproc_per_node=4 scripts/train_lm.py --config "${CONFIG_PATH}"
