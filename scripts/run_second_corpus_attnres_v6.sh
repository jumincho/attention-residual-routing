#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/torchrun}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT/configs/scale_heterogeneity_v6/attnres_24x512_wikitext103.yaml}"

cd "$ROOT"
"${PYTHON_BIN}" --standalone --nproc_per_node=4 scripts/train_lm.py --config "${CONFIG_PATH}"
