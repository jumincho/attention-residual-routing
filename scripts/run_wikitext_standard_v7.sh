#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/torchrun}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT/configs/scale_heterogeneity_v7/standard_24x512_wikitext103.yaml}"

"${PYTHON_BIN}" --standalone --nproc_per_node=4 scripts/train_lm.py --config "${CONFIG_PATH}"
