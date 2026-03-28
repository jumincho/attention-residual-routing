#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

WAIT_FOR_PATH="${WAIT_FOR_PATH:?set WAIT_FOR_PATH=/abs/path/to/file}"
SEED="${SEED:?set SEED=44}"

while [[ ! -f "$WAIT_FOR_PATH" ]]; do
  sleep 300
done

source "$ROOT/.venv/bin/activate"
CONFIG_PATH="$ROOT/configs/scale_heterogeneity_v8/attnres_24x512_ccnews_seed${SEED}.yaml" \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
"$ROOT/scripts/run_ccnews_seed_v8.sh"
