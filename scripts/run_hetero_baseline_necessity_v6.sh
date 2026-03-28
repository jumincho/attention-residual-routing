#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

EXPERIMENT_DIR="${EXPERIMENT_DIR:-$ROOT/results/scale24x512_ccnews_standard_v6}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_CONTROL="${RUN_CONTROL:-1}"

if [[ "$RUN_TRAIN" == "1" ]]; then
  bash "$ROOT/scripts/run_scale_heterogeneity_standard_v6.sh"
fi

if [[ "$RUN_CONTROL" == "1" ]]; then
  EXPERIMENT_DIR="$EXPERIMENT_DIR" bash "$ROOT/scripts/run_hetero_necessity_control_v6.sh"
fi

echo "[hetero-baseline-necessity-v6] complete"
