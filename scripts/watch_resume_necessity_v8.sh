#!/usr/bin/env bash
set -euo pipefail
ROOT="/raid2/chojm/attnres-routing-research"
PID="${1:-12441}"
while kill -0 "$PID" 2>/dev/null; do
  sleep 20
done
source "$ROOT/.venv/bin/activate"
WINNERS_CSV="$ROOT/results/ccnews_multiseed_multisplit_v8/v8_ccnews_dev_frozen_selection_winners.csv" \
FINAL_SPLITS='final_A final_B final_C' \
SEEDS='43 44' \
bash "$ROOT/scripts/run_ccnews_necessity_multiseed_v8.sh"
