#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

WAIT_PID="${WAIT_PID:-}"
if [[ -n "$WAIT_PID" ]]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 60
  done
fi

SELECTION_OUT="${SELECTION_OUT:-$ROOT/results/ccnews_multiseed_multisplit_v8/v8_ccnews_dev_frozen_selection.csv}"

"$ROOT/.venv/bin/python" "$ROOT/scripts/select_ccnews_v8_frozen_configs.py" \
  --seeds 42 43 44 \
  --steps 2500 3000 3500 4000 4500 5000 5500 6000 \
  --bank-sizes 32 64 \
  --feature-modes attnres \
  --output "$SELECTION_OUT"

SELECTION_CSV="${SELECTION_OUT%.*}_winners.csv" \
FINAL_SPLITS="${FINAL_SPLITS:-final_A final_B final_C}" \
DEPLOY_BATCH_SIZE="${DEPLOY_BATCH_SIZE:-16}" \
DEPLOY_NUM_SEQUENCES="${DEPLOY_NUM_SEQUENCES:-256}" \
DEPLOY_TIMING_REPEATS="${DEPLOY_TIMING_REPEATS:-5}" \
bash "$ROOT/scripts/run_ccnews_locked_multisplit_v8.sh"

echo "[ccnews-lockbox-after-followup-v8] complete"
