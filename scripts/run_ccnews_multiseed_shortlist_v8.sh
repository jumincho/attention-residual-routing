#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

SEEDS_RAW="${SEEDS:-42 43 44}"
read -r -a SEEDS <<<"$SEEDS_RAW"

for seed in "${SEEDS[@]}"; do
  SEED="$seed" \
  TOP_N="${TOP_N:-2}" \
  BANK_SIZES="${BANK_SIZES:-32 64}" \
  OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-regret_reduction_v8}" \
  PLOT_PREFIX="${PLOT_PREFIX:-regret_reduction_v8}" \
  BATCH_SIZE="${BATCH_SIZE:-128}" \
  bash "$ROOT/scripts/run_ccnews_shortlist_from_readiness_v8.sh"
done

echo "[ccnews-multiseed-shortlist-v8] complete"
