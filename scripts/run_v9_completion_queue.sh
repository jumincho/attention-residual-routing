#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
SEED43_PATTERN="${SEED43_PATTERN:-v9_repro_seed43_step6000_b32_hgb_pair_attnres_shard32}"
FEATURE_MODES="${FEATURE_MODES:-attnres stp_scalar attnres_stp_scalar attnres_stp_diff hidden hidden_stp_diff}"
LOCKED_ALLOWED_MODELS="${LOCKED_ALLOWED_MODELS:-rf_pair rf_pair_weighted hgb_pair hgb_pair_weighted retrieval_rerank_top2 retrieval_rerank_top4}"

log() {
  printf '[%s] %s\n' "$(TZ=Asia/Seoul date '+%F %T KST')" "$*"
}

log "queue armed: waiting for ${SEED43_PATTERN}"
while pgrep -af "$SEED43_PATTERN" >/dev/null; do
  sleep 60
done

log "seed43 forensic rerun cleared; starting compare"
PY="$PY" \
SEED=43 \
STEP=6000 \
MODEL_NAME=hgb_pair \
FEATURE_MODE=attnres \
BANK_SIZE=32 \
bash "$ROOT/scripts/run_compare_v8_seed_repro_v9.sh"

log "seed43 compare complete; starting fresh dev_select_v9"
PY="$PY" \
SEEDS="42 43 44" \
BANK_SIZES="32 64" \
FEATURE_MODES="$FEATURE_MODES" \
ORACLE_BATCH_SIZE=128 \
FAST_MODE=0 \
SKIP_EXISTING=1 \
bash "$ROOT/scripts/run_ccnews_dev_selection_v9.sh"

log "fresh dev_select_v9 complete; starting frozen final_D/E/F"
PY="$PY" \
SEEDS="42 43 44" \
STEPS="3000 3500 5500 6000" \
BANK_SIZES="32 64" \
FEATURE_MODES="$FEATURE_MODES" \
ALLOWED_MODELS="$LOCKED_ALLOWED_MODELS" \
SKIP_EXISTING=1 \
FINAL_SPLITS="final_D final_E final_F" \
ORACLE_BATCH_SIZE=128 \
bash "$ROOT/scripts/run_ccnews_locked_multisplit_v9.sh"

log "queue complete through frozen final_D/E/F"
