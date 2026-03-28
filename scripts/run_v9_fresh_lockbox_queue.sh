#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
LOG_DIR="${LOG_DIR:-$ROOT/results/v9_queue}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/run_v9_fresh_lockbox_queue.log") 2>&1

WAIT_PATTERN="${WAIT_PATTERN:-v9_repro_seed43_step6000_b32_hgb_pair_attnres_shard32}"
RUN_COMPARE_SEED43="${RUN_COMPARE_SEED43:-1}"
RUN_DEV_SELECTION="${RUN_DEV_SELECTION:-1}"
RUN_FREEZE="${RUN_FREEZE:-1}"
RUN_FINAL_LOCKBOX="${RUN_FINAL_LOCKBOX:-1}"

SELECTION_OUTPUT="${SELECTION_OUTPUT:-$ROOT/results/ccnews_multiseed_multisplit_v9/v9_ccnews_dev_frozen_selection.csv}"
FREEZE_LEDGER="${FREEZE_LEDGER:-$ROOT/results/lockbox_manifests_v9/v9_ccnews_p256d64_lockbox_selection_freeze.csv}"
FEATURE_MODES="${FEATURE_MODES:-attnres attnres_stp_scalar attnres_stp_diff hidden hidden_stp_diff}"
ALLOWED_MODELS="${ALLOWED_MODELS:-rf_pair hgb_pair retrieval_rerank_top4}"
BANK_SIZES="${BANK_SIZES:-32}"
STEPS="${STEPS:-3000 3500 5500 6000}"

wait_for_pattern() {
  local pattern="$1"
  echo "[$(date '+%F %T %Z')] waiting for pattern to clear: $pattern"
  while pgrep -f "$pattern" >/dev/null; do
    echo "[$(date '+%F %T %Z')] pattern still active: $pattern"
    sleep 60
  done
  echo "[$(date '+%F %T %Z')] pattern cleared: $pattern"
}

compare_seed43_repro() {
  echo "[$(date '+%F %T %Z')] comparing seed43 fresh repro outputs"
  SEED=43 \
  STEP=6000 \
  MODEL_NAME=hgb_pair \
  FEATURE_MODE=attnres \
  BANK_SIZE=32 \
  SPLITS="final_A final_B final_C" \
  bash "$ROOT/scripts/run_compare_v8_seed_repro_v9.sh"
}

run_dev_selection_v9() {
  echo "[$(date '+%F %T %Z')] running fresh V9 dev selection"
  TRAIN_MANIFEST="$ROOT/results/lockbox_manifests_v9/v9_ccnews_p256d64_lockbox_train4096.jsonl" \
  VAL_MANIFEST="$ROOT/results/lockbox_manifests_v9/v9_ccnews_p256d64_lockbox_validation512.jsonl" \
  DEV_MANIFEST="$ROOT/results/lockbox_manifests_v9/v9_ccnews_p256d64_lockbox_dev_select_v9_2048.jsonl" \
  STEPS_SEED42="${STEPS_SEED42:-3000 5500}" \
  STEPS_SEED43="${STEPS_SEED43:-3000 6000}" \
  STEPS_SEED44="${STEPS_SEED44:-3000 3500}" \
  BANK_SIZES="$BANK_SIZES" \
  FEATURE_MODES="$FEATURE_MODES" \
  ORACLE_BATCH_SIZE="${ORACLE_BATCH_SIZE:-24}" \
  TRAIN_TOTAL_SHARDS="${TRAIN_TOTAL_SHARDS:-16}" \
  VAL_TOTAL_SHARDS="${VAL_TOTAL_SHARDS:-4}" \
  DEV_TOTAL_SHARDS="${DEV_TOTAL_SHARDS:-8}" \
  FAST_MODE="${FAST_MODE:-1}" \
  SKIP_EXISTING="${SKIP_EXISTING:-1}" \
  bash "$ROOT/scripts/run_ccnews_dev_selection_v9.sh"
}

freeze_v9_selection() {
  echo "[$(date '+%F %T %Z')] freezing V9 dev selection"
  "$PY" "$ROOT/scripts/select_ccnews_v9_frozen_configs.py" \
    --seeds 42 43 44 \
    --steps $STEPS \
    --bank-sizes $BANK_SIZES \
    --feature-modes $FEATURE_MODES \
    --allowed-models $ALLOWED_MODELS \
    --output "$SELECTION_OUTPUT" \
    --freeze-ledger "$FREEZE_LEDGER"
}

run_final_lockbox_v9() {
  echo "[$(date '+%F %T %Z')] running frozen V9 final_D/E/F evaluation"
  SELECTION_OUTPUT="$SELECTION_OUTPUT" \
  TRAIN_MANIFEST="$ROOT/results/lockbox_manifests_v9/v9_ccnews_p256d64_lockbox_train4096.jsonl" \
  VAL_MANIFEST="$ROOT/results/lockbox_manifests_v9/v9_ccnews_p256d64_lockbox_validation512.jsonl" \
  FINAL_MANIFEST_TEMPLATE="$ROOT/results/lockbox_manifests_v9/v9_ccnews_p256d64_lockbox_%s.jsonl" \
  FEATURE_MODES="$FEATURE_MODES" \
  ALLOWED_MODELS="$ALLOWED_MODELS" \
  BANK_SIZES="$BANK_SIZES" \
  ORACLE_BATCH_SIZE="${ORACLE_BATCH_SIZE:-24}" \
  TRAIN_TOTAL_SHARDS="${TRAIN_TOTAL_SHARDS:-16}" \
  VAL_TOTAL_SHARDS="${VAL_TOTAL_SHARDS:-4}" \
  FINAL_TOTAL_SHARDS="${FINAL_TOTAL_SHARDS:-16}" \
  DEPLOY_NUM_SEQUENCES="${DEPLOY_NUM_SEQUENCES:-256}" \
  DEPLOY_BATCH_SIZE="${DEPLOY_BATCH_SIZE:-16}" \
  DEPLOY_TIMING_REPEATS="${DEPLOY_TIMING_REPEATS:-5}" \
  SKIP_EXISTING="${SKIP_EXISTING:-1}" \
  bash "$ROOT/scripts/run_ccnews_locked_multisplit_v9.sh"
}

echo "[$(date '+%F %T %Z')] v9 fresh lockbox queue start"
wait_for_pattern "$WAIT_PATTERN"

if [[ "$RUN_COMPARE_SEED43" == "1" ]]; then
  compare_seed43_repro
fi

if [[ "$RUN_DEV_SELECTION" == "1" ]]; then
  run_dev_selection_v9
fi

if [[ "$RUN_FREEZE" == "1" ]]; then
  freeze_v9_selection
fi

if [[ "$RUN_FINAL_LOCKBOX" == "1" ]]; then
  run_final_lockbox_v9
fi

echo "[$(date '+%F %T %Z')] v9 fresh lockbox queue complete"
