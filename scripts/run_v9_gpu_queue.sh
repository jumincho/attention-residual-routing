#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

LOG_DIR="${LOG_DIR:-$ROOT/results/v9_queue}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/run_v9_gpu_queue.log") 2>&1

WAIT_PATTERN="${WAIT_PATTERN:-v9_repro_seed44_step3500_b32_retrieval_rerank_top4_attnres_shard32}"
RUN_SEED43_REPRO="${RUN_SEED43_REPRO:-1}"
RUN_MATCHED_NECESSITY="${RUN_MATCHED_NECESSITY:-1}"
RUN_STP_HIDDEN="${RUN_STP_HIDDEN:-1}"
RUN_SEED44_COMPARE="${RUN_SEED44_COMPARE:-1}"
RUN_SEED43_COMPARE="${RUN_SEED43_COMPARE:-1}"

wait_for_pattern() {
  local pattern="$1"
  echo "[$(date '+%F %T %Z')] waiting for pattern to clear: $pattern"
  while pgrep -f "$pattern" >/dev/null; do
    echo "[$(date '+%F %T %Z')] pattern still active: $pattern"
    sleep 60
  done
  echo "[$(date '+%F %T %Z')] pattern cleared: $pattern"
}

launch_seed43_repro() {
  echo "[$(date '+%F %T %Z')] launching seed43 fresh raw-oracle repro"
  SEED=43 \
  STEP=6000 \
  MODEL_NAME=hgb_pair \
  FEATURE_MODE=attnres \
  BANK_SIZE=32 \
  FINAL_SPLITS="final_A final_B final_C" \
  REPRO_TAG=v9_repro_seed43_step6000_b32_hgb_pair_attnres_shard32 \
  TRAIN_TOTAL_SHARDS=32 \
  VAL_TOTAL_SHARDS=16 \
  FINAL_TOTAL_SHARDS=16 \
  ORACLE_BATCH_SIZE=24 \
  SKIP_EXISTING=1 \
  bash "$ROOT/scripts/run_v8_targeted_repro.sh"
}

compare_seed44_repro() {
  echo "[$(date '+%F %T %Z')] comparing seed44 fresh repro outputs"
  SEED=44 \
  STEP=3500 \
  MODEL_NAME=retrieval_rerank_top4 \
  FEATURE_MODE=attnres \
  BANK_SIZE=32 \
  SPLITS="final_A final_B final_C" \
  bash "$ROOT/scripts/run_compare_v8_seed_repro_v9.sh"
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

launch_matched_necessity() {
  echo "[$(date '+%F %T %Z')] launching matched-step standard necessity run"
  bash "$ROOT/scripts/run_ccnews_matched_necessity_v9.sh"
}

launch_stp_hidden() {
  echo "[$(date '+%F %T %Z')] launching STP hidden extraction run"
  bash "$ROOT/scripts/run_stp_hidden_extract_v9.sh"
}

echo "[$(date '+%F %T %Z')] v9 gpu queue start"
wait_for_pattern "$WAIT_PATTERN"

if [[ "$RUN_SEED44_COMPARE" == "1" ]]; then
  compare_seed44_repro
fi

if [[ "$RUN_SEED43_REPRO" == "1" ]]; then
  launch_seed43_repro
fi

if [[ "$RUN_SEED43_COMPARE" == "1" ]]; then
  compare_seed43_repro
fi

if [[ "$RUN_MATCHED_NECESSITY" == "1" ]]; then
  launch_matched_necessity
fi

if [[ "$RUN_STP_HIDDEN" == "1" ]]; then
  launch_stp_hidden
fi

echo "[$(date '+%F %T %Z')] v9 gpu queue complete"
