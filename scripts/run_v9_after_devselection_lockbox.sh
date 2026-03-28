#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
WAIT_PID="${WAIT_PID:-72884}"
LOG_DIR="${LOG_DIR:-$ROOT/results/v9_queue}"
LOG_PATH="${LOG_PATH:-$LOG_DIR/run_v9_after_devselection_lockbox.log}"

mkdir -p "$LOG_DIR"
exec >>"$LOG_PATH" 2>&1

log() {
  printf '[%s] %s\n' "$(TZ=Asia/Seoul date '+%F %T KST')" "$*"
}

log "watcher start: waiting for PID ${WAIT_PID} to clear before frozen final_D/E/F"
while kill -0 "$WAIT_PID" 2>/dev/null; do
  log "watcher: PID ${WAIT_PID} still active"
  sleep 60
done

if pgrep -af '[r]un_ccnews_locked_multisplit_v9.sh' >/dev/null; then
  log "watcher: final_D/E/F runner already active; exiting without duplicate launch"
  exit 0
fi

log "watcher: PID ${WAIT_PID} cleared; launching frozen final_D/E/F"
PY="$PY" \
SEEDS="${SEEDS:-42 43 44}" \
STEPS="${STEPS:-3000 3500 5500 6000}" \
BANK_SIZES="${BANK_SIZES:-32 64}" \
FEATURE_MODES="${FEATURE_MODES:-attnres stp_scalar attnres_stp_scalar attnres_stp_diff hidden hidden_stp_diff}" \
ALLOWED_MODELS="${ALLOWED_MODELS:-rf_pair rf_pair_weighted hgb_pair hgb_pair_weighted retrieval_rerank_top2 retrieval_rerank_top4}" \
SKIP_EXISTING="${SKIP_EXISTING:-1}" \
FINAL_SPLITS="${FINAL_SPLITS:-final_D final_E final_F}" \
ORACLE_BATCH_SIZE="${ORACLE_BATCH_SIZE:-128}" \
bash "$ROOT/scripts/run_ccnews_locked_multisplit_v9.sh"

log "watcher complete"
