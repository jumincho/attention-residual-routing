#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
SLEEP_SECS="${SLEEP_SECS:-300}"
SEED42_DONE_MARKER="${SEED42_DONE_MARKER:-$ROOT/results/regret_reduction_v8/v8_ccnews_seed42_full_step6000_dev2048_b64_attnres_summary.csv}"
STD43_DONE_MARKER="${STD43_DONE_MARKER:-$ROOT/results/scale24x512_ccnews_standard_seed43_v8/checkpoint_step_006000.pt}"
STD44_DONE_MARKER="${STD44_DONE_MARKER:-$ROOT/results/scale24x512_ccnews_standard_seed44_v8/checkpoint_step_006000.pt}"
WINNERS_CSV="${WINNERS_CSV:-$ROOT/results/ccnews_multiseed_multisplit_v8/v8_ccnews_dev_frozen_selection_winners.csv}"
SELECTION_CSV="${SELECTION_CSV:-$ROOT/results/ccnews_multiseed_multisplit_v8/v8_ccnews_dev_frozen_selection.csv}"
FINAL_SPLITS="${FINAL_SPLITS:-final_A final_B final_C}"
MANIFEST_DIR="${MANIFEST_DIR:-$ROOT/results/lockbox_manifests_v8}"
LOG_PATH="${LOG_PATH:-$ROOT/results/ccnews_multiseed_multisplit_v8/v8_pipeline.log}"

mkdir -p "$(dirname "$LOG_PATH")"
exec > >(tee -a "$LOG_PATH") 2>&1

timestamp() {
  date +"%Y-%m-%d %H:%M:%S %Z"
}

log() {
  echo "[$(timestamp)] $*"
}

wait_for_file() {
  local path="$1"
  while [[ ! -f "$path" ]]; do
    log "waiting for $path"
    sleep "$SLEEP_SECS"
  done
}

wait_for_process_or_file() {
  local path="$1"
  local pattern="$2"
  while [[ ! -f "$path" ]]; do
    if pgrep -af "$pattern" >/dev/null 2>&1; then
      log "waiting for running process to finish for $path"
    else
      return 1
    fi
    sleep "$SLEEP_SECS"
  done
  return 0
}

log "V8 remaining pipeline start"
log "seed42_done_marker=$SEED42_DONE_MARKER"
log "std43_done_marker=$STD43_DONE_MARKER"

wait_for_file "$SEED42_DONE_MARKER"
wait_for_file "$STD43_DONE_MARKER"

log "seed42 shortlist and standard seed43 are ready"

if [[ ! -f "$STD44_DONE_MARKER" ]]; then
  if wait_for_process_or_file "$STD44_DONE_MARKER" "scale24x512_ccnews_standard_seed44_v8|standard_24x512_ccnews_seed44(_resume4500)?\\.yaml"; then
    log "standard seed44 finished before follow-up"
  else
    log "standard seed44 still missing; training it before follow-up"
    CONFIG_PATH="${CONFIG_PATH_STD44:-configs/scale_heterogeneity_v8/standard_24x512_ccnews_seed44.yaml}" \
    bash "$ROOT/scripts/run_ccnews_standard_seed_v8.sh"
  fi
else
  log "standard seed44 already complete before follow-up"
fi

log "running seed43/44 follow-up shortlist"
bash "$ROOT/scripts/run_ccnews_followup_after_seed42_v8.sh"

log "running frozen selection + locked final multisplit eval"
SELECTION_OUT="$SELECTION_CSV" \
SELECTION_CSV="$WINNERS_CSV" \
FINAL_SPLITS="$FINAL_SPLITS" \
DEPLOY_BATCH_SIZE="${DEPLOY_BATCH_SIZE:-16}" \
DEPLOY_NUM_SEQUENCES="${DEPLOY_NUM_SEQUENCES:-256}" \
DEPLOY_TIMING_REPEATS="${DEPLOY_TIMING_REPEATS:-5}" \
bash "$ROOT/scripts/run_ccnews_lockbox_after_followup_v8.sh"

log "aggregating locked final results"
"$PY" "$ROOT/scripts/aggregate_ccnews_locked_v8.py" \
  --selection-csv "$WINNERS_CSV" \
  --final-splits $FINAL_SPLITS \
  --output-dir "$ROOT/results/ccnews_multiseed_multisplit_v8"

log "running systems-aware speedup sweep"
WINNERS_CSV="$WINNERS_CSV" \
FINAL_SPLITS="$FINAL_SPLITS" \
TEMPLATE_LIMITS="${TEMPLATE_LIMITS:-0 2 4}" \
NUM_SEQUENCES="${NUM_SEQUENCES:-256}" \
BATCH_SIZE="${BATCH_SIZE:-16}" \
TIMING_REPEATS="${TIMING_REPEATS:-5}" \
bash "$ROOT/scripts/run_ccnews_systems_speedup_v8.sh"

log "running cc_news necessity multiseed"
WINNERS_CSV="$WINNERS_CSV" \
SEEDS="${SEEDS_FOR_NECESSITY:-43 44}" \
FINAL_SPLITS="$FINAL_SPLITS" \
bash "$ROOT/scripts/run_ccnews_necessity_multiseed_v8.sh"

mkdir -p "$ROOT/results/ccnews_necessity_multiseed_v8" "$ROOT/results/systems_speedup_v8"

log "aggregating necessity multiseed results"
"$PY" "$ROOT/scripts/aggregate_ccnews_necessity_v8.py" \
  --winners-csv "$WINNERS_CSV" \
  --final-splits $FINAL_SPLITS \
  --output-dir "$ROOT/results/ccnews_necessity_multiseed_v8"

META_CSV="$ROOT/results/boundary_analysis_v8/v8_ccnews_lockbox_metadata.csv"
SUBGROUP_CSV="$ROOT/results/boundary_analysis_v8/v8_ccnews_subgroup_summary.csv"
mkdir -p "$ROOT/results/boundary_analysis_v8"

log "exporting cc_news metadata for subgroup analysis"
"$PY" "$ROOT/scripts/export_ccnews_metadata_v8.py" \
  --manifests \
  "$MANIFEST_DIR/v8_ccnews_p256d64_lockbox_final_A.jsonl" \
  "$MANIFEST_DIR/v8_ccnews_p256d64_lockbox_final_B.jsonl" \
  "$MANIFEST_DIR/v8_ccnews_p256d64_lockbox_final_C.jsonl" \
  --output "$META_CSV"

log "analyzing cc_news subgroups"
"$PY" "$ROOT/scripts/analyze_ccnews_subgroups_v8.py" \
  --pooled-per-seq "$ROOT/results/ccnews_multiseed_multisplit_v8/ccnews_v8_locked_pooled_per_sequence.csv" \
  --metadata-csv "$META_CSV" \
  --manifest-dir "$MANIFEST_DIR" \
  --output-csv "$SUBGROUP_CSV"

log "aggregating systems results"
"$PY" "$ROOT/scripts/aggregate_ccnews_systems_v8.py" \
  --winners-csv "$WINNERS_CSV" \
  --final-splits $FINAL_SPLITS \
  --template-limits ${TEMPLATE_LIMITS:-0 2 4} \
  --output-dir "$ROOT/results/systems_speedup_v8"

log "compiling summary_v8"
"$PY" "$ROOT/scripts/compile_summary_v8.py" \
  --repo-root "$ROOT" \
  --output "$ROOT/results/summary_v8.csv"

log "writing V8 docs"
"$PY" "$ROOT/scripts/write_v8_docs.py" \
  --repo-root "$ROOT"

log "V8 remaining pipeline complete"
