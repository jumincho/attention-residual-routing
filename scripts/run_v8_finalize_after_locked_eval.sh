#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
WINNERS_CSV="${WINNERS_CSV:-$ROOT/results/ccnews_multiseed_multisplit_v8/v8_ccnews_dev_frozen_selection_winners.csv}"
FINAL_SPLITS="${FINAL_SPLITS:-final_A final_B final_C}"
LOG_PATH="${LOG_PATH:-$ROOT/results/ccnews_multiseed_multisplit_v8/v8_finalize_after_locked_eval.log}"

mkdir -p "$(dirname "$LOG_PATH")"
exec > >(tee -a "$LOG_PATH") 2>&1

wait_for_locked_eval() {
  while ps -eo cmd | rg -q \
    'v8_ccnews_seed43_locked_final_C_step6000_b32_hgb_pair_attnres|v8_ccnews_seed44_locked_train|v8_ccnews_seed44_locked_val|v8_ccnews_seed44_locked_final_|run_locked_final_eval_v8\.sh'
  do
    echo "[$(date '+%F %T %Z')] waiting for locked-final jobs to finish"
    sleep 60
  done
}

echo "[$(date '+%F %T %Z')] V8 finalize watcher started"
wait_for_locked_eval

echo "[$(date '+%F %T %Z')] aggregating locked final results"
"$PY" scripts/aggregate_ccnews_locked_v8.py \
  --selection-csv "$WINNERS_CSV" \
  --final-splits $FINAL_SPLITS \
  --output-dir results/ccnews_multiseed_multisplit_v8

echo "[$(date '+%F %T %Z')] running systems-aware evaluation"
WINNERS_CSV="$WINNERS_CSV" \
FINAL_SPLITS="$FINAL_SPLITS" \
TEMPLATE_LIMITS="${TEMPLATE_LIMITS:-0 2 4}" \
NUM_SEQUENCES="${NUM_SEQUENCES:-256}" \
BATCH_SIZE="${BATCH_SIZE:-16}" \
TIMING_REPEATS="${TIMING_REPEATS:-5}" \
bash scripts/run_ccnews_systems_speedup_v8.sh

echo "[$(date '+%F %T %Z')] aggregating systems-aware results"
"$PY" scripts/aggregate_ccnews_systems_v8.py \
  --winners-csv "$WINNERS_CSV" \
  --final-splits $FINAL_SPLITS \
  --template-limits ${TEMPLATE_LIMITS:-0 2 4} \
  --output-dir results/systems_speedup_v8

echo "[$(date '+%F %T %Z')] running multiseed necessity"
WINNERS_CSV="$WINNERS_CSV" \
SEEDS="${SEEDS:-43 44}" \
FINAL_SPLITS="$FINAL_SPLITS" \
ORACLE_BATCH_SIZE="${ORACLE_BATCH_SIZE:-32}" \
HIDDEN_BATCH_SIZE="${HIDDEN_BATCH_SIZE:-16}" \
TRAIN_TOTAL_SHARDS="${TRAIN_TOTAL_SHARDS:-16}" \
VAL_TOTAL_SHARDS="${VAL_TOTAL_SHARDS:-8}" \
FINAL_TOTAL_SHARDS="${FINAL_TOTAL_SHARDS:-8}" \
bash scripts/run_ccnews_necessity_multiseed_v8.sh

echo "[$(date '+%F %T %Z')] aggregating necessity results"
"$PY" scripts/aggregate_ccnews_necessity_v8.py \
  --winners-csv "$WINNERS_CSV" \
  --final-splits $FINAL_SPLITS \
  --output-dir results/ccnews_necessity_multiseed_v8

echo "[$(date '+%F %T %Z')] exporting metadata and subgroup analysis"
"$PY" scripts/export_ccnews_metadata_v8.py \
  --manifests \
  "$ROOT/results/lockbox_manifests_v8/v8_ccnews_p256d64_lockbox_final_A.jsonl" \
  "$ROOT/results/lockbox_manifests_v8/v8_ccnews_p256d64_lockbox_final_B.jsonl" \
  "$ROOT/results/lockbox_manifests_v8/v8_ccnews_p256d64_lockbox_final_C.jsonl" \
  --output "$ROOT/results/boundary_analysis_v8/v8_ccnews_lockbox_metadata.csv"

"$PY" scripts/analyze_ccnews_subgroups_v8.py \
  --pooled-per-seq "$ROOT/results/ccnews_multiseed_multisplit_v8/ccnews_v8_locked_pooled_per_sequence.csv" \
  --metadata-csv "$ROOT/results/boundary_analysis_v8/v8_ccnews_lockbox_metadata.csv" \
  --manifest-dir "$ROOT/results/lockbox_manifests_v8" \
  --output-csv "$ROOT/results/boundary_analysis_v8/v8_ccnews_subgroup_summary.csv"

echo "[$(date '+%F %T %Z')] compiling summary and docs"
"$PY" scripts/compile_summary_v8.py \
  --repo-root "$ROOT" \
  --output "$ROOT/results/summary_v8.csv"

"$PY" scripts/write_v8_docs.py \
  --repo-root "$ROOT"

echo "[$(date '+%F %T %Z')] V8 finalize watcher complete"
