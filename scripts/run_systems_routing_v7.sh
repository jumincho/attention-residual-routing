#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

CHECKPOINT="${CHECKPOINT:-$ROOT/results/scale24x512_ccnews_attnres_dense_v7/checkpoint_step_003000.pt}"
BANK_TAG="${BANK_TAG:-v7_ccnews_p256d64_lockbox_step3000_bank}"
TRAIN_TAGS="${TRAIN_TAGS:-v7_ccnews_p256d64_lockbox_step3000_train4096}"
EVAL_TAGS="${EVAL_TAGS:-v7_ccnews_p256d64_lockbox_step3000_dev2048}"
MANIFEST_PATH="${MANIFEST_PATH:-$ROOT/results/lockbox_manifests_v7/v7_ccnews_p256d64_lockbox_dev_test.jsonl}"
FEATURE_MODE="${FEATURE_MODE:-attnres}"
SELECTED_MODEL="${SELECTED_MODEL:-rf_pair}"
SKIP_COUNT="${SKIP_COUNT:-1}"
BANK_SIZE="${BANK_SIZE:-32}"
NUM_SEQUENCES="${NUM_SEQUENCES:-256}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TIMING_REPEATS="${TIMING_REPEATS:-5}"
TEMPLATE_LIMITS_RAW="${TEMPLATE_LIMITS:-0 2 4 8}"

read -r -a TEMPLATE_LIMITS <<<"$TEMPLATE_LIMITS_RAW"

for template_limit in "${TEMPLATE_LIMITS[@]}"; do
  output_tag="v7_ccnews_step3000_skip${SKIP_COUNT}_${SELECTED_MODEL}_tpl${template_limit}"
  TEMPLATE_LIMIT="$template_limit" \
  CHECKPOINT="$CHECKPOINT" \
  BANK_TAG="$BANK_TAG" \
  TRAIN_TAGS="$TRAIN_TAGS" \
  EVAL_TAGS="$EVAL_TAGS" \
  MANIFEST_PATH="$MANIFEST_PATH" \
  FEATURE_MODE="$FEATURE_MODE" \
  SELECTED_MODEL="$SELECTED_MODEL" \
  SKIP_COUNT="$SKIP_COUNT" \
  BANK_SIZE="$BANK_SIZE" \
  NUM_SEQUENCES="$NUM_SEQUENCES" \
  BATCH_SIZE="$BATCH_SIZE" \
  TIMING_REPEATS="$TIMING_REPEATS" \
  OUTPUT_TAG="$output_tag" \
  bash "$ROOT/scripts/run_deployment_measurement_v7.sh"
done

echo "[systems-routing-v7] complete"
