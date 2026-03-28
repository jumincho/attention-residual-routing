#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

EXPERIMENT_DIR="${EXPERIMENT_DIR:-$ROOT/results/scale24x512_ccnews_attnres_dense_v7}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:?set CHECKPOINT_STEPS='3000 5000 5500'}"
BANK_SIZES="${BANK_SIZES:-32 64}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-$ROOT/results/lockbox_manifests_v7/v7_ccnews_p256d64_lockbox_train4096.jsonl}"
EVAL_MANIFEST="${EVAL_MANIFEST:-$ROOT/results/lockbox_manifests_v7/v7_ccnews_p256d64_lockbox_dev_test.jsonl}"
TAG_PREFIX="${TAG_PREFIX:-v7_ccnews_p256d64_lockbox}"
PROMPT_LEN="${PROMPT_LEN:-256}"
DECODE_LEN="${DECODE_LEN:-64}"

for step in $CHECKPOINT_STEPS; do
  for bank_size in $BANK_SIZES; do
    CHECKPOINT="$EXPERIMENT_DIR/checkpoint_step_$(printf '%06d' "$step").pt" \
    BANK_TAG="${TAG_PREFIX}_step${step}_bank" \
    TRAIN_MANIFEST="$TRAIN_MANIFEST" \
    EVAL_MANIFEST="$EVAL_MANIFEST" \
    PROMPT_LEN="$PROMPT_LEN" \
    DECODE_LEN="$DECODE_LEN" \
    BANK_SIZE="$bank_size" \
    FAST_MODE=0 \
    TRAIN_TAG="${TAG_PREFIX}_step${step}_train4096" \
    EVAL_TAG="${TAG_PREFIX}_step${step}_dev2048" \
    OUTPUT_PREFIX="v7_ccnews_step${step}_dev2048_b${bank_size}" \
    bash "$ROOT/scripts/run_hetero_selector_scaling_v7.sh"
  done
done

echo "[ccnews-shortlist-full-scorer-v7] complete"
