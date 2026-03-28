#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

EXPERIMENT_DIR="${EXPERIMENT_DIR:?set EXPERIMENT_DIR=/abs/path/to/experiment_dir}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:?set CHECKPOINT_STEPS='3000 3500'}"
BANK_SIZES="${BANK_SIZES:-32 64}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-$ROOT/results/lockbox_manifests_v8/v8_ccnews_p256d64_lockbox_train2048.jsonl}"
EVAL_MANIFEST="${EVAL_MANIFEST:-$ROOT/results/lockbox_manifests_v8/v8_ccnews_p256d64_lockbox_dev_select2048.jsonl}"
TAG_PREFIX="${TAG_PREFIX:-v8_ccnews_p256d64_lockbox}"
OUTPUT_PREFIX_BASE="${OUTPUT_PREFIX_BASE:-${TAG_PREFIX}}"
PROMPT_LEN="${PROMPT_LEN:-256}"
DECODE_LEN="${DECODE_LEN:-64}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-regret_reduction_v8}"
PLOT_PREFIX="${PLOT_PREFIX:-regret_reduction_v8}"
BATCH_SIZE="${BATCH_SIZE:-128}"

EXPERIMENT_DIR="$EXPERIMENT_DIR" \
CHECKPOINT_STEPS="$CHECKPOINT_STEPS" \
TRAIN_MANIFEST="$TRAIN_MANIFEST" \
VAL_MANIFEST="${VAL_MANIFEST:-$ROOT/results/lockbox_manifests_v8/v8_ccnews_p256d64_lockbox_validation.jsonl}" \
DEV_MANIFEST="$EVAL_MANIFEST" \
TAG_PREFIX="$TAG_PREFIX" \
PROMPT_LEN="$PROMPT_LEN" \
DECODE_LEN="$DECODE_LEN" \
BANK_SIZES="$BANK_SIZES" \
BATCH_SIZE="$BATCH_SIZE" \
bash "$ROOT/scripts/run_hetero_bank_audit_v7.sh"

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
    OUTPUT_SUBDIR="$OUTPUT_SUBDIR" \
    PLOT_PREFIX="$PLOT_PREFIX" \
    BATCH_SIZE="$BATCH_SIZE" \
    TRAIN_TAG="${TAG_PREFIX}_step${step}_train2048" \
    EVAL_TAG="${TAG_PREFIX}_step${step}_dev2048" \
    OUTPUT_PREFIX="${OUTPUT_PREFIX_BASE}_step${step}_dev2048_b${bank_size}" \
    bash "$ROOT/scripts/run_hetero_selector_scaling_v7.sh"
  done
done

echo "[ccnews-shortlist-full-v8] complete"
