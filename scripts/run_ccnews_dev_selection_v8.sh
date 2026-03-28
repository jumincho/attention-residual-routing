#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

EXPERIMENT_DIR="${EXPERIMENT_DIR:?set EXPERIMENT_DIR=/abs/path/to/experiment_dir}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-2500 3000 3500 4000 4500 5000 5500 6000}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-$ROOT/results/lockbox_manifests_v8/v8_ccnews_p256d64_lockbox_train.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-$ROOT/results/lockbox_manifests_v8/v8_ccnews_p256d64_lockbox_validation.jsonl}"
DEV_MANIFEST="${DEV_MANIFEST:-$ROOT/results/lockbox_manifests_v8/v8_ccnews_p256d64_lockbox_dev_select.jsonl}"
TAG_PREFIX="${TAG_PREFIX:-v8_ccnews_p256d64_lockbox}"
OUTPUT_PREFIX_BASE="${OUTPUT_PREFIX_BASE:-${TAG_PREFIX}}"
BANK_SIZES="${BANK_SIZES:-32 64}"
PROMPT_LEN="${PROMPT_LEN:-256}"
DECODE_LEN="${DECODE_LEN:-64}"
FAST_MODE="${FAST_MODE:-1}"
SUMMARY_DIR="${SUMMARY_DIR:-$ROOT/results/ccnews_multiseed_multisplit_v8}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-regret_reduction_v8}"
PLOT_PREFIX="${PLOT_PREFIX:-regret_reduction_v8}"

EXPERIMENT_DIR="$EXPERIMENT_DIR" \
CHECKPOINT_STEPS="$CHECKPOINT_STEPS" \
TRAIN_MANIFEST="$TRAIN_MANIFEST" \
VAL_MANIFEST="$VAL_MANIFEST" \
DEV_MANIFEST="$DEV_MANIFEST" \
TAG_PREFIX="$TAG_PREFIX" \
PROMPT_LEN="$PROMPT_LEN" \
DECODE_LEN="$DECODE_LEN" \
BANK_SIZES="$BANK_SIZES" \
SUMMARY_DIR="$SUMMARY_DIR" \
bash "$ROOT/scripts/run_hetero_bank_audit_v7.sh"

for step in $CHECKPOINT_STEPS; do
  for bank_size in $BANK_SIZES; do
    CHECKPOINT="$EXPERIMENT_DIR/checkpoint_step_$(printf '%06d' "$step").pt" \
    BANK_TAG="${TAG_PREFIX}_step${step}_bank" \
    TRAIN_MANIFEST="$TRAIN_MANIFEST" \
    EVAL_MANIFEST="$DEV_MANIFEST" \
    PROMPT_LEN="$PROMPT_LEN" \
    DECODE_LEN="$DECODE_LEN" \
    BANK_SIZE="$bank_size" \
    FAST_MODE="$FAST_MODE" \
    OUTPUT_SUBDIR="$OUTPUT_SUBDIR" \
    PLOT_PREFIX="$PLOT_PREFIX" \
    TRAIN_TAG="${TAG_PREFIX}_step${step}_train8192" \
    EVAL_TAG="${TAG_PREFIX}_step${step}_dev4096" \
    OUTPUT_PREFIX="${OUTPUT_PREFIX_BASE}_step${step}_dev_b${bank_size}" \
    bash "$ROOT/scripts/run_hetero_selector_scaling_v7.sh"
  done
done

TAGS=""
for step in $CHECKPOINT_STEPS; do
  TAGS="${TAGS} ${TAG_PREFIX}_step${step}_val"
done

"$ROOT/.venv/bin/python" "$ROOT/scripts/compile_routing_readiness_v8.py" \
  --tags ${TAGS} \
  --experiment-dir "$EXPERIMENT_DIR" \
  --bank-size 32 \
  --bank-skip 1 \
  --feature-mode attnres \
  --selector-prefix "${OUTPUT_PREFIX_BASE}_step{step}_dev_b32" \
  --selector-dir 'results/regret_reduction_v8' \
  --output-tag "${TAG_PREFIX}_main"

echo "[ccnews-dev-selection-v8] complete"
