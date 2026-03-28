#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

EXPERIMENT_DIR="${EXPERIMENT_DIR:-$ROOT/results/scale24x512_ccnews_attnres_signalpres_v6}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-1500 2000 2500 3000}"
BANK_SIZE="${BANK_SIZE:-32}"
FEATURE_MODE="${FEATURE_MODE:-attnres}"
RUN_BANK_AUDIT="${RUN_BANK_AUDIT:-1}"
RUN_SCORER="${RUN_SCORER:-1}"
RUN_READINESS="${RUN_READINESS:-1}"

MAIN_TAG="${MAIN_TAG:-v6sp_ccnews_p256d64_stride160}"
TRAIN_TAG_PREFIX="${TRAIN_TAG_PREFIX:-${MAIN_TAG}_step}"
EVAL_TAG_PREFIX="${EVAL_TAG_PREFIX:-${MAIN_TAG}_step}"
BANK_TAG_PREFIX="${BANK_TAG_PREFIX:-${MAIN_TAG}_step}"
OUTPUT_PREFIX_PREFIX="${OUTPUT_PREFIX_PREFIX:-v6sp_ccnews_step}"
READINESS_OUTPUT_TAG="${READINESS_OUTPUT_TAG:-v6sp_ccnews_main}"

if [[ "$RUN_BANK_AUDIT" == "1" ]]; then
  EXPERIMENT_DIR="$EXPERIMENT_DIR" \
  CHECKPOINT_STEPS="$CHECKPOINT_STEPS" \
  MAIN_TAG="$MAIN_TAG" \
  MAIN_TRAIN_TARGET=4096 \
  MAIN_VAL_TARGET=1024 \
  MAIN_TEST_TARGET=1024 \
  AUX_STEPS="" \
  AUX_TAG="unused_v6sp_aux" \
  bash "$ROOT/scripts/run_hetero_bank_audit_v6.sh"
fi

if [[ "$RUN_SCORER" == "1" ]]; then
  for step in $CHECKPOINT_STEPS; do
    CHECKPOINT="$EXPERIMENT_DIR/checkpoint_step_$(printf '%06d' "$step").pt" \
    BANK_TAG="${BANK_TAG_PREFIX}${step}_bank" \
    TRAIN_MANIFEST="$ROOT/results/selector_data_scale/${MAIN_TAG}_train.jsonl" \
    EVAL_MANIFEST="$ROOT/results/selector_data_scale/${MAIN_TAG}_test.jsonl" \
    BANK_SIZE="$BANK_SIZE" \
    RUN_HIDDEN=0 \
    TRAIN_TAG="${TRAIN_TAG_PREFIX}${step}_train4096" \
    EVAL_TAG="${EVAL_TAG_PREFIX}${step}_test" \
    OUTPUT_PREFIX="${OUTPUT_PREFIX_PREFIX}${step}_test1024_b${BANK_SIZE}" \
    bash "$ROOT/scripts/run_hetero_selector_scaling_v6.sh"
  done
fi

if [[ "$RUN_READINESS" == "1" ]]; then
  tag_list=()
  for step in $CHECKPOINT_STEPS; do
    tag_list+=("${MAIN_TAG}_step${step}_val")
  done
  TAGS="${tag_list[*]}" \
  EXPERIMENT_DIR="$EXPERIMENT_DIR" \
  BANK_SIZE="$BANK_SIZE" \
  BANK_SKIP=1 \
  FEATURE_MODE="$FEATURE_MODE" \
  SELECTOR_PREFIX="${OUTPUT_PREFIX_PREFIX}{step}_test1024_b${BANK_SIZE}" \
  OUTPUT_TAG="$READINESS_OUTPUT_TAG" \
  bash "$ROOT/scripts/run_routing_readiness_v6.sh"
fi

echo "[signal-preserving-eval-v6] complete"
