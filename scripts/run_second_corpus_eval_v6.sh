#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$ROOT/results/scale24x512_wikitext103_attnres_v6}"
DATASET_NAME="${DATASET_NAME:-wikitext103}"
TOKENIZER_NAME="${TOKENIZER_NAME:-openai-community/gpt2}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-1000 2000 3000}"
BANK_SIZE="${BANK_SIZE:-32}"
FEATURE_MODE="${FEATURE_MODE:-attnres}"
PROMPT_LEN="${PROMPT_LEN:-256}"
DECODE_LEN="${DECODE_LEN:-64}"
STRIDE="${STRIDE:-80}"
MAX_WINDOWS_PER_DOC="${MAX_WINDOWS_PER_DOC:-32}"
TRAIN_TARGET="${TRAIN_TARGET:-4096}"
VAL_TARGET="${VAL_TARGET:-1024}"
TEST_TARGET="${TEST_TARGET:-1024}"
SETTING_TAG="${SETTING_TAG:-v6wiki_p256d64_stride080}"
RUN_MANIFEST="${RUN_MANIFEST:-1}"
RUN_BANK_AUDIT="${RUN_BANK_AUDIT:-1}"
RUN_SCORER="${RUN_SCORER:-1}"
RUN_READINESS="${RUN_READINESS:-1}"
READINESS_OUTPUT_TAG="${READINESS_OUTPUT_TAG:-v6wiki_main}"

if [[ "$RUN_MANIFEST" == "1" ]]; then
  "$PY" "$ROOT/scripts/build_selector_sequence_manifest.py" \
    --dataset-name "$DATASET_NAME" \
    --tokenizer-name "$TOKENIZER_NAME" \
    --prompt-len "$PROMPT_LEN" \
    --decode-len "$DECODE_LEN" \
    --train-target "$TRAIN_TARGET" \
    --validation-target "$VAL_TARGET" \
    --test-target "$TEST_TARGET" \
    --max-windows-per-doc "$MAX_WINDOWS_PER_DOC" \
    --stride "$STRIDE" \
    --tag "$SETTING_TAG"
fi

if [[ "$RUN_BANK_AUDIT" == "1" ]]; then
  EXPERIMENT_DIR="$EXPERIMENT_DIR" \
  CHECKPOINT_STEPS="$CHECKPOINT_STEPS" \
  MAIN_TAG="$SETTING_TAG" \
  MAIN_PROMPT_LEN="$PROMPT_LEN" \
  MAIN_DECODE_LEN="$DECODE_LEN" \
  MAIN_STRIDE="$STRIDE" \
  MAIN_TRAIN_TARGET="$TRAIN_TARGET" \
  MAIN_VAL_TARGET="$VAL_TARGET" \
  MAIN_TEST_TARGET="$TEST_TARGET" \
  AUX_STEPS="" \
  AUX_TAG="unused_v6owt_aux" \
  bash "$ROOT/scripts/run_hetero_bank_audit_v6.sh"
fi

if [[ "$RUN_SCORER" == "1" ]]; then
  for step in $CHECKPOINT_STEPS; do
    CHECKPOINT="$EXPERIMENT_DIR/checkpoint_step_$(printf '%06d' "$step").pt" \
    BANK_TAG="${SETTING_TAG}_step${step}_bank" \
    TRAIN_MANIFEST="$ROOT/results/selector_data_scale/${SETTING_TAG}_train.jsonl" \
    EVAL_MANIFEST="$ROOT/results/selector_data_scale/${SETTING_TAG}_test.jsonl" \
    BANK_SIZE="$BANK_SIZE" \
    RUN_HIDDEN=0 \
    TRAIN_TAG="${SETTING_TAG}_step${step}_train4096" \
    EVAL_TAG="${SETTING_TAG}_step${step}_test" \
    OUTPUT_PREFIX="v6wiki_step${step}_test1024_b${BANK_SIZE}" \
    bash "$ROOT/scripts/run_hetero_selector_scaling_v6.sh"
  done
fi

if [[ "$RUN_READINESS" == "1" ]]; then
  tag_list=()
  for step in $CHECKPOINT_STEPS; do
    tag_list+=("${SETTING_TAG}_step${step}_val")
  done
  TAGS="${tag_list[*]}" \
  EXPERIMENT_DIR="$EXPERIMENT_DIR" \
  BANK_SIZE="$BANK_SIZE" \
  BANK_SKIP=1 \
  FEATURE_MODE="$FEATURE_MODE" \
  SELECTOR_PREFIX="v6wiki_step{step}_test1024_b${BANK_SIZE}" \
  OUTPUT_TAG="$READINESS_OUTPUT_TAG" \
  bash "$ROOT/scripts/run_routing_readiness_v6.sh"
fi

echo "[second-corpus-eval-v6] complete"
