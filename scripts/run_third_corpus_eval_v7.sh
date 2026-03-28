#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
DATASET_NAME="${DATASET_NAME:-fineweb_edu_sample10bt_local_v7}"
TOKENIZER_NAME="${TOKENIZER_NAME:-openai-community/gpt2}"
PROMPT_LEN="${PROMPT_LEN:-256}"
DECODE_LEN="${DECODE_LEN:-64}"
TRAIN_TARGET="${TRAIN_TARGET:-4096}"
VAL_TARGET="${VAL_TARGET:-1024}"
DEV_TARGET="${DEV_TARGET:-1024}"
FINAL_TARGET="${FINAL_TARGET:-1024}"
MAX_WINDOWS_PER_DOC="${MAX_WINDOWS_PER_DOC:-8}"
STRIDE="${STRIDE:-160}"
LOCKBOX_TAG="${LOCKBOX_TAG:-v7_fineweb_p256d64_lockbox}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$ROOT/results/scale24x512_finewebedu_sample10bt_attnres_v7}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-1000 2000 3000}"
BANK_SIZES="${BANK_SIZES:-32 64}"

"$PY" "$ROOT/scripts/build_lockbox_manifests_v7.py" \
  --dataset-name "$DATASET_NAME" \
  --tokenizer-name "$TOKENIZER_NAME" \
  --prompt-len "$PROMPT_LEN" \
  --decode-len "$DECODE_LEN" \
  --train-target "$TRAIN_TARGET" \
  --validation-target "$VAL_TARGET" \
  --dev-test-target "$DEV_TARGET" \
  --final-test-target "$FINAL_TARGET" \
  --max-windows-per-doc "$MAX_WINDOWS_PER_DOC" \
  --stride "$STRIDE" \
  --tag "$LOCKBOX_TAG"

EXPERIMENT_DIR="$EXPERIMENT_DIR" \
CHECKPOINT_STEPS="$CHECKPOINT_STEPS" \
TRAIN_MANIFEST="$ROOT/results/lockbox_manifests_v7/${LOCKBOX_TAG}_train.jsonl" \
VAL_MANIFEST="$ROOT/results/lockbox_manifests_v7/${LOCKBOX_TAG}_validation.jsonl" \
DEV_MANIFEST="$ROOT/results/lockbox_manifests_v7/${LOCKBOX_TAG}_dev_test.jsonl" \
TAG_PREFIX="$LOCKBOX_TAG" \
BANK_SIZES="$BANK_SIZES" \
PROMPT_LEN="$PROMPT_LEN" \
DECODE_LEN="$DECODE_LEN" \
bash "$ROOT/scripts/run_hetero_bank_audit_v7.sh"

for step in $CHECKPOINT_STEPS; do
  for bank_size in $BANK_SIZES; do
    CHECKPOINT="$EXPERIMENT_DIR/checkpoint_step_$(printf '%06d' "$step").pt" \
    BANK_TAG="${LOCKBOX_TAG}_step${step}_bank" \
    TRAIN_MANIFEST="$ROOT/results/lockbox_manifests_v7/${LOCKBOX_TAG}_train.jsonl" \
    EVAL_MANIFEST="$ROOT/results/lockbox_manifests_v7/${LOCKBOX_TAG}_dev_test.jsonl" \
    PROMPT_LEN="$PROMPT_LEN" \
    DECODE_LEN="$DECODE_LEN" \
    BANK_SIZE="$bank_size" \
    FAST_MODE=1 \
    TRAIN_TAG="${LOCKBOX_TAG}_step${step}_train4096" \
    EVAL_TAG="${LOCKBOX_TAG}_step${step}_dev1024" \
    OUTPUT_PREFIX="v7_fineweb_step${step}_dev1024_b${bank_size}" \
    bash "$ROOT/scripts/run_hetero_selector_scaling_v7.sh"
  done
done

TAGS=""
for step in $CHECKPOINT_STEPS; do
  TAGS="${TAGS} ${LOCKBOX_TAG}_step${step}_val"
done

EXPERIMENT_DIR="$EXPERIMENT_DIR" \
TAGS="${TAGS}" \
BANK_SIZE=32 \
BANK_SKIP=1 \
FEATURE_MODE=attnres \
SELECTOR_PREFIX='v7_fineweb_step{step}_dev1024_b32' \
SELECTOR_DIR='results/regret_reduction_v7' \
OUTPUT_TAG='v7_fineweb_main' \
bash "$ROOT/scripts/run_routing_readiness_v7.sh"

echo "[third-corpus-eval-v7] complete"
