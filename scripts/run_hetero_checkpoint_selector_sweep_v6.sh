#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$ROOT/results/scale24x512_ccnews_attnres_v5}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-$ROOT/results/selector_data_scale/v6_ccnews_p256d64_stride160_train4096.jsonl}"
EVAL_MANIFEST="${EVAL_MANIFEST:-$ROOT/results/selector_data_scale/v6_ccnews_p256d64_stride160_test.jsonl}"
PROMPT_LEN="${PROMPT_LEN:-256}"
DECODE_LEN="${DECODE_LEN:-64}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TRAIN_SHARDS="${TRAIN_SHARDS:-8}"
EVAL_SHARDS="${EVAL_SHARDS:-4}"
BANK_SIZE="${BANK_SIZE:-32}"
RUN_HIDDEN="${RUN_HIDDEN:-0}"
RUN_V6_FULL="${RUN_V6_FULL:-0}"
STEPS_RAW="${STEPS:-1000 2000 3000 5000}"

read -r -a STEPS_ARR <<<"$STEPS_RAW"

checkpoint_for_step() {
  local step="$1"
  if [[ "$step" == "3000" ]]; then
    echo "$EXPERIMENT_DIR/best_checkpoint.pt"
  else
    printf '%s/checkpoint_step_%06d.pt\n' "$EXPERIMENT_DIR" "$step"
  fi
}

for STEP in "${STEPS_ARR[@]}"; do
  CHECKPOINT="$(checkpoint_for_step "$STEP")"
  if [[ ! -f "$CHECKPOINT" ]]; then
    echo "[selector-sweep-v6] missing checkpoint for step=${STEP}: ${CHECKPOINT}" >&2
    exit 1
  fi

  BANK_TAG="${BANK_TAG_PREFIX:-v6_ccnews_p256d64_stride160_step}${STEP}_bank"
  TRAIN_TAG="${TRAIN_TAG_PREFIX:-v6_ccnews_p256d64_stride160_step}${STEP}_train4096"
  EVAL_TAG="${EVAL_TAG_PREFIX:-v6_ccnews_p256d64_stride160_step}${STEP}_test"
  OUTPUT_PREFIX="${OUTPUT_PREFIX_PREFIX:-v6_ccnews_step}${STEP}_test1024_b${BANK_SIZE}"

  echo "[selector-sweep-v6] step=${STEP} checkpoint=${CHECKPOINT}"
  CHECKPOINT="$CHECKPOINT" \
  BANK_TAG="$BANK_TAG" \
  TRAIN_MANIFEST="$TRAIN_MANIFEST" \
  EVAL_MANIFEST="$EVAL_MANIFEST" \
  PROMPT_LEN="$PROMPT_LEN" \
  DECODE_LEN="$DECODE_LEN" \
  BATCH_SIZE="$BATCH_SIZE" \
  TRAIN_SHARDS="$TRAIN_SHARDS" \
  EVAL_SHARDS="$EVAL_SHARDS" \
  BANK_SIZE="$BANK_SIZE" \
  RUN_HIDDEN="$RUN_HIDDEN" \
  TRAIN_TAG="$TRAIN_TAG" \
  EVAL_TAG="$EVAL_TAG" \
  OUTPUT_PREFIX="$OUTPUT_PREFIX" \
  bash "$ROOT/scripts/run_hetero_selector_scaling_v6.sh"

  if [[ "$RUN_V6_FULL" == "1" ]]; then
    if [[ ! -f "$ROOT/results/hetero_scorer_v6/${OUTPUT_PREFIX}_full_summary.csv" ]]; then
      "$PY" scripts/train_candidate_conditioned_ranker_v6.py \
        --bank-tag "$BANK_TAG" \
        --bank-size "$BANK_SIZE" \
        --train-tags "$TRAIN_TAG" \
        --eval-tags "$EVAL_TAG" \
        --hidden-train-tags "$TRAIN_TAG" \
        --hidden-eval-tags "$EVAL_TAG" \
        --output-tag "$OUTPUT_PREFIX" \
        --feature-mode full
    fi
  fi
done

echo "[selector-sweep-v6] complete"
