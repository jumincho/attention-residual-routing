#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT=/abs/path/to/checkpoint.pt}"
BANK_TAG="${BANK_TAG:?set BANK_TAG=v6_ccnews_p256d64_stride160_step5000_bank}"
TRAIN_TAGS="${TRAIN_TAGS:?set TRAIN_TAGS=v6_ccnews_p256d64_stride160_step5000_train4096}"
EVAL_TAGS="${EVAL_TAGS:?set EVAL_TAGS=v6_ccnews_p256d64_stride160_step5000_test}"
MANIFEST_PATH="${MANIFEST_PATH:?set MANIFEST_PATH=$ROOT/results/selector_data_scale/v6_ccnews_p256d64_stride160_test.jsonl}"
OUTPUT_TAG="${OUTPUT_TAG:?set OUTPUT_TAG=v6_ccnews_step5000_skip1_hgb_pair}"
FEATURE_MODE="${FEATURE_MODE:-attnres}"
SELECTED_MODEL="${SELECTED_MODEL:-hgb_pair}"
SKIP_COUNT="${SKIP_COUNT:-1}"
BANK_SIZE="${BANK_SIZE:-32}"
NUM_SEQUENCES="${NUM_SEQUENCES:-256}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TIMING_REPEATS="${TIMING_REPEATS:-5}"
HIDDEN_TRAIN_TAGS="${HIDDEN_TRAIN_TAGS:-}"
HIDDEN_EVAL_TAGS="${HIDDEN_EVAL_TAGS:-}"

cmd=(
  "$PY" scripts/evaluate_deployment_measurement_v6.py
  --checkpoint "$CHECKPOINT"
  --bank-tag "$BANK_TAG"
  --bank-size "$BANK_SIZE"
  --skip-count "$SKIP_COUNT"
  --train-tags $TRAIN_TAGS
  --eval-tags $EVAL_TAGS
  --manifest-path "$MANIFEST_PATH"
  --feature-mode "$FEATURE_MODE"
  --selected-model "$SELECTED_MODEL"
  --output-tag "$OUTPUT_TAG"
  --num-sequences "$NUM_SEQUENCES"
  --batch-size "$BATCH_SIZE"
  --timing-repeats "$TIMING_REPEATS"
  --precision fp16
  --device cuda
)

if [[ -n "${HIDDEN_TRAIN_TAGS// }" ]]; then
  # shellcheck disable=SC2206
  ht=($HIDDEN_TRAIN_TAGS)
  cmd+=(--hidden-train-tags "${ht[@]}")
fi
if [[ -n "${HIDDEN_EVAL_TAGS// }" ]]; then
  # shellcheck disable=SC2206
  he=($HIDDEN_EVAL_TAGS)
  cmd+=(--hidden-eval-tags "${he[@]}")
fi

"${cmd[@]}"

echo "[deployment-measurement-v6] complete"
