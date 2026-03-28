#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
BANK_SIZE="${BANK_SIZE:-32}"
FEATURE_MODE="${FEATURE_MODE:-full}"
STEPS_RAW="${STEPS:-1000 2000 3000 5000}"

read -r -a STEPS_ARR <<<"$STEPS_RAW"

for STEP in "${STEPS_ARR[@]}"; do
  BANK_TAG="${BANK_TAG_PREFIX:-v6_ccnews_p256d64_stride160_step}${STEP}_bank"
  TRAIN_TAG="${TRAIN_TAG_PREFIX:-v6_ccnews_p256d64_stride160_step}${STEP}_train4096"
  EVAL_TAG="${EVAL_TAG_PREFIX:-v6_ccnews_p256d64_stride160_step}${STEP}_test"
  OUTPUT_PREFIX="${OUTPUT_PREFIX_PREFIX:-v6_ccnews_step}${STEP}_test1024_b${BANK_SIZE}"

  echo "[hetero-scorer-v6] step=${STEP} bank_tag=${BANK_TAG} feature_mode=${FEATURE_MODE}"
  "$PY" scripts/train_candidate_conditioned_ranker_v6.py \
    --bank-tag "$BANK_TAG" \
    --bank-size "$BANK_SIZE" \
    --train-tags "$TRAIN_TAG" \
    --eval-tags "$EVAL_TAG" \
    --hidden-train-tags "$TRAIN_TAG" \
    --hidden-eval-tags "$EVAL_TAG" \
    --output-tag "$OUTPUT_PREFIX" \
    --feature-mode "$FEATURE_MODE"
done

echo "[hetero-scorer-v6] complete"
