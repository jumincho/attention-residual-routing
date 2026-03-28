#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$ROOT/results/scale24x512_ccnews_attnres_v5}"
TAGS_RAW="${TAGS:-v6_ccnews_p256d64_stride160_step1000_val v6_ccnews_p256d64_stride160_step2000_val v6_ccnews_p256d64_stride160_step3000_val v6_ccnews_p256d64_stride160_step5000_val}"
BANK_SIZE="${BANK_SIZE:-32}"
BANK_SKIP="${BANK_SKIP:-1}"
FEATURE_MODE="${FEATURE_MODE:-attnres}"
SELECTOR_PREFIX="${SELECTOR_PREFIX:-}"
if [[ -z "$SELECTOR_PREFIX" ]]; then
  SELECTOR_PREFIX='v6_ccnews_step{step}_test1024_b32'
fi
OUTPUT_TAG="${OUTPUT_TAG:-v6_ccnews_main}"

read -r -a TAGS_ARR <<<"$TAGS_RAW"

"$PY" scripts/compile_routing_readiness_v6.py \
  --tags "${TAGS_ARR[@]}" \
  --experiment-dir "$EXPERIMENT_DIR" \
  --bank-size "$BANK_SIZE" \
  --bank-skip "$BANK_SKIP" \
  --feature-mode "$FEATURE_MODE" \
  --selector-prefix "$SELECTOR_PREFIX" \
  --selector-dir "results/ranker_v5" \
  --output-tag "$OUTPUT_TAG"
