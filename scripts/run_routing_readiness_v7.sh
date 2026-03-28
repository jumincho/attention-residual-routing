#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:?set EXPERIMENT_DIR=/abs/path/to/experiment_dir}"
TAGS_RAW="${TAGS:?set TAGS='v7_ccnews_p256d64_lockbox_step3000_val v7_ccnews_p256d64_lockbox_step3500_val'}"
BANK_SIZE="${BANK_SIZE:-32}"
BANK_SKIP="${BANK_SKIP:-1}"
FEATURE_MODE="${FEATURE_MODE:-attnres}"
SELECTOR_PREFIX="${SELECTOR_PREFIX:?set SELECTOR_PREFIX=v7_ccnews_step{step}_dev2048_b32}"
SELECTOR_DIR="${SELECTOR_DIR:-results/regret_reduction_v7}"
OUTPUT_TAG="${OUTPUT_TAG:?set OUTPUT_TAG=v7_ccnews_main}"

read -r -a TAGS_ARR <<<"$TAGS_RAW"

"$PY" scripts/compile_routing_readiness_v7.py \
  --tags "${TAGS_ARR[@]}" \
  --experiment-dir "$EXPERIMENT_DIR" \
  --bank-size "$BANK_SIZE" \
  --bank-skip "$BANK_SKIP" \
  --feature-mode "$FEATURE_MODE" \
  --selector-prefix "$SELECTOR_PREFIX" \
  --selector-dir "$SELECTOR_DIR" \
  --output-tag "$OUTPUT_TAG"
