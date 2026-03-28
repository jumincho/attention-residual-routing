#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
PER_SEQUENCE_CSV="${PER_SEQUENCE_CSV:?set PER_SEQUENCE_CSV=/abs/path/to/per_sequence.csv}"
FEATURE_MODE="${FEATURE_MODE:-attnres}"
SKIP_COUNT="${SKIP_COUNT:-1}"
SELECTED_MODEL="${SELECTED_MODEL:-hgb_pair}"
OUTPUT_TAG="${OUTPUT_TAG:?set OUTPUT_TAG=v6_ccnews_step5000_skip1_hgb_pair}"

"$PY" scripts/evaluate_ranker_calibrated_abstention_v6.py \
  --per-sequence-csv "$PER_SEQUENCE_CSV" \
  --feature-mode "$FEATURE_MODE" \
  --skip-count "$SKIP_COUNT" \
  --selected-model "$SELECTED_MODEL" \
  --output-tag "$OUTPUT_TAG"

echo "[calibrated-abstention-v6] complete"
