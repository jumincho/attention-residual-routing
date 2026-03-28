#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
SEED="${SEED:?set SEED=44}"
STEP="${STEP:?set STEP=3500}"
MODEL_NAME="${MODEL_NAME:?set MODEL_NAME=retrieval_rerank_top4}"
FEATURE_MODE="${FEATURE_MODE:-attnres}"
BANK_SIZE="${BANK_SIZE:-32}"
SPLITS_RAW="${SPLITS:-final_A final_B final_C}"
DEFAULT_ORIGINAL_TEMPLATE="v8_locked_seed${SEED}_{split}_step${STEP}_b${BANK_SIZE}_${MODEL_NAME}_${FEATURE_MODE}"
DEFAULT_RERUN_TEMPLATE="v9_repro_seed${SEED}_step${STEP}_b${BANK_SIZE}_${MODEL_NAME}_${FEATURE_MODE}_shard32_{split}"
ORIGINAL_TEMPLATE="${ORIGINAL_TEMPLATE:-$DEFAULT_ORIGINAL_TEMPLATE}"
RERUN_TEMPLATE="${RERUN_TEMPLATE:-$DEFAULT_RERUN_TEMPLATE}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/results/v8_forensics}"

mkdir -p "$OUTPUT_DIR"
read -r -a SPLITS <<<"$SPLITS_RAW"

for split in "${SPLITS[@]}"; do
  original_prefix="${ORIGINAL_TEMPLATE/\{split\}/$split}"
  rerun_prefix="${RERUN_TEMPLATE/\{split\}/$split}"
  output_csv="$OUTPUT_DIR/${rerun_prefix}_compare.csv"
  echo "[v8-repro-compare] split=${split} original=${original_prefix} rerun=${rerun_prefix}"
  "$PY" "$ROOT/scripts/compare_v8_repro_outputs.py" \
    --original-prefix "$original_prefix" \
    --rerun-prefix "$rerun_prefix" \
    --feature-mode "$FEATURE_MODE" \
    --model-name "$MODEL_NAME" \
    --bank-size "$BANK_SIZE" \
    --skip-count 1 \
    --output-csv "$output_csv"
done

echo "[v8-repro-compare] complete"
