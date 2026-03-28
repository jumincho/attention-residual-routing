#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
TAG="${TAG:-v9_ccnews_p256d64_lockbox}"
DATASET_NAME="${DATASET_NAME:-cc_news}"
PROMPT_LEN="${PROMPT_LEN:-256}"
DECODE_LEN="${DECODE_LEN:-64}"
STRIDE="${STRIDE:-160}"
MAX_WINDOWS_PER_DOC="${MAX_WINDOWS_PER_DOC:-16}"
SEED="${SEED:-20260326}"
TRAIN_COUNT="${TRAIN_COUNT:-8192}"
VALIDATION_COUNT="${VALIDATION_COUNT:-2048}"
DEV_COUNT="${DEV_COUNT:-4096}"
FINAL_COUNT="${FINAL_COUNT:-4096}"
OUT_DIR="${OUT_DIR:-$ROOT/results/lockbox_manifests_v9}"

"$PY" "$ROOT/scripts/build_lockbox_manifests_v9.py" \
  --dataset-name "$DATASET_NAME" \
  --prompt-len "$PROMPT_LEN" \
  --decode-len "$DECODE_LEN" \
  --stride "$STRIDE" \
  --max-windows-per-doc "$MAX_WINDOWS_PER_DOC" \
  --seed "$SEED" \
  --tag "$TAG" \
  --output-dir "$OUT_DIR" \
  --partition "train=${TRAIN_COUNT}" \
  --partition "validation=${VALIDATION_COUNT}" \
  --partition "dev_select_v9=${DEV_COUNT}" \
  --partition "final_D=${FINAL_COUNT}" \
  --partition "final_E=${FINAL_COUNT}" \
  --partition "final_F=${FINAL_COUNT}"

"$PY" "$ROOT/scripts/subset_manifest.py" \
  --input "$OUT_DIR/${TAG}_train.jsonl" \
  --output "$OUT_DIR/${TAG}_train2048.jsonl" \
  --target-count 2048

"$PY" "$ROOT/scripts/subset_manifest.py" \
  --input "$OUT_DIR/${TAG}_train.jsonl" \
  --output "$OUT_DIR/${TAG}_train4096.jsonl" \
  --target-count 4096

"$PY" "$ROOT/scripts/subset_manifest.py" \
  --input "$OUT_DIR/${TAG}_validation.jsonl" \
  --output "$OUT_DIR/${TAG}_validation512.jsonl" \
  --target-count 512

"$PY" "$ROOT/scripts/subset_manifest.py" \
  --input "$OUT_DIR/${TAG}_dev_select_v9.jsonl" \
  --output "$OUT_DIR/${TAG}_dev_select_v9_2048.jsonl" \
  --target-count 2048

echo "[build-lockbox-v9] complete tag=${TAG}"
