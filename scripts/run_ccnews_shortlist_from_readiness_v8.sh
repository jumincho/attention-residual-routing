#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

SEED="${SEED:?set SEED=42}"
TOP_N="${TOP_N:-2}"

case "$SEED" in
  42)
    EXPERIMENT_DIR="${EXPERIMENT_DIR:-$ROOT/results/scale24x512_ccnews_attnres_dense_v7}"
    TAG_PREFIX="${TAG_PREFIX:-v8_ccnews_seed42_fast}"
    OUTPUT_PREFIX_BASE="${OUTPUT_PREFIX_BASE:-v8_ccnews_seed42_full}"
    ;;
  43)
    EXPERIMENT_DIR="${EXPERIMENT_DIR:-$ROOT/results/scale24x512_ccnews_attnres_seed43_v8}"
    TAG_PREFIX="${TAG_PREFIX:-v8_ccnews_seed43_fast}"
    OUTPUT_PREFIX_BASE="${OUTPUT_PREFIX_BASE:-v8_ccnews_seed43_full}"
    ;;
  44)
    EXPERIMENT_DIR="${EXPERIMENT_DIR:-$ROOT/results/scale24x512_ccnews_attnres_seed44_v8}"
    TAG_PREFIX="${TAG_PREFIX:-v8_ccnews_seed44_fast}"
    OUTPUT_PREFIX_BASE="${OUTPUT_PREFIX_BASE:-v8_ccnews_seed44_full}"
    ;;
  *)
    echo "unsupported seed ${SEED}" >&2
    exit 1
    ;;
esac

READINESS_PATH="${READINESS_PATH:-$ROOT/results/readiness_v8/${TAG_PREFIX}_main_routing_readiness_v4.csv}"
CHECKPOINT_STEPS="$("$ROOT/.venv/bin/python" - <<PY
import pandas as pd
df = pd.read_csv("${READINESS_PATH}")
print(" ".join(str(int(x)) for x in df.sort_values("routing_readiness_v4", ascending=False).head(${TOP_N})["step"].tolist()))
PY
)"

echo "[ccnews-shortlist-from-readiness-v8] seed=${SEED} shortlist=${CHECKPOINT_STEPS}"

EXPERIMENT_DIR="$EXPERIMENT_DIR" \
CHECKPOINT_STEPS="$CHECKPOINT_STEPS" \
TRAIN_MANIFEST="${TRAIN_MANIFEST:-$ROOT/results/lockbox_manifests_v8/v8_ccnews_p256d64_lockbox_train2048.jsonl}" \
VAL_MANIFEST="${VAL_MANIFEST:-$ROOT/results/lockbox_manifests_v8/v8_ccnews_p256d64_lockbox_validation.jsonl}" \
EVAL_MANIFEST="${EVAL_MANIFEST:-$ROOT/results/lockbox_manifests_v8/v8_ccnews_p256d64_lockbox_dev_select2048.jsonl}" \
TAG_PREFIX="$TAG_PREFIX" \
OUTPUT_PREFIX_BASE="$OUTPUT_PREFIX_BASE" \
BANK_SIZES="${BANK_SIZES:-32 64}" \
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-regret_reduction_v8}" \
PLOT_PREFIX="${PLOT_PREFIX:-regret_reduction_v8}" \
BATCH_SIZE="${BATCH_SIZE:-128}" \
bash "$ROOT/scripts/run_ccnews_shortlist_full_scorer_v8.sh"
