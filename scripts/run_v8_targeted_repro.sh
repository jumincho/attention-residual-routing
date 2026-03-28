#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
SEED="${SEED:?set SEED=43|44}"
STEP="${STEP:?set STEP=3500|5500|6000}"
MODEL_NAME="${MODEL_NAME:?set MODEL_NAME=hgb_pair|retrieval_rerank_top4}"
FEATURE_MODE="${FEATURE_MODE:-attnres}"
BANK_SIZE="${BANK_SIZE:-32}"
FINAL_SPLITS="${FINAL_SPLITS:-final_A final_B final_C}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
REPRO_TAG="${REPRO_TAG:-v9_repro_seed${SEED}_step${STEP}_b${BANK_SIZE}_${MODEL_NAME}_${FEATURE_MODE}}"
LOG_DIR="${LOG_DIR:-$ROOT/results/v8_forensics/logs}"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/${REPRO_TAG}.log") 2>&1

case "$SEED" in
  42) EXPERIMENT_DIR="$ROOT/results/scale24x512_ccnews_attnres_dense_v7" ;;
  43) EXPERIMENT_DIR="$ROOT/results/scale24x512_ccnews_attnres_seed43_v8" ;;
  44) EXPERIMENT_DIR="$ROOT/results/scale24x512_ccnews_attnres_seed44_v8" ;;
  *) echo "unsupported seed ${SEED}" >&2; exit 1 ;;
esac

CHECKPOINT="$EXPERIMENT_DIR/checkpoint_step_$(printf '%06d' "$STEP").pt"
TRAIN_TAG="${TRAIN_TAG:-${REPRO_TAG}_train}"
VAL_TAG="${VAL_TAG:-${REPRO_TAG}_val}"

echo "[v8-targeted-repro] seed=${SEED} step=${STEP} model=${MODEL_NAME} feature_mode=${FEATURE_MODE} bank_size=${BANK_SIZE}"
echo "[v8-targeted-repro] checkpoint=${CHECKPOINT}"

for split in $FINAL_SPLITS; do
  OUTPUT_PREFIX="${REPRO_TAG}_${split}"
  SUMMARY_PATH="$ROOT/results/regret_reduction_v8/${OUTPUT_PREFIX}_${FEATURE_MODE}_summary.csv"

  if [[ "$SKIP_EXISTING" == "1" && -f "$SUMMARY_PATH" ]]; then
    echo "[v8-targeted-repro] skip split=${split} existing=${SUMMARY_PATH}"
    continue
  fi

  echo "[v8-targeted-repro] running split=${split}"
  CHECKPOINT="$CHECKPOINT" \
  CORPUS_TAG="v8_ccnews_p256d64_lockbox" \
  FINAL_SPLIT="$split" \
  BANK_TAG="${REPRO_TAG}_${split}_bank" \
  TRAIN_TAG="$TRAIN_TAG" \
  VAL_TAG="$VAL_TAG" \
  FINAL_TAG="${REPRO_TAG}_${split}" \
  TRAIN_MANIFEST="$ROOT/results/lockbox_manifests_v8/v8_ccnews_p256d64_lockbox_train.jsonl" \
  VAL_MANIFEST="$ROOT/results/lockbox_manifests_v8/v8_ccnews_p256d64_lockbox_validation.jsonl" \
  FINAL_MANIFEST="$ROOT/results/lockbox_manifests_v8/v8_ccnews_p256d64_lockbox_${split}.jsonl" \
  BANK_SIZE="$BANK_SIZE" \
  SKIP_COUNT=1 \
  FEATURE_MODE="$FEATURE_MODE" \
  SELECTOR_MODEL="$MODEL_NAME" \
  OUTPUT_PREFIX="$OUTPUT_PREFIX" \
  DEPLOY_NUM_SEQUENCES="${DEPLOY_NUM_SEQUENCES:-256}" \
  DEPLOY_BATCH_SIZE="${DEPLOY_BATCH_SIZE:-16}" \
  DEPLOY_TIMING_REPEATS="${DEPLOY_TIMING_REPEATS:-5}" \
  ORACLE_BATCH_SIZE="${ORACLE_BATCH_SIZE:-64}" \
  ORACLE_NUM_CHUNKS="${ORACLE_NUM_CHUNKS:-4}" \
  TRAIN_TOTAL_SHARDS="${TRAIN_TOTAL_SHARDS:-8}" \
  VAL_TOTAL_SHARDS="${VAL_TOTAL_SHARDS:-4}" \
  FINAL_TOTAL_SHARDS="${FINAL_TOTAL_SHARDS:-4}" \
  bash "$ROOT/scripts/run_locked_final_eval_v8.sh"
done

echo "[v8-targeted-repro] complete"
