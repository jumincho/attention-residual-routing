#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

SEEDS_RAW="${SEEDS:-42 43 44}"
read -r -a SEEDS <<<"$SEEDS_RAW"

for seed in "${SEEDS[@]}"; do
  case "$seed" in
    42)
      experiment_dir="${EXPERIMENT_DIR_SEED42:-$ROOT/results/scale24x512_ccnews_attnres_dense_v7}"
      checkpoint_steps="${CHECKPOINT_STEPS_SEED42:-3000 3500 4000 4500 5000 5500 6000}"
      ;;
    43)
      experiment_dir="${EXPERIMENT_DIR_SEED43:-$ROOT/results/scale24x512_ccnews_attnres_seed43_v8}"
      checkpoint_steps="${CHECKPOINT_STEPS_SEED43:-2500 3000 3500 4000 4500 5000 5500 6000}"
      ;;
    44)
      experiment_dir="${EXPERIMENT_DIR_SEED44:-$ROOT/results/scale24x512_ccnews_attnres_seed44_v8}"
      checkpoint_steps="${CHECKPOINT_STEPS_SEED44:-2500 3000 3500 4000 4500 5000 5500 6000}"
      ;;
    *)
      echo "unsupported seed ${seed}" >&2
      exit 1
      ;;
  esac

  echo "[ccnews-multiseed-fast-v8] seed=${seed} steps=${checkpoint_steps}"
  EXPERIMENT_DIR="$experiment_dir" \
  CHECKPOINT_STEPS="$checkpoint_steps" \
  TRAIN_MANIFEST="${TRAIN_MANIFEST:-$ROOT/results/lockbox_manifests_v8/v8_ccnews_p256d64_lockbox_train512.jsonl}" \
  VAL_MANIFEST="${VAL_MANIFEST:-$ROOT/results/lockbox_manifests_v8/v8_ccnews_p256d64_lockbox_validation512.jsonl}" \
  DEV_MANIFEST="${DEV_MANIFEST:-$ROOT/results/lockbox_manifests_v8/v8_ccnews_p256d64_lockbox_dev_select512.jsonl}" \
  TAG_PREFIX="v8_ccnews_seed${seed}_fast" \
  BANK_SIZES="${BANK_SIZES:-32}" \
  FAST_MODE="${FAST_MODE:-1}" \
  OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-regret_reduction_v8}" \
  PLOT_PREFIX="${PLOT_PREFIX:-regret_reduction_v8}" \
  SUMMARY_DIR="${SUMMARY_DIR:-$ROOT/results/ccnews_multiseed_multisplit_v8}" \
  BATCH_SIZE="${BATCH_SIZE:-128}" \
  bash "$ROOT/scripts/run_ccnews_dev_selection_v8.sh"
done

echo "[ccnews-multiseed-fast-v8] complete"
