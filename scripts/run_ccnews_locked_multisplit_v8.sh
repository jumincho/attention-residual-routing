#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

SELECTION_CSV="${SELECTION_CSV:?set SELECTION_CSV=/abs/path/to/selection_winners.csv}"
FINAL_SPLITS="${FINAL_SPLITS:-final_A final_B final_C}"
DEPLOY_BATCH_SIZE="${DEPLOY_BATCH_SIZE:-16}"
DEPLOY_NUM_SEQUENCES="${DEPLOY_NUM_SEQUENCES:-256}"
DEPLOY_TIMING_REPEATS="${DEPLOY_TIMING_REPEATS:-5}"

while IFS=, read -r seed step bank_size feature_mode model_name; do
  case "$seed" in
    42) EXPERIMENT_DIR="$ROOT/results/scale24x512_ccnews_attnres_dense_v7" ;;
    43) EXPERIMENT_DIR="$ROOT/results/scale24x512_ccnews_attnres_seed43_v8" ;;
    44) EXPERIMENT_DIR="$ROOT/results/scale24x512_ccnews_attnres_seed44_v8" ;;
    *) echo "unsupported seed ${seed}" >&2; exit 1 ;;
  esac
  for split in $FINAL_SPLITS; do
    CHECKPOINT="$EXPERIMENT_DIR/checkpoint_step_$(printf '%06d' "$step").pt" \
    CORPUS_TAG="v8_ccnews_p256d64_lockbox" \
    FINAL_SPLIT="$split" \
    BANK_TAG="v8_ccnews_seed${seed}_locked_${split}_step${step}_bank" \
    TRAIN_TAG="v8_ccnews_seed${seed}_locked_train" \
    VAL_TAG="v8_ccnews_seed${seed}_locked_val" \
    FINAL_TAG="v8_ccnews_seed${seed}_locked_${split}" \
    TRAIN_MANIFEST="$ROOT/results/lockbox_manifests_v8/v8_ccnews_p256d64_lockbox_train.jsonl" \
    VAL_MANIFEST="$ROOT/results/lockbox_manifests_v8/v8_ccnews_p256d64_lockbox_validation.jsonl" \
    FINAL_MANIFEST="$ROOT/results/lockbox_manifests_v8/v8_ccnews_p256d64_lockbox_${split}.jsonl" \
    BANK_SIZE="$bank_size" \
    SKIP_COUNT=1 \
    FEATURE_MODE="$feature_mode" \
    SELECTOR_MODEL="$model_name" \
    OUTPUT_PREFIX="v8_locked_seed${seed}_${split}_step${step}_b${bank_size}_${model_name}_${feature_mode}" \
    TEMPLATE_LIMIT="${TEMPLATE_LIMIT:-0}" \
    DEPLOY_NUM_SEQUENCES="$DEPLOY_NUM_SEQUENCES" \
    DEPLOY_BATCH_SIZE="$DEPLOY_BATCH_SIZE" \
    DEPLOY_TIMING_REPEATS="$DEPLOY_TIMING_REPEATS" \
    ORACLE_BATCH_SIZE="${ORACLE_BATCH_SIZE:-64}" \
    bash "$ROOT/scripts/run_locked_final_eval_v8.sh"
  done
done < <(
  "$ROOT/.venv/bin/python" - <<'PY' "$SELECTION_CSV"
import csv
import sys

path = sys.argv[1]
with open(path, newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        print(
            ",".join(
                [
                    str(row["seed"]),
                    str(row["step"]),
                    str(row["bank_size"]),
                    str(row["feature_mode"]),
                    str(row["model_name"]),
                ]
            )
        )
PY
)

echo "[ccnews-locked-multisplit-v8] complete"
