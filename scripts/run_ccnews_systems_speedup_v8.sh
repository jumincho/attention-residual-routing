#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

WINNERS_CSV="${WINNERS_CSV:?set WINNERS_CSV=/abs/path/to/winners.csv}"
FINAL_SPLITS="${FINAL_SPLITS:-final_A final_B final_C}"
TEMPLATE_LIMITS="${TEMPLATE_LIMITS:-0 2 4}"
NUM_SEQUENCES="${NUM_SEQUENCES:-256}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TIMING_REPEATS="${TIMING_REPEATS:-5}"

while IFS=, read -r seed step bank_size feature_mode model_name; do
  case "$seed" in
    42) EXPERIMENT_DIR="$ROOT/results/scale24x512_ccnews_attnres_dense_v7" ;;
    43) EXPERIMENT_DIR="$ROOT/results/scale24x512_ccnews_attnres_seed43_v8" ;;
    44) EXPERIMENT_DIR="$ROOT/results/scale24x512_ccnews_attnres_seed44_v8" ;;
    *) echo "unsupported seed ${seed}" >&2; exit 1 ;;
  esac
  for split in $FINAL_SPLITS; do
    for template_limit in $TEMPLATE_LIMITS; do
      CHECKPOINT="$EXPERIMENT_DIR/checkpoint_step_$(printf '%06d' "$step").pt" \
      BANK_TAG="v8_ccnews_seed${seed}_locked_${split}_step${step}_bank" \
      TRAIN_TAGS="v8_ccnews_seed${seed}_locked_train" \
      EVAL_TAGS="v8_ccnews_seed${seed}_locked_${split}" \
      MANIFEST_PATH="$ROOT/results/lockbox_manifests_v8/v8_ccnews_p256d64_lockbox_${split}.jsonl" \
      FEATURE_MODE="$feature_mode" \
      SELECTED_MODEL="$model_name" \
      SKIP_COUNT=1 \
      BANK_SIZE="$bank_size" \
      NUM_SEQUENCES="$NUM_SEQUENCES" \
      BATCH_SIZE="$BATCH_SIZE" \
      TIMING_REPEATS="$TIMING_REPEATS" \
      TEMPLATE_LIMIT="$template_limit" \
      OUTPUT_TAG="v8_systems_seed${seed}_${split}_step${step}_b${bank_size}_${model_name}_tpl${template_limit}" \
      bash "$ROOT/scripts/run_deployment_measurement_v7.sh"
    done
  done
done < <(
  "$ROOT/.venv/bin/python" - <<'PY' "$WINNERS_CSV"
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

echo "[ccnews-systems-speedup-v8] complete"
