#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
SEEDS_RAW="${SEEDS:-42 43 44}"
BANK_SIZES_RAW="${BANK_SIZES:-32 64}"
FEATURE_MODES_RAW="${FEATURE_MODES:-attnres stp_scalar attnres_stp_scalar attnres_stp_diff hidden hidden_stp_diff}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-$ROOT/results/lockbox_manifests_v9/v9_ccnews_p256d64_lockbox_train2048.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-$ROOT/results/lockbox_manifests_v9/v9_ccnews_p256d64_lockbox_validation.jsonl}"
DEV_MANIFEST="${DEV_MANIFEST:-$ROOT/results/lockbox_manifests_v9/v9_ccnews_p256d64_lockbox_dev_select_v9_2048.jsonl}"
PROMPT_LEN="${PROMPT_LEN:-256}"
DECODE_LEN="${DECODE_LEN:-64}"
ORACLE_BATCH_SIZE="${ORACLE_BATCH_SIZE:-128}"
ORACLE_NUM_CHUNKS="${ORACLE_NUM_CHUNKS:-4}"
TRAIN_TOTAL_SHARDS="${TRAIN_TOTAL_SHARDS:-8}"
VAL_TOTAL_SHARDS="${VAL_TOTAL_SHARDS:-4}"
DEV_TOTAL_SHARDS="${DEV_TOTAL_SHARDS:-4}"
GPUS_RAW="${GPUS:-0 1 2 3}"
FAST_MODE="${FAST_MODE:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-regret_reduction_v9}"
PLOT_PREFIX="${PLOT_PREFIX:-regret_reduction_v9}"
HIDDEN_PCA_DIM="${HIDDEN_PCA_DIM:-64}"
STP_PCA_DIM="${STP_PCA_DIM:-64}"

read -r -a SEEDS <<<"$SEEDS_RAW"
read -r -a BANK_SIZES <<<"$BANK_SIZES_RAW"
read -r -a FEATURE_MODES <<<"$FEATURE_MODES_RAW"
read -r -a GPUS <<<"$GPUS_RAW"

checkpoint_dir_for_seed() {
  local seed="$1"
  case "$seed" in
    42) echo "$ROOT/results/scale24x512_ccnews_attnres_dense_v7" ;;
    43) echo "$ROOT/results/scale24x512_ccnews_attnres_seed43_v8" ;;
    44) echo "$ROOT/results/scale24x512_ccnews_attnres_seed44_v8" ;;
    *) echo "unsupported seed ${seed}" >&2; return 1 ;;
  esac
}

tag_prefix_for_seed() {
  local seed="$1"
  echo "v9_ccnews_seed${seed}_full"
}

shortlist_steps_for_seed() {
  local seed="$1"
  case "$seed" in
    # Preserve the narrow replicated V8 route-best vs LM-best neighborhoods.
    42) echo "${STEPS_SEED42:-3000 5500 6000}" ;;
    43) echo "${STEPS_SEED43:-3000 6000}" ;;
    44) echo "${STEPS_SEED44:-3000 3500}" ;;
    *) echo "unsupported seed ${seed}" >&2; return 1 ;;
  esac
}

run_oracle_group() {
  local checkpoint_path="$1"
  local manifest_path="$2"
  local prefix="$3"
  local total_shards="$4"
  local summary_path="$ROOT/results/oracles/${prefix}_oracle_mask_alignment_summary.csv"

  if [[ "$SKIP_EXISTING" == "1" && -f "$summary_path" ]]; then
    echo "[ccnews-dev-selection-v9] skip oracle ${prefix}"
    return
  fi

  "$PY" "$ROOT/scripts/run_sharded_oracle_eval.py" \
    --checkpoint "$checkpoint_path" \
    --manifest-path "$manifest_path" \
    --prompt-len "$PROMPT_LEN" \
    --decode-len "$DECODE_LEN" \
    --tag-prefix "$prefix" \
    --total-shards "$total_shards" \
    --gpus "${GPUS[@]}" \
    --batch-size "$ORACLE_BATCH_SIZE" \
    --num-chunks "$ORACLE_NUM_CHUNKS" \
    --score-mode utility_over_variance \
    --skip-counts 1 2 3
  mapfile -t shard_tags < <(seq -f "${prefix}_s%02g" 0 $((total_shards - 1)))
  "$PY" "$ROOT/scripts/merge_oracle_shards.py" --tags "${shard_tags[@]}" --output-tag "$prefix"
}

run_hidden_group() {
  local checkpoint_path="$1"
  local manifest_path="$2"
  local prefix="$3"
  local total_shards="$4"
  local hidden_path="$ROOT/results/rich_features/${prefix}_hidden_prompt_features.csv"

  if [[ "$SKIP_EXISTING" == "1" && -f "$hidden_path" ]]; then
    echo "[ccnews-dev-selection-v9] skip hidden ${prefix}"
    return
  fi

  "$PY" "$ROOT/scripts/run_sharded_hidden_extract.py" \
    --checkpoint "$checkpoint_path" \
    --manifest-path "$manifest_path" \
    --prompt-len "$PROMPT_LEN" \
    --decode-len "$DECODE_LEN" \
    --tag-prefix "$prefix" \
    --total-shards "$total_shards" \
    --gpus "${GPUS[@]}" \
    --python-bin "$PY"
  mapfile -t shard_tags < <(seq -f "${prefix}_s%02g" 0 $((total_shards - 1)))
  "$PY" "$ROOT/scripts/merge_hidden_shards.py" --tags "${shard_tags[@]}" --output-tag "$prefix"
}

for seed in "${SEEDS[@]}"; do
  checkpoint_dir="$(checkpoint_dir_for_seed "$seed")"
  tag_prefix="$(tag_prefix_for_seed "$seed")"
  read -r -a steps <<<"$(shortlist_steps_for_seed "$seed")"

  echo "[ccnews-dev-selection-v9] seed=${seed} steps=${steps[*]}"

  for step in "${steps[@]}"; do
    checkpoint_path="$checkpoint_dir/checkpoint_step_$(printf '%06d' "$step").pt"
    train_tag="${tag_prefix}_step${step}_train2048"
    val_tag="${tag_prefix}_step${step}_val"
    dev_tag="${tag_prefix}_step${step}_dev2048"
    bank_tag="${tag_prefix}_step${step}_bank"

    run_oracle_group "$checkpoint_path" "$TRAIN_MANIFEST" "$train_tag" "$TRAIN_TOTAL_SHARDS"
    run_oracle_group "$checkpoint_path" "$VAL_MANIFEST" "$val_tag" "$VAL_TOTAL_SHARDS"
    run_oracle_group "$checkpoint_path" "$DEV_MANIFEST" "$dev_tag" "$DEV_TOTAL_SHARDS"
    run_hidden_group "$checkpoint_path" "$TRAIN_MANIFEST" "$train_tag" "$TRAIN_TOTAL_SHARDS"
    run_hidden_group "$checkpoint_path" "$DEV_MANIFEST" "$dev_tag" "$DEV_TOTAL_SHARDS"

    if [[ "$SKIP_EXISTING" != "1" || ! -f "$ROOT/results/bank_hygiene/${bank_tag}_summary.csv" ]]; then
      "$PY" "$ROOT/scripts/evaluate_train_only_bank_hygiene.py" \
        --calib-tags "$val_tag" \
        --eval-tags "$dev_tag" \
        --output-tag "$bank_tag" \
        --bank-sizes "${BANK_SIZES[@]}"
    else
      echo "[ccnews-dev-selection-v9] skip bank hygiene ${bank_tag}"
    fi

    for bank_size in "${BANK_SIZES[@]}"; do
      output_tag="${tag_prefix}_step${step}_dev2048_b${bank_size}"
      BANK_TAG="$bank_tag" \
      BANK_SIZE="$bank_size" \
      TRAIN_TAGS="$train_tag" \
      EVAL_TAGS="$dev_tag" \
      HIDDEN_TRAIN_TAGS="$train_tag" \
      HIDDEN_EVAL_TAGS="$dev_tag" \
      OUTPUT_TAG="$output_tag" \
      FEATURE_MODES="${FEATURE_MODES[*]}" \
      FAST_MODE="$FAST_MODE" \
      OUTPUT_SUBDIR="$OUTPUT_SUBDIR" \
      PLOT_PREFIX="$PLOT_PREFIX" \
      HIDDEN_PCA_DIM="$HIDDEN_PCA_DIM" \
      STP_PCA_DIM="$STP_PCA_DIM" \
      SKIP_EXISTING="$SKIP_EXISTING" \
      bash "$ROOT/scripts/run_stp_feature_selector_v9.sh"
    done
  done
done

echo "[ccnews-dev-selection-v9] complete"
