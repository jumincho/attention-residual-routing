#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
WINNERS_CSV="${WINNERS_CSV:-$ROOT/results/ccnews_multiseed_multisplit_v8/v8_ccnews_dev_frozen_selection_winners.csv}"
SEEDS_RAW="${SEEDS:-43 44}"
FINAL_SPLITS="${FINAL_SPLITS:-final_A final_B final_C}"
PROMPT_LEN="${PROMPT_LEN:-256}"
DECODE_LEN="${DECODE_LEN:-64}"
ORACLE_BATCH_SIZE="${ORACLE_BATCH_SIZE:-16}"
TRAIN_TOTAL_SHARDS="${TRAIN_TOTAL_SHARDS:-32}"
VAL_TOTAL_SHARDS="${VAL_TOTAL_SHARDS:-16}"
FINAL_TOTAL_SHARDS="${FINAL_TOTAL_SHARDS:-16}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-regret_reduction_v9}"
PLOT_PREFIX="${PLOT_PREFIX:-regret_reduction_v9}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

read -r -a SEEDS <<<"$SEEDS_RAW"

winner_field() {
  local seed="$1"
  local column="$2"
  "$PY" - <<PY
import pandas as pd
df = pd.read_csv("$WINNERS_CSV")
row = df[df["seed"] == $seed].iloc[0]
print(row["$column"])
PY
}

standard_checkpoint() {
  local seed="$1"
  local step="$2"
  local base
  case "$seed" in
    43) base="$ROOT/results/scale24x512_ccnews_standard_seed43_v8" ;;
    44) base="$ROOT/results/scale24x512_ccnews_standard_seed44_v8" ;;
    *)
      echo "unsupported matched standard seed ${seed}" >&2
      exit 1
      ;;
  esac
  printf "%s/checkpoint_step_%06d.pt\n" "$base" "$step"
}

run_oracle_group() {
  local checkpoint_path="$1"
  local manifest_path="$2"
  local prefix="$3"
  local total_shards="$4"
  if [[ "$SKIP_EXISTING" == "1" && -f "$ROOT/results/oracles/${prefix}_oracle_mask_alignment_summary.csv" ]]; then
    echo "[matched-necessity-v9] skip oracle ${prefix}"
    return
  fi
  "$PY" "$ROOT/scripts/run_sharded_oracle_eval.py" \
    --checkpoint "$checkpoint_path" \
    --manifest-path "$manifest_path" \
    --prompt-len "$PROMPT_LEN" \
    --decode-len "$DECODE_LEN" \
    --tag-prefix "$prefix" \
    --total-shards "$total_shards" \
    --gpus 0 1 2 3 \
    --batch-size "$ORACLE_BATCH_SIZE" \
    --num-chunks 4 \
    --score-mode utility_over_variance \
    --skip-counts 1
  mapfile -t shard_tags < <(seq -f "${prefix}_s%02g" 0 $((total_shards - 1)))
  "$PY" "$ROOT/scripts/merge_oracle_shards.py" --tags "${shard_tags[@]}" --output-tag "$prefix"
}

run_hidden_group() {
  local checkpoint_path="$1"
  local manifest_path="$2"
  local prefix="$3"
  local total_shards="$4"
  if [[ "$SKIP_EXISTING" == "1" && -f "$ROOT/results/rich_features/${prefix}_hidden_prompt_features.csv" ]]; then
    echo "[matched-necessity-v9] skip hidden ${prefix}"
    return
  fi
  "$PY" "$ROOT/scripts/run_sharded_hidden_extract.py" \
    --checkpoint "$checkpoint_path" \
    --manifest-path "$manifest_path" \
    --prompt-len "$PROMPT_LEN" \
    --decode-len "$DECODE_LEN" \
    --tag-prefix "$prefix" \
    --total-shards "$total_shards" \
    --gpus 0 1 2 3 \
    --python-bin "$PY"
  mapfile -t shard_tags < <(seq -f "${prefix}_s%02g" 0 $((total_shards - 1)))
  "$PY" "$ROOT/scripts/merge_hidden_shards.py" --tags "${shard_tags[@]}" --output-tag "$prefix"
}

for seed in "${SEEDS[@]}"; do
  step="$(winner_field "$seed" step)"
  bank_size="$(winner_field "$seed" bank_size)"
  std_ckpt="$(standard_checkpoint "$seed" "$step")"

  train_tag="v9_ccnews_stdmatched_seed${seed}_train"
  val_tag="v9_ccnews_stdmatched_seed${seed}_val"

  run_oracle_group "$std_ckpt" "$ROOT/results/lockbox_manifests_v8/v8_ccnews_p256d64_lockbox_train.jsonl" "$train_tag" "$TRAIN_TOTAL_SHARDS"
  run_oracle_group "$std_ckpt" "$ROOT/results/lockbox_manifests_v8/v8_ccnews_p256d64_lockbox_validation.jsonl" "$val_tag" "$VAL_TOTAL_SHARDS"
  run_hidden_group "$std_ckpt" "$ROOT/results/lockbox_manifests_v8/v8_ccnews_p256d64_lockbox_train.jsonl" "$train_tag" "$TRAIN_TOTAL_SHARDS"

  for split in $FINAL_SPLITS; do
    final_tag="v9_ccnews_stdmatched_seed${seed}_${split}"
    bank_tag="v9_ccnews_stdmatched_seed${seed}_${split}_step${step}_bank"
    out_tag="v9_necessity_standard_matched_seed${seed}_${split}_step${step}_b${bank_size}"

    run_oracle_group "$std_ckpt" "$ROOT/results/lockbox_manifests_v8/v8_ccnews_p256d64_lockbox_${split}.jsonl" "$final_tag" "$FINAL_TOTAL_SHARDS"
    run_hidden_group "$std_ckpt" "$ROOT/results/lockbox_manifests_v8/v8_ccnews_p256d64_lockbox_${split}.jsonl" "$final_tag" "$FINAL_TOTAL_SHARDS"

    if [[ "$SKIP_EXISTING" != "1" || ! -f "$ROOT/results/bank_hygiene/${bank_tag}_summary.csv" ]]; then
      "$PY" "$ROOT/scripts/evaluate_train_only_bank_hygiene.py" \
        --calib-tags "$val_tag" \
        --eval-tags "$final_tag" \
        --output-tag "$bank_tag" \
        --bank-sizes "$bank_size"
    fi

    if [[ "$SKIP_EXISTING" != "1" || ! -f "$ROOT/results/${OUTPUT_SUBDIR}/${out_tag}_hidden_summary.csv" ]]; then
      "$PY" "$ROOT/scripts/train_candidate_conditioned_ranker_v7.py" \
        --bank-tag "$bank_tag" \
        --bank-size "$bank_size" \
        --train-tags "$train_tag" \
        --eval-tags "$final_tag" \
        --hidden-train-tags "$train_tag" \
        --hidden-eval-tags "$final_tag" \
        --output-tag "$out_tag" \
        --feature-mode hidden \
        --skip-counts 1 \
        --output-subdir "$OUTPUT_SUBDIR" \
        --plot-prefix "$PLOT_PREFIX"
    fi
  done
done

echo "[matched-necessity-v9] complete"
