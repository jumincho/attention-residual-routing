#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
WINNERS_CSV="${WINNERS_CSV:?set WINNERS_CSV=/abs/path/to/winners.csv}"
SEEDS_RAW="${SEEDS:-43 44}"
FINAL_SPLITS="${FINAL_SPLITS:-final_A final_B final_C}"
PROMPT_LEN="${PROMPT_LEN:-256}"
DECODE_LEN="${DECODE_LEN:-64}"
ORACLE_BATCH_SIZE="${ORACLE_BATCH_SIZE:-64}"
HIDDEN_BATCH_SIZE="${HIDDEN_BATCH_SIZE:-16}"
BANK_SIZES_OVERRIDE="${BANK_SIZES_OVERRIDE:-}"
TRAIN_TOTAL_SHARDS="${TRAIN_TOTAL_SHARDS:-16}"
VAL_TOTAL_SHARDS="${VAL_TOTAL_SHARDS:-8}"
FINAL_TOTAL_SHARDS="${FINAL_TOTAL_SHARDS:-8}"

read -r -a SEEDS <<<"$SEEDS_RAW"

best_standard_step() {
  local metrics_path="$1"
  "$PY" - <<PY
import pandas as pd
df = pd.read_csv("$metrics_path")
keep = df[df["step"].isin([2500,3000,3500,4000,4500,5000,5500,6000])].copy()
keep = keep[pd.notna(keep["val_loss"])]
best = int(keep.sort_values("val_loss").iloc[0]["step"])
print(best)
PY
}

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

run_oracle_group() {
  local checkpoint_path="$1"
  local manifest_path="$2"
  local prefix="$3"
  local total_shards="$4"
  if [[ -f "$ROOT/results/oracles/${prefix}_oracle_mask_alignment_summary.csv" ]]; then
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
  if [[ -f "$ROOT/results/rich_features/${prefix}_hidden_prompt_features.csv" ]]; then
    return
  fi
  "$PY" "$ROOT/scripts/run_sharded_hidden_extract.py" \
    --checkpoint "$checkpoint_path" \
    --manifest-path "$manifest_path" \
    --prompt-len "$PROMPT_LEN" \
    --decode-len "$DECODE_LEN" \
    --tag-prefix "$prefix" \
    --total-shards "$total_shards" \
    --gpus 0 1 2 3
  mapfile -t shard_tags < <(seq -f "${prefix}_s%02g" 0 $((total_shards - 1)))
  "$PY" "$ROOT/scripts/merge_hidden_shards.py" --tags "${shard_tags[@]}" --output-tag "$prefix"
}

for seed in "${SEEDS[@]}"; do
  attn_step="$(winner_field "$seed" step)"
  bank_size="$(winner_field "$seed" bank_size)"
  if [[ -n "$BANK_SIZES_OVERRIDE" ]]; then
    bank_size="$BANK_SIZES_OVERRIDE"
  fi
  case "$seed" in
    43)
      ATTN_DIR="$ROOT/results/scale24x512_ccnews_attnres_seed43_v8"
      STD_DIR="$ROOT/results/scale24x512_ccnews_standard_seed43_v8"
      ;;
    44)
      ATTN_DIR="$ROOT/results/scale24x512_ccnews_attnres_seed44_v8"
      STD_DIR="$ROOT/results/scale24x512_ccnews_standard_seed44_v8"
      ;;
    *)
      echo "unsupported necessity seed ${seed}" >&2
      exit 1
      ;;
  esac
  attn_ckpt="$ATTN_DIR/checkpoint_step_$(printf '%06d' "$attn_step").pt"
  std_step="$(best_standard_step "$STD_DIR/metrics.csv")"
  std_ckpt="$STD_DIR/checkpoint_step_$(printf '%06d' "$std_step").pt"

  run_hidden_group "$attn_ckpt" "$ROOT/results/lockbox_manifests_v8/v8_ccnews_p256d64_lockbox_train.jsonl" "v8_ccnews_seed${seed}_locked_train" "$TRAIN_TOTAL_SHARDS"
  for split in $FINAL_SPLITS; do
    attn_final_tag="v8_ccnews_seed${seed}_locked_${split}"
    attn_bank_tag="v8_ccnews_seed${seed}_locked_${split}_step${attn_step}_bank"
    run_hidden_group "$attn_ckpt" "$ROOT/results/lockbox_manifests_v8/v8_ccnews_p256d64_lockbox_${split}.jsonl" "$attn_final_tag" "$FINAL_TOTAL_SHARDS"
    attn_out_tag="v8_necessity_attnres_seed${seed}_${split}_step${attn_step}_b${bank_size}"
    if [[ ! -f "$ROOT/results/regret_reduction_v8/${attn_out_tag}_hidden_summary.csv" ]]; then
      "$PY" "$ROOT/scripts/train_candidate_conditioned_ranker_v7.py" \
        --bank-tag "$attn_bank_tag" \
        --bank-size "$bank_size" \
        --train-tags "v8_ccnews_seed${seed}_locked_train" \
        --eval-tags "$attn_final_tag" \
        --hidden-train-tags "v8_ccnews_seed${seed}_locked_train" \
        --hidden-eval-tags "$attn_final_tag" \
        --output-tag "$attn_out_tag" \
        --feature-mode hidden \
        --skip-counts 1 \
        --output-subdir regret_reduction_v8 \
        --plot-prefix regret_reduction_v8
    fi
  done

  run_oracle_group "$std_ckpt" "$ROOT/results/lockbox_manifests_v8/v8_ccnews_p256d64_lockbox_train.jsonl" "v8_ccnews_std_seed${seed}_train" "$TRAIN_TOTAL_SHARDS"
  run_oracle_group "$std_ckpt" "$ROOT/results/lockbox_manifests_v8/v8_ccnews_p256d64_lockbox_validation.jsonl" "v8_ccnews_std_seed${seed}_val" "$VAL_TOTAL_SHARDS"
  run_hidden_group "$std_ckpt" "$ROOT/results/lockbox_manifests_v8/v8_ccnews_p256d64_lockbox_train.jsonl" "v8_ccnews_std_seed${seed}_train" "$TRAIN_TOTAL_SHARDS"
  for split in $FINAL_SPLITS; do
    std_final_tag="v8_ccnews_std_seed${seed}_${split}"
    std_bank_tag="v8_ccnews_std_seed${seed}_${split}_step${std_step}_bank"
    run_oracle_group "$std_ckpt" "$ROOT/results/lockbox_manifests_v8/v8_ccnews_p256d64_lockbox_${split}.jsonl" "$std_final_tag" "$FINAL_TOTAL_SHARDS"
    if [[ ! -f "$ROOT/results/bank_hygiene/${std_bank_tag}_summary.csv" ]]; then
      "$PY" "$ROOT/scripts/evaluate_train_only_bank_hygiene.py" \
        --calib-tags "v8_ccnews_std_seed${seed}_val" \
        --eval-tags "$std_final_tag" \
        --output-tag "$std_bank_tag" \
        --bank-sizes "$bank_size"
    fi
    run_hidden_group "$std_ckpt" "$ROOT/results/lockbox_manifests_v8/v8_ccnews_p256d64_lockbox_${split}.jsonl" "$std_final_tag" "$FINAL_TOTAL_SHARDS"
    std_out_tag="v8_necessity_standard_seed${seed}_${split}_step${std_step}_b${bank_size}"
    if [[ ! -f "$ROOT/results/regret_reduction_v8/${std_out_tag}_hidden_summary.csv" ]]; then
      "$PY" "$ROOT/scripts/train_candidate_conditioned_ranker_v7.py" \
        --bank-tag "$std_bank_tag" \
        --bank-size "$bank_size" \
        --train-tags "v8_ccnews_std_seed${seed}_train" \
        --eval-tags "$std_final_tag" \
        --hidden-train-tags "v8_ccnews_std_seed${seed}_train" \
        --hidden-eval-tags "$std_final_tag" \
        --output-tag "$std_out_tag" \
        --feature-mode hidden \
        --skip-counts 1 \
        --output-subdir regret_reduction_v8 \
        --plot-prefix regret_reduction_v8
    fi
  done
done

echo "[ccnews-necessity-multiseed-v8] complete"
