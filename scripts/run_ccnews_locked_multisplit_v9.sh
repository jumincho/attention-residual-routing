#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
SELECTION_CSV="${SELECTION_CSV:-$ROOT/results/ccnews_multiseed_multisplit_v9/v9_ccnews_dev_frozen_selection.csv}"
SELECTION_HELPER="${SELECTION_HELPER:-$ROOT/scripts/select_ccnews_v9_frozen_configs.py}"
SELECTION_OUTPUT="${SELECTION_OUTPUT:-$ROOT/results/ccnews_multiseed_multisplit_v9/v9_ccnews_dev_frozen_selection.csv}"
SELECTION_CSV="${SELECTION_CSV:-$SELECTION_OUTPUT}"
WINNERS_CSV="${WINNERS_CSV:-${SELECTION_CSV%.*}_winners.csv}"
FREEZE_LEDGER="${FREEZE_LEDGER:-$ROOT/results/lockbox_manifests_v9/v9_ccnews_p256d64_lockbox_selection_freeze.csv}"
FINAL_SPLITS="${FINAL_SPLITS:-final_D final_E final_F}"
DEPLOY_BATCH_SIZE="${DEPLOY_BATCH_SIZE:-16}"
DEPLOY_NUM_SEQUENCES="${DEPLOY_NUM_SEQUENCES:-256}"
DEPLOY_TIMING_REPEATS="${DEPLOY_TIMING_REPEATS:-5}"
ORACLE_BATCH_SIZE="${ORACLE_BATCH_SIZE:-64}"
ORACLE_NUM_CHUNKS="${ORACLE_NUM_CHUNKS:-4}"
TRAIN_TOTAL_SHARDS="${TRAIN_TOTAL_SHARDS:-32}"
VAL_TOTAL_SHARDS="${VAL_TOTAL_SHARDS:-16}"
FINAL_TOTAL_SHARDS="${FINAL_TOTAL_SHARDS:-16}"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-regret_reduction_v9}"
PLOT_PREFIX="${PLOT_PREFIX:-regret_reduction_v9}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CORPUS_TAG="${CORPUS_TAG:-v9_ccnews_p256d64_lockbox}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-$ROOT/results/lockbox_manifests_v9/v9_ccnews_p256d64_lockbox_train.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-$ROOT/results/lockbox_manifests_v9/v9_ccnews_p256d64_lockbox_validation.jsonl}"
FINAL_MANIFEST_TEMPLATE="${FINAL_MANIFEST_TEMPLATE:-$ROOT/results/lockbox_manifests_v9/v9_ccnews_p256d64_lockbox_%s.jsonl}"

read -r -a SEEDS_ARR <<<"${SEEDS:-42 43 44}"
read -r -a STEPS_ARR <<<"${STEPS:-2500 3000 3500 4000 4500 5000 5500 6000}"
read -r -a BANK_SIZES_ARR <<<"${BANK_SIZES:-32 64}"
read -r -a FEATURE_MODES_ARR <<<"${FEATURE_MODES:-attnres stp_scalar attnres_stp_scalar attnres_stp_diff hidden hidden_stp_diff}"
read -r -a ALLOWED_MODELS_ARR <<<"${ALLOWED_MODELS:-rf_pair rf_pair_weighted hgb_pair hgb_pair_weighted hgb_delta_cls binary_gate_top1 ternary_gate_top2 knn_prompt_v2 knn_pair_v2 ensemble_hgb_knn retrieval_rerank_top2 retrieval_rerank_top4}"

ensure_selection() {
  if [[ -s "$WINNERS_CSV" ]]; then
    echo "[ccnews-locked-multisplit-v9] using existing winners csv: $WINNERS_CSV"
    return
  fi

  echo "[ccnews-locked-multisplit-v9] building winners csv from V9 dev-select tables"
  "$PY" "$SELECTION_HELPER" \
    --seeds "${SEEDS_ARR[@]}" \
    --steps "${STEPS_ARR[@]}" \
    --bank-sizes "${BANK_SIZES_ARR[@]}" \
    --feature-modes "${FEATURE_MODES_ARR[@]}" \
    --allowed-models "${ALLOWED_MODELS_ARR[@]}" \
    --output "$SELECTION_OUTPUT" \
    --freeze-ledger "$FREEZE_LEDGER"
}

run_oracle_group() {
  local checkpoint_path="$1"
  local manifest_path="$2"
  local prefix="$3"
  local total_shards="$4"
  local summary_path="$ROOT/results/oracles/${prefix}_oracle_mask_alignment_summary.csv"
  if [[ "$SKIP_EXISTING" == "1" && -f "$summary_path" ]]; then
    echo "[ccnews-locked-multisplit-v9] skip oracle ${prefix}"
    return
  fi
  "$PY" "$ROOT/scripts/run_sharded_oracle_eval.py" \
    --checkpoint "$checkpoint_path" \
    --manifest-path "$manifest_path" \
    --prompt-len 256 \
    --decode-len 64 \
    --tag-prefix "$prefix" \
    --total-shards "$total_shards" \
    --gpus 0 1 2 3 \
    --batch-size "$ORACLE_BATCH_SIZE" \
    --num-chunks "$ORACLE_NUM_CHUNKS" \
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
  local hidden_path="$ROOT/results/rich_features/${prefix}_hidden_prompt_features.csv"
  if [[ "$SKIP_EXISTING" == "1" && -f "$hidden_path" ]]; then
    echo "[ccnews-locked-multisplit-v9] skip hidden ${prefix}"
    return
  fi
  "$PY" "$ROOT/scripts/run_sharded_hidden_extract.py" \
    --checkpoint "$checkpoint_path" \
    --manifest-path "$manifest_path" \
    --prompt-len 256 \
    --decode-len 64 \
    --tag-prefix "$prefix" \
    --total-shards "$total_shards" \
    --gpus 0 1 2 3 \
    --python-bin "$PY"
  mapfile -t shard_tags < <(seq -f "${prefix}_s%02g" 0 $((total_shards - 1)))
  "$PY" "$ROOT/scripts/merge_hidden_shards.py" --tags "${shard_tags[@]}" --output-tag "$prefix"
}

run_split_bundle() {
  local seed="$1"
  local step="$2"
  local bank_size="$3"
  local feature_mode="$4"
  local model_name="$5"
  local split="$6"
  local checkpoint_dir
  case "$seed" in
    42) checkpoint_dir="$ROOT/results/scale24x512_ccnews_attnres_dense_v7" ;;
    43) checkpoint_dir="$ROOT/results/scale24x512_ccnews_attnres_seed43_v8" ;;
    44) checkpoint_dir="$ROOT/results/scale24x512_ccnews_attnres_seed44_v8" ;;
    *) echo "unsupported seed ${seed}" >&2; exit 1 ;;
  esac

  local checkpoint_path="$checkpoint_dir/checkpoint_step_$(printf '%06d' "$step").pt"
  local train_tag="v9_ccnews_seed${seed}_step${step}_b${bank_size}_locked_train"
  local val_tag="v9_ccnews_seed${seed}_step${step}_b${bank_size}_locked_val"
  local final_tag="v9_ccnews_seed${seed}_step${step}_b${bank_size}_locked_${split}"
  local bank_tag="v9_ccnews_seed${seed}_locked_${split}_step${step}_bank"
  local template_suffix=""
  if [[ "${TEMPLATE_LIMIT:-0}" != "0" ]]; then
    template_suffix="_tpl${TEMPLATE_LIMIT}"
  fi
  local output_prefix="v9_locked_seed${seed}_${split}_step${step}_b${bank_size}_${model_name}_${feature_mode}${template_suffix}"
  local train_manifest="$TRAIN_MANIFEST"
  local val_manifest="$VAL_MANIFEST"
  local final_manifest
  printf -v final_manifest "$FINAL_MANIFEST_TEMPLATE" "$split"

  run_oracle_group "$checkpoint_path" "$train_manifest" "$train_tag" "$TRAIN_TOTAL_SHARDS"
  run_oracle_group "$checkpoint_path" "$val_manifest" "$val_tag" "$VAL_TOTAL_SHARDS"
  run_hidden_group "$checkpoint_path" "$train_manifest" "$train_tag" "$TRAIN_TOTAL_SHARDS"
  run_hidden_group "$checkpoint_path" "$val_manifest" "$val_tag" "$VAL_TOTAL_SHARDS"
  run_oracle_group "$checkpoint_path" "$final_manifest" "$final_tag" "$FINAL_TOTAL_SHARDS"
  run_hidden_group "$checkpoint_path" "$final_manifest" "$final_tag" "$FINAL_TOTAL_SHARDS"

  if [[ "$SKIP_EXISTING" != "1" || ! -f "$ROOT/results/bank_hygiene/${bank_tag}_summary.csv" ]]; then
    "$PY" "$ROOT/scripts/evaluate_train_only_bank_hygiene.py" \
      --calib-tags "$val_tag" \
      --eval-tags "$final_tag" \
      --output-tag "$bank_tag" \
      --bank-sizes "$bank_size"
  fi

  "$PY" "$ROOT/scripts/train_candidate_conditioned_ranker_v7.py" \
    --bank-tag "$bank_tag" \
    --bank-size "$bank_size" \
    --train-tags "$train_tag" \
    --eval-tags "$final_tag" \
    --hidden-train-tags "$train_tag" \
    --hidden-eval-tags "$final_tag" \
    --output-tag "$output_prefix" \
    --feature-mode "$feature_mode" \
    --skip-counts 1 \
    --output-subdir "$RESULTS_SUBDIR" \
    --plot-prefix "$PLOT_PREFIX"

  "$PY" "$ROOT/scripts/evaluate_deployment_measurement_v7.py" \
    --checkpoint "$checkpoint_path" \
    --bank-tag "$bank_tag" \
    --bank-size "$bank_size" \
    --skip-count 1 \
    --train-tags "$train_tag" \
    --eval-tags "$final_tag" \
    --manifest-path "$final_manifest" \
    --feature-mode "$feature_mode" \
    --selected-model "$model_name" \
    --output-tag "${output_prefix}_deploy" \
    --template-limit "${TEMPLATE_LIMIT:-0}" \
    --num-sequences "$DEPLOY_NUM_SEQUENCES" \
    --batch-size "$DEPLOY_BATCH_SIZE" \
    --timing-repeats "$DEPLOY_TIMING_REPEATS" \
    --precision fp16 \
    --device cuda
}

echo "[ccnews-locked-multisplit-v9] start"
ensure_selection

if [[ ! -s "$WINNERS_CSV" ]]; then
  echo "[ccnews-locked-multisplit-v9] no frozen winners available at $WINNERS_CSV" >&2
  exit 1
fi

while IFS=, read -r seed step bank_size feature_mode model_name; do
  [[ "$seed" == "seed" || -z "$seed" ]] && continue
  for split in $FINAL_SPLITS; do
    run_split_bundle "$seed" "$step" "$bank_size" "$feature_mode" "$model_name" "$split"
  done
done < <(
  "$PY" - <<'PY' "$WINNERS_CSV"
import csv
import sys

path = sys.argv[1]
with open(path, newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        print(",".join([str(row["seed"]), str(row["step"]), str(row["bank_size"]), str(row["feature_mode"]), str(row["model_name"]) ]))
PY
)

echo "[ccnews-locked-multisplit-v9] complete"
