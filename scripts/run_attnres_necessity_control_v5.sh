#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
LOG_DIR="$ROOT/results/attnres_necessity_control_v5/logs"
mkdir -p "$LOG_DIR"
MARKER_FILE="$ROOT/results/attnres_necessity_control_v5/complete.txt"

STD_CKPT="${STD_CKPT:-$ROOT/results/benchmark24x384_wikitext_standard_ddp_bs24/best_checkpoint.pt}"
PROMPT_LEN="${PROMPT_LEN:-256}"
DECODE_LEN="${DECODE_LEN:-64}"
BATCH_SIZE="${BATCH_SIZE:-128}"

TRAIN_FULL="$ROOT/results/selector_data_scale/v5_best200_p256d64_stride160_train.jsonl"
VAL_FULL="$ROOT/results/selector_data_scale/v5_best200_p256d64_stride160_validation.jsonl"
TEST_FULL="$ROOT/results/selector_data_scale/v5_best200_p256d64_stride160_test.jsonl"

TRAIN_MANIFEST="$ROOT/results/selector_data_scale/v5_best200_p256d64_stride160_train4k.jsonl"
VAL_MANIFEST="$ROOT/results/selector_data_scale/v5_best200_p256d64_stride160_val1k.jsonl"
TEST_MANIFEST="$ROOT/results/selector_data_scale/v5_best200_p256d64_stride160_test1k.jsonl"

TRAIN_TAG="v5stdctrl_current_p256d64_train4k"
VAL_TAG="v5stdctrl_current_p256d64_val1k"
TEST_TAG="v5stdctrl_current_p256d64_test1k"
BANK_TAG="v5stdctrl_current_p256d64_bank_test"
RANKER_TAG="v5stdctrl_current_p256d64_test_b32"

echo "[necessity-v5] checkpoint=$STD_CKPT"
echo "[necessity-v5] estimated oracle time: train4k ~80-90m, val/test ~20-25m each on 4 GPUs"
echo "[necessity-v5] estimated hidden extract: train4k ~10m, val/test ~3-5m each on 4 GPUs"

if [[ ! -f "$TRAIN_MANIFEST" ]]; then
  "$PY" scripts/subset_manifest.py --input "$TRAIN_FULL" --output "$TRAIN_MANIFEST" --target-count 4096 --mode head
fi
if [[ ! -f "$VAL_MANIFEST" ]]; then
  "$PY" scripts/subset_manifest.py --input "$VAL_FULL" --output "$VAL_MANIFEST" --target-count 1024 --mode head
fi
if [[ ! -f "$TEST_MANIFEST" ]]; then
  "$PY" scripts/subset_manifest.py --input "$TEST_FULL" --output "$TEST_MANIFEST" --target-count 1024 --mode head
fi

run_oracle_group() {
  local manifest_path="$1"
  local prefix="$2"
  local total_shards="$3"
  local logfile="$LOG_DIR/${prefix}_oracle.log"
  if [[ -f "results/oracles/${prefix}_oracle_mask_alignment_summary.csv" ]]; then
    echo "[necessity-v5] skip oracle $prefix (already exists)"
    return
  fi
  echo "[necessity-v5] oracle $prefix -> $logfile"
  "$PY" scripts/run_sharded_oracle_eval.py \
    --checkpoint "$STD_CKPT" \
    --manifest-path "$manifest_path" \
    --prompt-len "$PROMPT_LEN" \
    --decode-len "$DECODE_LEN" \
    --tag-prefix "$prefix" \
    --total-shards "$total_shards" \
    --gpus 0 1 2 3 \
    --batch-size "$BATCH_SIZE" \
    --num-chunks 4 \
    --score-mode utility_over_variance \
    --skip-counts 1 2 3 \
    >"$logfile" 2>&1
  mapfile -t shard_tags < <(seq -f "${prefix}_s%02g" 0 $((total_shards - 1)))
  "$PY" scripts/merge_oracle_shards.py --tags "${shard_tags[@]}" --output-tag "$prefix"
}

run_hidden_group() {
  local manifest_path="$1"
  local prefix="$2"
  local total_shards="$3"
  local logfile="$LOG_DIR/${prefix}_hidden.log"
  if [[ -f "results/rich_features/${prefix}_hidden_prompt_features.csv" ]]; then
    echo "[necessity-v5] skip hidden $prefix (already exists)"
    return
  fi
  echo "[necessity-v5] hidden $prefix -> $logfile"
  "$PY" scripts/run_sharded_hidden_extract.py \
    --checkpoint "$STD_CKPT" \
    --manifest-path "$manifest_path" \
    --prompt-len "$PROMPT_LEN" \
    --decode-len "$DECODE_LEN" \
    --tag-prefix "$prefix" \
    --total-shards "$total_shards" \
    --gpus 0 1 2 3 \
    >"$logfile" 2>&1
  mapfile -t shard_tags < <(seq -f "${prefix}_s%02g" 0 $((total_shards - 1)))
  "$PY" scripts/merge_hidden_shards.py --tags "${shard_tags[@]}" --output-tag "$prefix"
}

run_oracle_group "$TRAIN_MANIFEST" "$TRAIN_TAG" 8
run_oracle_group "$VAL_MANIFEST" "$VAL_TAG" 4
run_oracle_group "$TEST_MANIFEST" "$TEST_TAG" 4

run_hidden_group "$TRAIN_MANIFEST" "$TRAIN_TAG" 8
run_hidden_group "$VAL_MANIFEST" "$VAL_TAG" 4
run_hidden_group "$TEST_MANIFEST" "$TEST_TAG" 4

if [[ ! -f "results/bank_hygiene/${BANK_TAG}_summary.csv" ]]; then
  "$PY" scripts/evaluate_train_only_bank_hygiene.py \
    --calib-tags "$VAL_TAG" \
    --eval-tags "$TEST_TAG" \
    --output-tag "$BANK_TAG" \
    --bank-sizes 16 32 64
fi

if [[ ! -f "results/ranker_v5/${RANKER_TAG}_hidden_summary.csv" ]]; then
  "$PY" scripts/train_candidate_conditioned_ranker_v5.py \
    --bank-tag "$BANK_TAG" \
    --bank-size 32 \
    --train-tags "$TRAIN_TAG" \
    --eval-tags "$TEST_TAG" \
    --hidden-train-tags "$TRAIN_TAG" \
    --hidden-eval-tags "$TEST_TAG" \
    --output-tag "$RANKER_TAG" \
    --feature-mode hidden
fi

printf 'checkpoint=%s\n' "$STD_CKPT" > "$MARKER_FILE"
echo "[necessity-v5] complete"
