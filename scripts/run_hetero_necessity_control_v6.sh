#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:?set EXPERIMENT_DIR=$ROOT/results/scale24x512_ccnews_standard_v6}"
BEST_CKPT_PATH="${BEST_CKPT_PATH:-$EXPERIMENT_DIR/best_checkpoint.pt}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-$ROOT/results/selector_data_scale/v6_ccnews_p256d64_stride160_train4096.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-$ROOT/results/selector_data_scale/v6_ccnews_p256d64_stride160_validation.jsonl}"
TEST_MANIFEST="${TEST_MANIFEST:-$ROOT/results/selector_data_scale/v6_ccnews_p256d64_stride160_test.jsonl}"
PROMPT_LEN="${PROMPT_LEN:-256}"
DECODE_LEN="${DECODE_LEN:-64}"
BATCH_SIZE="${BATCH_SIZE:-64}"
BANK_SIZE="${BANK_SIZE:-32}"
TRAIN_SHARDS="${TRAIN_SHARDS:-8}"
VAL_SHARDS="${VAL_SHARDS:-4}"
TEST_SHARDS="${TEST_SHARDS:-4}"

TRAIN_TAG="${TRAIN_TAG:-v6_stdccnews_train4096}"
VAL_TAG="${VAL_TAG:-v6_stdccnews_val1024}"
TEST_TAG="${TEST_TAG:-v6_stdccnews_test1024}"
BANK_TAG="${BANK_TAG:-v6_stdccnews_bank}"
RANKER_TAG="${RANKER_TAG:-v6_stdccnews_test1024_b32}"

echo "[necessity-v6] checkpoint=${BEST_CKPT_PATH}"
echo "[necessity-v6] estimated train-oracle time: ~20-40m on 4 GPUs"
echo "[necessity-v6] estimated val/test oracles + bank + hidden ranker: ~25-45m on 4 GPUs"

run_oracle_group() {
  local ckpt="$1"
  local manifest_path="$2"
  local prefix="$3"
  local total_shards="$4"
  if [[ -f "results/oracles/${prefix}_oracle_mask_alignment_summary.csv" ]]; then
    echo "[necessity-v6] skip oracle ${prefix}"
    return
  fi
  "$PY" scripts/run_sharded_oracle_eval.py \
    --checkpoint "$ckpt" \
    --manifest-path "$manifest_path" \
    --prompt-len "$PROMPT_LEN" \
    --decode-len "$DECODE_LEN" \
    --tag-prefix "$prefix" \
    --total-shards "$total_shards" \
    --gpus 0 1 2 3 \
    --batch-size "$BATCH_SIZE" \
    --num-chunks 4 \
    --score-mode utility_over_variance \
    --skip-counts 1 2 3
  mapfile -t shard_tags < <(seq -f "${prefix}_s%02g" 0 $((total_shards - 1)))
  "$PY" scripts/merge_oracle_shards.py --tags "${shard_tags[@]}" --output-tag "$prefix"
}

run_hidden_group() {
  local ckpt="$1"
  local manifest_path="$2"
  local prefix="$3"
  local total_shards="$4"
  if [[ -f "results/rich_features/${prefix}_hidden_prompt_features.csv" ]]; then
    echo "[necessity-v6] skip hidden ${prefix}"
    return
  fi
  "$PY" scripts/run_sharded_hidden_extract.py \
    --checkpoint "$ckpt" \
    --manifest-path "$manifest_path" \
    --prompt-len "$PROMPT_LEN" \
    --decode-len "$DECODE_LEN" \
    --tag-prefix "$prefix" \
    --total-shards "$total_shards" \
    --gpus 0 1 2 3
  mapfile -t shard_tags < <(seq -f "${prefix}_s%02g" 0 $((total_shards - 1)))
  "$PY" scripts/merge_hidden_shards.py --tags "${shard_tags[@]}" --output-tag "$prefix"
}

run_oracle_group "$BEST_CKPT_PATH" "$VAL_MANIFEST" "$VAL_TAG" "$VAL_SHARDS"
run_oracle_group "$BEST_CKPT_PATH" "$TEST_MANIFEST" "$TEST_TAG" "$TEST_SHARDS"
run_oracle_group "$BEST_CKPT_PATH" "$TRAIN_MANIFEST" "$TRAIN_TAG" "$TRAIN_SHARDS"

if [[ ! -f "results/bank_hygiene/${BANK_TAG}_summary.csv" ]]; then
  "$PY" scripts/evaluate_train_only_bank_hygiene.py \
    --calib-tags "$VAL_TAG" \
    --eval-tags "$TEST_TAG" \
    --output-tag "$BANK_TAG" \
    --bank-sizes 16 32 64
fi

run_hidden_group "$BEST_CKPT_PATH" "$TRAIN_MANIFEST" "$TRAIN_TAG" "$TRAIN_SHARDS"
run_hidden_group "$BEST_CKPT_PATH" "$TEST_MANIFEST" "$TEST_TAG" "$TEST_SHARDS"

if [[ ! -f "results/ranker_v5/${RANKER_TAG}_hidden_summary.csv" ]]; then
  "$PY" scripts/train_candidate_conditioned_ranker_v5.py \
    --bank-tag "$BANK_TAG" \
    --bank-size "$BANK_SIZE" \
    --train-tags "$TRAIN_TAG" \
    --eval-tags "$TEST_TAG" \
    --hidden-train-tags "$TRAIN_TAG" \
    --hidden-eval-tags "$TEST_TAG" \
    --output-tag "$RANKER_TAG" \
    --feature-mode hidden
fi

echo "[necessity-v6] complete"
