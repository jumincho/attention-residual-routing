#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT=/abs/path/to/checkpoint.pt}"
BANK_TAG="${BANK_TAG:?set BANK_TAG=v7_ccnews_p256d64_lockbox_step5000_bank}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:?set TRAIN_MANIFEST=/abs/path/to/train.jsonl}"
EVAL_MANIFEST="${EVAL_MANIFEST:?set EVAL_MANIFEST=/abs/path/to/dev_test.jsonl}"
PROMPT_LEN="${PROMPT_LEN:-256}"
DECODE_LEN="${DECODE_LEN:-64}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TRAIN_SHARDS="${TRAIN_SHARDS:-8}"
EVAL_SHARDS="${EVAL_SHARDS:-4}"
GPUS="${GPUS:-0 1 2 3}"
BANK_SIZE="${BANK_SIZE:-32}"
RUN_HIDDEN="${RUN_HIDDEN:-0}"
FAST_MODE="${FAST_MODE:-1}"
TRAIN_TAG="${TRAIN_TAG:?set TRAIN_TAG=v7_ccnews_step5000_train8192}"
EVAL_TAG="${EVAL_TAG:?set EVAL_TAG=v7_ccnews_step5000_dev2048}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:?set OUTPUT_PREFIX=v7_ccnews_step5000_dev2048_b32}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-regret_reduction_v7}"
PLOT_PREFIX="${PLOT_PREFIX:-regret_reduction_v7}"

echo "[selector-v7] checkpoint=${CHECKPOINT}"
echo "[selector-v7] bank_tag=${BANK_TAG} train_tag=${TRAIN_TAG} eval_tag=${EVAL_TAG} bank_size=${BANK_SIZE}"
echo "[selector-v7] estimated train-oracle time: ~35-80m for 8192 sequences on 4 GPUs"
echo "[selector-v7] estimated ranker time: ~5-20m on CPU depending on feature mode"

run_oracle_group() {
  local ckpt="$1"
  local manifest_path="$2"
  local prefix="$3"
  local total_shards="$4"
  if [[ -f "results/oracles/${prefix}_oracle_mask_alignment_summary.csv" ]]; then
    echo "[selector-v7] skip oracle ${prefix}"
    return
  fi
  "$PY" scripts/run_sharded_oracle_eval.py \
    --checkpoint "$ckpt" \
    --manifest-path "$manifest_path" \
    --prompt-len "$PROMPT_LEN" \
    --decode-len "$DECODE_LEN" \
    --tag-prefix "$prefix" \
    --total-shards "$total_shards" \
    --gpus ${GPUS} \
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
    echo "[selector-v7] skip hidden ${prefix}"
    return
  fi
  "$PY" scripts/run_sharded_hidden_extract.py \
    --checkpoint "$ckpt" \
    --manifest-path "$manifest_path" \
    --prompt-len "$PROMPT_LEN" \
    --decode-len "$DECODE_LEN" \
    --tag-prefix "$prefix" \
    --total-shards "$total_shards" \
    --gpus ${GPUS}
  mapfile -t shard_tags < <(seq -f "${prefix}_s%02g" 0 $((total_shards - 1)))
  "$PY" scripts/merge_hidden_shards.py --tags "${shard_tags[@]}" --output-tag "$prefix"
}

run_oracle_group "$CHECKPOINT" "$TRAIN_MANIFEST" "$TRAIN_TAG" "$TRAIN_SHARDS"
run_oracle_group "$CHECKPOINT" "$EVAL_MANIFEST" "$EVAL_TAG" "$EVAL_SHARDS"

if [[ "$RUN_HIDDEN" == "1" ]]; then
  run_hidden_group "$CHECKPOINT" "$TRAIN_MANIFEST" "$TRAIN_TAG" "$TRAIN_SHARDS"
  run_hidden_group "$CHECKPOINT" "$EVAL_MANIFEST" "$EVAL_TAG" "$EVAL_SHARDS"
fi

"$PY" scripts/train_candidate_conditioned_ranker_v7.py \
  --bank-tag "$BANK_TAG" \
  --bank-size "$BANK_SIZE" \
  --train-tags "$TRAIN_TAG" \
  --eval-tags "$EVAL_TAG" \
  --output-tag "$OUTPUT_PREFIX" \
  --feature-mode attnres \
  --output-subdir "$OUTPUT_SUBDIR" \
  --plot-prefix "$PLOT_PREFIX" \
  $( [[ "$FAST_MODE" == "1" ]] && printf '%s' '--fast-mode' )

if [[ "$RUN_HIDDEN" == "1" ]]; then
  "$PY" scripts/train_candidate_conditioned_ranker_v7.py \
    --bank-tag "$BANK_TAG" \
    --bank-size "$BANK_SIZE" \
    --train-tags "$TRAIN_TAG" \
    --eval-tags "$EVAL_TAG" \
    --hidden-train-tags "$TRAIN_TAG" \
    --hidden-eval-tags "$EVAL_TAG" \
    --output-tag "$OUTPUT_PREFIX" \
    --feature-mode full \
    --output-subdir "$OUTPUT_SUBDIR" \
    --plot-prefix "$PLOT_PREFIX" \
    $( [[ "$FAST_MODE" == "1" ]] && printf '%s' '--fast-mode' )
fi

echo "[selector-v7] complete"
