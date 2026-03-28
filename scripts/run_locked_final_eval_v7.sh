#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"

CORPUS_TAG="${CORPUS_TAG:?set CORPUS_TAG=v7_ccnews_p256d64_lockbox}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT=/abs/path/to/checkpoint.pt}"
BANK_TAG="${BANK_TAG:?set BANK_TAG=${CORPUS_TAG}_step3000_bank}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:?set TRAIN_MANIFEST=/abs/path/to/train.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:?set VAL_MANIFEST=/abs/path/to/validation.jsonl}"
FINAL_MANIFEST="${FINAL_MANIFEST:?set FINAL_MANIFEST=/abs/path/to/final_test.jsonl}"
PROMPT_LEN="${PROMPT_LEN:-256}"
DECODE_LEN="${DECODE_LEN:-64}"
BANK_SIZE="${BANK_SIZE:-32}"
SKIP_COUNT="${SKIP_COUNT:-1}"
FEATURE_MODE="${FEATURE_MODE:-attnres}"
FAST_MODE="${FAST_MODE:-0}"
SELECTOR_MODEL="${SELECTOR_MODEL:-hgb_pair}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:?set OUTPUT_PREFIX=v7_locked_ccnews_step3000_b32}"
RUN_HIDDEN="${RUN_HIDDEN:-0}"
TEMPLATE_LIMIT="${TEMPLATE_LIMIT:-0}"
DEPLOY_NUM_SEQUENCES="${DEPLOY_NUM_SEQUENCES:-256}"
DEPLOY_BATCH_SIZE="${DEPLOY_BATCH_SIZE:-16}"
DEPLOY_TIMING_REPEATS="${DEPLOY_TIMING_REPEATS:-5}"
ORACLE_BATCH_SIZE="${ORACLE_BATCH_SIZE:-64}"
ORACLE_NUM_CHUNKS="${ORACLE_NUM_CHUNKS:-4}"
TRAIN_TOTAL_SHARDS="${TRAIN_TOTAL_SHARDS:-8}"
VAL_TOTAL_SHARDS="${VAL_TOTAL_SHARDS:-4}"
FINAL_TOTAL_SHARDS="${FINAL_TOTAL_SHARDS:-4}"
ORACLE_SKIP_COUNTS="${ORACLE_SKIP_COUNTS:-1 2 3}"

VAL_TAG="${VAL_TAG:-${CORPUS_TAG}_locked_val}"
FINAL_TAG="${FINAL_TAG:-${CORPUS_TAG}_locked_final}"
TRAIN_TAG="${TRAIN_TAG:-${CORPUS_TAG}_locked_train}"

run_oracle_group() {
  local ckpt="$1"
  local manifest_path="$2"
  local prefix="$3"
  local total_shards="$4"
  if [[ -f "results/oracles/${prefix}_oracle_mask_alignment_summary.csv" ]]; then
    echo "[lockbox-v7] skip oracle ${prefix}"
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
    --batch-size "$ORACLE_BATCH_SIZE" \
    --num-chunks "$ORACLE_NUM_CHUNKS" \
    --score-mode utility_over_variance \
    --skip-counts ${ORACLE_SKIP_COUNTS}
  mapfile -t shard_tags < <(seq -f "${prefix}_s%02g" 0 $((total_shards - 1)))
  "$PY" scripts/merge_oracle_shards.py --tags "${shard_tags[@]}" --output-tag "$prefix"
}

run_hidden_group() {
  local ckpt="$1"
  local manifest_path="$2"
  local prefix="$3"
  local total_shards="$4"
  if [[ -f "results/rich_features/${prefix}_hidden_prompt_features.csv" ]]; then
    echo "[lockbox-v7] skip hidden ${prefix}"
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

run_oracle_group "$CHECKPOINT" "$TRAIN_MANIFEST" "$TRAIN_TAG" "$TRAIN_TOTAL_SHARDS"
run_oracle_group "$CHECKPOINT" "$VAL_MANIFEST" "$VAL_TAG" "$VAL_TOTAL_SHARDS"
run_oracle_group "$CHECKPOINT" "$FINAL_MANIFEST" "$FINAL_TAG" "$FINAL_TOTAL_SHARDS"

if [[ ! -f "results/bank_hygiene/${BANK_TAG}_summary.csv" ]]; then
  "$PY" scripts/evaluate_train_only_bank_hygiene.py \
    --calib-tags "$VAL_TAG" \
    --eval-tags "$FINAL_TAG" \
    --output-tag "$BANK_TAG" \
    --bank-sizes "$BANK_SIZE"
fi

if [[ "$RUN_HIDDEN" == "1" ]]; then
  run_hidden_group "$CHECKPOINT" "$TRAIN_MANIFEST" "$TRAIN_TAG" 8
  run_hidden_group "$CHECKPOINT" "$FINAL_MANIFEST" "$FINAL_TAG" "$FINAL_TOTAL_SHARDS"
fi

cmd=(
  "$PY" scripts/train_candidate_conditioned_ranker_v7.py
  --bank-tag "$BANK_TAG"
  --bank-size "$BANK_SIZE"
  --train-tags "$TRAIN_TAG"
  --eval-tags "$FINAL_TAG"
  --output-tag "$OUTPUT_PREFIX"
  --feature-mode "$FEATURE_MODE"
)
if [[ "$FAST_MODE" == "1" ]]; then
  cmd+=(--fast-mode)
fi
if [[ "$RUN_HIDDEN" == "1" ]]; then
  cmd+=(--hidden-train-tags "$TRAIN_TAG" --hidden-eval-tags "$FINAL_TAG")
fi
"${cmd[@]}"

if [[ -n "${SELECTOR_MODEL// }" ]]; then
  "$PY" scripts/evaluate_deployment_measurement_v7.py \
    --checkpoint "$CHECKPOINT" \
    --bank-tag "$BANK_TAG" \
    --bank-size "$BANK_SIZE" \
    --skip-count "$SKIP_COUNT" \
    --train-tags "$TRAIN_TAG" \
    --eval-tags "$FINAL_TAG" \
    --manifest-path "$FINAL_MANIFEST" \
    --feature-mode "$FEATURE_MODE" \
    --selected-model "$SELECTOR_MODEL" \
    --output-tag "${OUTPUT_PREFIX}_deploy" \
    --template-limit "$TEMPLATE_LIMIT" \
    --num-sequences "$DEPLOY_NUM_SEQUENCES" \
    --batch-size "$DEPLOY_BATCH_SIZE" \
    --timing-repeats "$DEPLOY_TIMING_REPEATS" \
    --precision fp16 \
    --device cuda
fi

echo "[lockbox-v7] complete"
