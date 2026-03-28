#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
ATTNRES_CKPT="${ATTNRES_CKPT:-$ROOT/results/scale24x512_wikitext103_attnres_v6/checkpoint_step_003000.pt}"
STD_CKPT="${STD_CKPT:-$ROOT/results/scale24x512_wikitext103_standard_v7/checkpoint_step_003000.pt}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-$ROOT/results/lockbox_manifests_v7/v7_wiki_p256d64_lockbox_train.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-$ROOT/results/lockbox_manifests_v7/v7_wiki_p256d64_lockbox_validation.jsonl}"
DEV_MANIFEST="${DEV_MANIFEST:-$ROOT/results/lockbox_manifests_v7/v7_wiki_p256d64_lockbox_dev_test.jsonl}"
PROMPT_LEN="${PROMPT_LEN:-256}"
DECODE_LEN="${DECODE_LEN:-64}"
BATCH_SIZE="${BATCH_SIZE:-64}"
BANK_SIZE="${BANK_SIZE:-32}"
SKIP_COUNT="${SKIP_COUNT:-2}"
TRAIN_SHARDS="${TRAIN_SHARDS:-8}"
VAL_SHARDS="${VAL_SHARDS:-4}"
DEV_SHARDS="${DEV_SHARDS:-4}"
TAG_PREFIX="${TAG_PREFIX:-v7_wiki_p256d64_lockbox_step3000}"

run_oracle_group() {
  local ckpt="$1"
  local manifest_path="$2"
  local prefix="$3"
  local total_shards="$4"
  if [[ -f "results/oracles/${prefix}_oracle_mask_alignment_summary.csv" ]]; then
    echo "[wikitext-necessity-v7] skip oracle ${prefix}"
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
    echo "[wikitext-necessity-v7] skip hidden ${prefix}"
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

ATTNRES_TRAIN_TAG="${TAG_PREFIX}_attnres_train"
ATTNRES_VAL_TAG="${TAG_PREFIX}_attnres_val"
ATTNRES_DEV_TAG="${TAG_PREFIX}_attnres_dev"
ATTNRES_BANK_TAG="${TAG_PREFIX}_attnres_bank"

STD_TRAIN_TAG="${TAG_PREFIX}_std_train"
STD_VAL_TAG="${TAG_PREFIX}_std_val"
STD_DEV_TAG="${TAG_PREFIX}_std_dev"
STD_BANK_TAG="${TAG_PREFIX}_std_bank"

run_oracle_group "$ATTNRES_CKPT" "$TRAIN_MANIFEST" "$ATTNRES_TRAIN_TAG" "$TRAIN_SHARDS"
run_oracle_group "$ATTNRES_CKPT" "$VAL_MANIFEST" "$ATTNRES_VAL_TAG" "$VAL_SHARDS"
run_oracle_group "$ATTNRES_CKPT" "$DEV_MANIFEST" "$ATTNRES_DEV_TAG" "$DEV_SHARDS"

if [[ ! -f "results/bank_hygiene/${ATTNRES_BANK_TAG}_summary.csv" ]]; then
  "$PY" scripts/evaluate_train_only_bank_hygiene.py \
    --calib-tags "$ATTNRES_VAL_TAG" \
    --eval-tags "$ATTNRES_DEV_TAG" \
    --output-tag "$ATTNRES_BANK_TAG" \
    --bank-sizes 32 64
fi

run_hidden_group "$ATTNRES_CKPT" "$TRAIN_MANIFEST" "$ATTNRES_TRAIN_TAG" "$TRAIN_SHARDS"
run_hidden_group "$ATTNRES_CKPT" "$DEV_MANIFEST" "$ATTNRES_DEV_TAG" "$DEV_SHARDS"

"$PY" scripts/train_candidate_conditioned_ranker_v7.py \
  --bank-tag "$ATTNRES_BANK_TAG" \
  --bank-size "$BANK_SIZE" \
  --train-tags "$ATTNRES_TRAIN_TAG" \
  --eval-tags "$ATTNRES_DEV_TAG" \
  --output-tag "${TAG_PREFIX}_attnres_dev_b${BANK_SIZE}" \
  --feature-mode attnres

"$PY" scripts/train_candidate_conditioned_ranker_v7.py \
  --bank-tag "$ATTNRES_BANK_TAG" \
  --bank-size "$BANK_SIZE" \
  --train-tags "$ATTNRES_TRAIN_TAG" \
  --eval-tags "$ATTNRES_DEV_TAG" \
  --hidden-train-tags "$ATTNRES_TRAIN_TAG" \
  --hidden-eval-tags "$ATTNRES_DEV_TAG" \
  --output-tag "${TAG_PREFIX}_hidden_dev_b${BANK_SIZE}" \
  --feature-mode hidden \
  --fast-mode

run_oracle_group "$STD_CKPT" "$TRAIN_MANIFEST" "$STD_TRAIN_TAG" "$TRAIN_SHARDS"
run_oracle_group "$STD_CKPT" "$VAL_MANIFEST" "$STD_VAL_TAG" "$VAL_SHARDS"
run_oracle_group "$STD_CKPT" "$DEV_MANIFEST" "$STD_DEV_TAG" "$DEV_SHARDS"

if [[ ! -f "results/bank_hygiene/${STD_BANK_TAG}_summary.csv" ]]; then
  "$PY" scripts/evaluate_train_only_bank_hygiene.py \
    --calib-tags "$STD_VAL_TAG" \
    --eval-tags "$STD_DEV_TAG" \
    --output-tag "$STD_BANK_TAG" \
    --bank-sizes 32 64
fi

run_hidden_group "$STD_CKPT" "$TRAIN_MANIFEST" "$STD_TRAIN_TAG" "$TRAIN_SHARDS"
run_hidden_group "$STD_CKPT" "$DEV_MANIFEST" "$STD_DEV_TAG" "$DEV_SHARDS"

"$PY" scripts/train_candidate_conditioned_ranker_v7.py \
  --bank-tag "$STD_BANK_TAG" \
  --bank-size "$BANK_SIZE" \
  --train-tags "$STD_TRAIN_TAG" \
  --eval-tags "$STD_DEV_TAG" \
  --hidden-train-tags "$STD_TRAIN_TAG" \
  --hidden-eval-tags "$STD_DEV_TAG" \
  --output-tag "${TAG_PREFIX}_std_hidden_dev_b${BANK_SIZE}" \
  --feature-mode hidden \
  --fast-mode

echo "[wikitext-necessity-v7] complete"
