#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$ROOT/results/scale24x512_ccnews_attnres_v5}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-$ROOT/results/selector_data_scale/v6_ccnews_p256d64_stride160_train4096.jsonl}"
EVAL_MANIFEST="${EVAL_MANIFEST:-$ROOT/results/selector_data_scale/v6_ccnews_p256d64_stride160_test.jsonl}"
PROMPT_LEN="${PROMPT_LEN:-256}"
DECODE_LEN="${DECODE_LEN:-64}"
TRAIN_SHARDS="${TRAIN_SHARDS:-8}"
EVAL_SHARDS="${EVAL_SHARDS:-4}"
GPUS="${GPUS:-0 1 2 3}"
STEPS_RAW="${STEPS:-1000 3000}"

read -r -a STEPS_ARR <<<"$STEPS_RAW"

checkpoint_for_step() {
  local step="$1"
  if [[ "$step" == "3000" ]]; then
    echo "$EXPERIMENT_DIR/best_checkpoint.pt"
  else
    printf '%s/checkpoint_step_%06d.pt\n' "$EXPERIMENT_DIR" "$step"
  fi
}

run_hidden_group() {
  local ckpt="$1"
  local manifest_path="$2"
  local prefix="$3"
  local total_shards="$4"
  if [[ -f "results/rich_features/${prefix}_hidden_prompt_features.csv" ]]; then
    echo "[hetero-hidden-v6] skip hidden ${prefix}"
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

for STEP in "${STEPS_ARR[@]}"; do
  CHECKPOINT="$(checkpoint_for_step "$STEP")"
  if [[ ! -f "$CHECKPOINT" ]]; then
    echo "[hetero-hidden-v6] missing checkpoint for step=${STEP}: ${CHECKPOINT}" >&2
    exit 1
  fi

  TRAIN_TAG="${TRAIN_TAG_PREFIX:-v6_ccnews_p256d64_stride160_step}${STEP}_train4096"
  EVAL_TAG="${EVAL_TAG_PREFIX:-v6_ccnews_p256d64_stride160_step}${STEP}_test"

  echo "[hetero-hidden-v6] step=${STEP} checkpoint=${CHECKPOINT}"
  run_hidden_group "$CHECKPOINT" "$TRAIN_MANIFEST" "$TRAIN_TAG" "$TRAIN_SHARDS"
  run_hidden_group "$CHECKPOINT" "$EVAL_MANIFEST" "$EVAL_TAG" "$EVAL_SHARDS"
done

echo "[hetero-hidden-v6] complete"
