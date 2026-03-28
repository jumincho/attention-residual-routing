#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$ROOT/results/scale24x512_ccnews_attnres_v5}"
MANIFEST_PATH="${MANIFEST_PATH:-$ROOT/results/selector_data_scale/v5_ccnews_p256d64_stride160_validation256.jsonl}"
CHECKPOINT_STEPS=(${CHECKPOINT_STEPS:-100 500 1000 2000 5000})
SHARDS="${SHARDS:-4}"
PROMPT_LEN="${PROMPT_LEN:-256}"
DECODE_LEN="${DECODE_LEN:-64}"
BATCH_SIZE="${BATCH_SIZE:-64}"

for step in "${CHECKPOINT_STEPS[@]}"; do
  ckpt="$(printf '%s/checkpoint_step_%06d.pt' "${EXPERIMENT_DIR}" "${step}")"
  if [[ ! -f "${ckpt}" ]]; then
    echo "[routing_ckpt_v5] missing checkpoint ${ckpt}, skipping"
    continue
  fi
  base_tag="v5_ccnews_step$(printf '%04d' "${step}")_cal256"
  echo "[routing_ckpt_v5] step=${step} checkpoint=${ckpt}"
  "${PYTHON_BIN}" "$ROOT/scripts/run_sharded_oracle_eval.py" \
    --checkpoint "${ckpt}" \
    --manifest-path "${MANIFEST_PATH}" \
    --prompt-len "${PROMPT_LEN}" \
    --decode-len "${DECODE_LEN}" \
    --tag-prefix "${base_tag}" \
    --total-shards "${SHARDS}" \
    --gpus 0 1 2 3 \
    --batch-size "${BATCH_SIZE}" \
    --num-chunks 4 \
    --score-mode utility_over_variance \
    --skip-counts 1 2 3
  tags=()
  for ((shard=0; shard<SHARDS; shard++)); do
    tags+=("${base_tag}_s$(printf '%02d' "${shard}")")
  done
  "${PYTHON_BIN}" "$ROOT/scripts/merge_oracle_shards.py" --tags "${tags[@]}" --output-tag "${base_tag}"
done

echo "[routing_ckpt_v5] complete"
