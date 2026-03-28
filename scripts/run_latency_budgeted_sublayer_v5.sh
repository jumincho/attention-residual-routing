#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
CHECKPOINT="${CHECKPOINT:-${1:-$ROOT/results/benchmark24x384_wikitext_attnres_ddp_bs24/best_checkpoint.pt}}"
LATENCY_CSV="${LATENCY_CSV:-$ROOT/results/headroom/audit_best200_latency_microbench.csv}"
ANCHOR_BANK_CSV="${ANCHOR_BANK_CSV:-$ROOT/results/bank_hygiene/v5clean_best200_p256d64_bank_val_candidate_bank.csv}"
PROMPT_LEN="${PROMPT_LEN:-256}"
DECODE_LEN="${DECODE_LEN:-64}"
ANCHOR_BANK_SIZE="${ANCHOR_BANK_SIZE:-32}"
MAX_EDITS="${MAX_EDITS:-2}"
MIN_REDUCTION="${MIN_REDUCTION:-0.02}"
MAX_REDUCTION="${MAX_REDUCTION:-0.20}"
BATCH_SIZE="${BATCH_SIZE:-128}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-$ROOT/results/selector_data_scale/v5_best200_p256d64_stride160_train4k.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-$ROOT/results/selector_data_scale/v5_best200_p256d64_stride160_val1k.jsonl}"
TEST_MANIFEST="${TEST_MANIFEST:-$ROOT/results/selector_data_scale/v5_best200_p256d64_stride160_test1k.jsonl}"
MARKER_FILE="${MARKER_FILE:-$ROOT/results/latency_budgeted_sublayer_v5/complete.txt}"

if [[ ! -f "${TRAIN_MANIFEST}" ]]; then
  "${PYTHON_BIN}" "$ROOT/scripts/subset_manifest.py" \
    --input "$ROOT/results/selector_data_scale/v5_best200_p256d64_stride160_train.jsonl" \
    --output "${TRAIN_MANIFEST}" \
    --target-count 4096 \
    --mode head
fi
if [[ ! -f "${VAL_MANIFEST}" ]]; then
  "${PYTHON_BIN}" "$ROOT/scripts/subset_manifest.py" \
    --input "$ROOT/results/selector_data_scale/v5_best200_p256d64_stride160_validation.jsonl" \
    --output "${VAL_MANIFEST}" \
    --target-count 1024 \
    --mode head
fi
if [[ ! -f "${TEST_MANIFEST}" ]]; then
  "${PYTHON_BIN}" "$ROOT/scripts/subset_manifest.py" \
    --input "$ROOT/results/selector_data_scale/v5_best200_p256d64_stride160_test.jsonl" \
    --output "${TEST_MANIFEST}" \
    --target-count 1024 \
    --mode head
fi

run_split() {
  local split_name="$1"
  local manifest_path="$2"
  local total_shards="$3"
  local output_tag="$4"

  echo "[sublayer_v5] split=${split_name} total_shards=${total_shards} tag=${output_tag}"
  local start=0
  while [[ "${start}" -lt "${total_shards}" ]]; do
    local pids=()
    for gpu in 0 1 2 3; do
      local shard=$((start + gpu))
      if [[ "${shard}" -ge "${total_shards}" ]]; then
        break
      fi
      CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "$ROOT/scripts/evaluate_sublayer_candidate_losses_v5.py" \
        --checkpoint "${CHECKPOINT}" \
        --manifest-path "${manifest_path}" \
        --anchor-bank-csv "${ANCHOR_BANK_CSV}" \
        --latency-csv "${LATENCY_CSV}" \
        --tag "${output_tag}" \
        --prompt-len "${PROMPT_LEN}" \
        --decode-len "${DECODE_LEN}" \
        --anchor-bank-size "${ANCHOR_BANK_SIZE}" \
        --max-edits "${MAX_EDITS}" \
        --min-reduction "${MIN_REDUCTION}" \
        --max-reduction "${MAX_REDUCTION}" \
        --batch-size "${BATCH_SIZE}" \
        --num-shards "${total_shards}" \
        --shard-index "${shard}" \
        --device cuda \
        --precision fp16 &
      pids+=($!)
    done
    for pid in "${pids[@]}"; do
      wait "${pid}"
    done
    start=$((start + 4))
  done

  local tags=()
  for ((shard=0; shard<total_shards; shard++)); do
    tags+=("${output_tag}_s$(printf '%02d' "${shard}")")
  done
  "${PYTHON_BIN}" "$ROOT/scripts/merge_sublayer_shards_v5.py" --tags "${tags[@]}" --output-tag "${output_tag}"
}

run_split "train4k" "${TRAIN_MANIFEST}" 8 "v5_sublayer_best200_train4k"
run_split "val1k" "${VAL_MANIFEST}" 4 "v5_sublayer_best200_val1k"
run_split "test1k" "${TEST_MANIFEST}" 4 "v5_sublayer_best200_test1k"

printf 'checkpoint=%s\n' "${CHECKPOINT}" > "${MARKER_FILE}"
echo "[sublayer_v5] complete"
