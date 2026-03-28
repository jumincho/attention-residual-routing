#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:?set EXPERIMENT_DIR=/abs/path/to/experiment_dir}"
CHECKPOINT_STEPS_RAW="${CHECKPOINT_STEPS:?set CHECKPOINT_STEPS='3000 3500 4000 4500 5000 5500 6000'}"
BANK_SIZES=(${BANK_SIZES:-32 64 128})
GPUS=(${GPUS:-0 1 2 3})
PROMPT_LEN="${PROMPT_LEN:-256}"
DECODE_LEN="${DECODE_LEN:-64}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_CHUNKS="${NUM_CHUNKS:-4}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:?set TRAIN_MANIFEST=/abs/path/to/train.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:?set VAL_MANIFEST=/abs/path/to/validation.jsonl}"
DEV_MANIFEST="${DEV_MANIFEST:?set DEV_MANIFEST=/abs/path/to/dev_test.jsonl}"
TAG_PREFIX="${TAG_PREFIX:?set TAG_PREFIX=v7_ccnews_p256d64_lockbox}"
SUMMARY_DIR="${SUMMARY_DIR:-$ROOT/results/hetero_bank_hygiene_v7}"
LOG_DIR="${LOG_DIR:-$SUMMARY_DIR/logs}"
mkdir -p "$LOG_DIR" "$SUMMARY_DIR"

read -r -a CHECKPOINT_STEPS <<<"$CHECKPOINT_STEPS_RAW"

checkpoint_for_step() {
  local step="$1"
  local candidate
  candidate="$(printf '%s/checkpoint_step_%06d.pt' "$EXPERIMENT_DIR" "$step")"
  if [[ -f "$candidate" ]]; then
    echo "$candidate"
    return
  fi
  echo "missing checkpoint for step ${step} in ${EXPERIMENT_DIR}" >&2
  return 1
}

run_oracle_group() {
  local checkpoint_path="$1"
  local manifest_path="$2"
  local prefix="$3"
  local total_shards="$4"
  local logfile="$LOG_DIR/${prefix}.log"
  if [[ -f "$ROOT/results/oracles/${prefix}_oracle_mask_alignment_summary.csv" ]]; then
    echo "[v7-bank-audit] skip oracle ${prefix}"
    return
  fi
  echo "[v7-bank-audit] oracle ${prefix}"
  "$PY" "$ROOT/scripts/run_sharded_oracle_eval.py" \
    --checkpoint "$checkpoint_path" \
    --manifest-path "$manifest_path" \
    --prompt-len "$PROMPT_LEN" \
    --decode-len "$DECODE_LEN" \
    --tag-prefix "$prefix" \
    --total-shards "$total_shards" \
    --gpus "${GPUS[@]}" \
    --batch-size "$BATCH_SIZE" \
    --num-chunks "$NUM_CHUNKS" \
    --score-mode utility_over_variance \
    --skip-counts 1 2 3 \
    >"$logfile" 2>&1
  mapfile -t shard_tags < <(seq -f "${prefix}_s%02g" 0 $((total_shards - 1)))
  "$PY" "$ROOT/scripts/merge_oracle_shards.py" --tags "${shard_tags[@]}" --output-tag "$prefix"
}

summary_tags=()
for step in "${CHECKPOINT_STEPS[@]}"; do
  checkpoint_path="$(checkpoint_for_step "$step")"
  val_tag="${TAG_PREFIX}_step${step}_val"
  dev_tag="${TAG_PREFIX}_step${step}_dev"
  bank_tag="${TAG_PREFIX}_step${step}_bank"

  run_oracle_group "$checkpoint_path" "$VAL_MANIFEST" "$val_tag" 4
  run_oracle_group "$checkpoint_path" "$DEV_MANIFEST" "$dev_tag" 4

  if [[ ! -f "$ROOT/results/bank_hygiene/${bank_tag}_summary.csv" ]]; then
    "$PY" "$ROOT/scripts/evaluate_train_only_bank_hygiene.py" \
      --calib-tags "$val_tag" \
      --eval-tags "$dev_tag" \
      --output-tag "$bank_tag" \
      --bank-sizes "${BANK_SIZES[@]}"
  fi
  summary_tags+=("$bank_tag")
done

"$PY" "$ROOT/scripts/summarize_hetero_bank_audit_v6.py" \
  --bank-tags "${summary_tags[@]}" \
  --output-dir "$SUMMARY_DIR"

echo "[v7-bank-audit] complete"
