#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$ROOT/.venv/bin/python}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$ROOT/results/scale24x512_ccnews_attnres_v5}"
CHECKPOINT_STEPS_RAW="${CHECKPOINT_STEPS-1000 2000 3000 5000}"
BASH_CHECKPOINT_STEPS=()
if [[ -n "${CHECKPOINT_STEPS_RAW// }" ]]; then
  # shellcheck disable=SC2206
  BASH_CHECKPOINT_STEPS=(${CHECKPOINT_STEPS_RAW})
fi
BANK_SIZES=(${BANK_SIZES:-16 32 64 128})
GPUS=(${GPUS:-0 1 2 3})
LOG_DIR="${LOG_DIR:-$ROOT/results/hetero_bank_hygiene_v6/logs}"
mkdir -p "$LOG_DIR"

MAIN_TAG="${MAIN_TAG:-v6_ccnews_p256d64_stride160}"
MAIN_PROMPT_LEN="${MAIN_PROMPT_LEN:-256}"
MAIN_DECODE_LEN="${MAIN_DECODE_LEN:-64}"
MAIN_STRIDE="${MAIN_STRIDE:-160}"
MAIN_TRAIN_TARGET="${MAIN_TRAIN_TARGET:-8192}"
MAIN_VAL_TARGET="${MAIN_VAL_TARGET:-1024}"
MAIN_TEST_TARGET="${MAIN_TEST_TARGET:-1024}"

AUX_TAG="${AUX_TAG:-v6_ccnews_p128d128_stride128}"
AUX_PROMPT_LEN="${AUX_PROMPT_LEN:-128}"
AUX_DECODE_LEN="${AUX_DECODE_LEN:-128}"
AUX_STRIDE="${AUX_STRIDE:-128}"
AUX_TRAIN_TARGET="${AUX_TRAIN_TARGET:-4096}"
AUX_VAL_TARGET="${AUX_VAL_TARGET:-1024}"
AUX_TEST_TARGET="${AUX_TEST_TARGET:-1024}"
AUX_STEPS_RAW="${AUX_STEPS-1000 3000}"
BASH_AUX_STEPS=()
if [[ -n "${AUX_STEPS_RAW// }" ]]; then
  # shellcheck disable=SC2206
  BASH_AUX_STEPS=(${AUX_STEPS_RAW})
fi

BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_CHUNKS="${NUM_CHUNKS:-4}"

build_manifest_if_missing() {
  local tag="$1"
  local prompt_len="$2"
  local decode_len="$3"
  local stride="$4"
  local train_target="$5"
  local val_target="$6"
  local test_target="$7"

  if [[ -f "$ROOT/results/selector_data_scale/${tag}_train.jsonl" ]] \
    && [[ -f "$ROOT/results/selector_data_scale/${tag}_validation.jsonl" ]] \
    && [[ -f "$ROOT/results/selector_data_scale/${tag}_test.jsonl" ]]; then
    return
  fi

  echo "[v6-bank-audit] building manifests tag=${tag} prompt=${prompt_len} decode=${decode_len} stride=${stride}"
  "$PY" "$ROOT/scripts/build_selector_sequence_manifest.py" \
    --dataset-name cc_news \
    --tokenizer-name openai-community/gpt2 \
    --prompt-len "$prompt_len" \
    --decode-len "$decode_len" \
    --train-target "$train_target" \
    --validation-target "$val_target" \
    --test-target "$test_target" \
    --stride "$stride" \
    --tag "$tag"
}

checkpoint_for_step() {
  local step="$1"
  local candidate
  candidate="$(printf '%s/checkpoint_step_%06d.pt' "$EXPERIMENT_DIR" "$step")"
  if [[ -f "$candidate" ]]; then
    echo "$candidate"
    return
  fi
  if [[ "$step" == "3000" ]]; then
    echo "$EXPERIMENT_DIR/best_checkpoint.pt"
    return
  fi
  echo "missing checkpoint for step ${step}" >&2
  return 1
}

run_oracle_group() {
  local checkpoint_path="$1"
  local manifest_path="$2"
  local prompt_len="$3"
  local decode_len="$4"
  local prefix="$5"
  local total_shards="$6"
  local logfile="$LOG_DIR/${prefix}.log"

  if [[ -f "$ROOT/results/oracles/${prefix}_oracle_mask_alignment_summary.csv" ]]; then
    echo "[v6-bank-audit] skip oracle ${prefix}"
    return
  fi

  echo "[v6-bank-audit] oracle ${prefix} prompt=${prompt_len} decode=${decode_len} shards=${total_shards}"
  "$PY" "$ROOT/scripts/run_sharded_oracle_eval.py" \
    --checkpoint "$checkpoint_path" \
    --manifest-path "$manifest_path" \
    --prompt-len "$prompt_len" \
    --decode-len "$decode_len" \
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

run_bank_audit() {
  local setting_tag="$1"
  local step="$2"

  local val_tag="${setting_tag}_step${step}_val"
  local test_tag="${setting_tag}_step${step}_test"
  local bank_tag="${setting_tag}_step${step}_bank"
  local checkpoint_path
  checkpoint_path="$(checkpoint_for_step "$step")"

  local prompt_len decode_len train_manifest val_manifest test_manifest
  if [[ "$setting_tag" == "$MAIN_TAG" ]]; then
    prompt_len="$MAIN_PROMPT_LEN"
    decode_len="$MAIN_DECODE_LEN"
  else
    prompt_len="$AUX_PROMPT_LEN"
    decode_len="$AUX_DECODE_LEN"
  fi
  val_manifest="$ROOT/results/selector_data_scale/${setting_tag}_validation.jsonl"
  test_manifest="$ROOT/results/selector_data_scale/${setting_tag}_test.jsonl"

  run_oracle_group "$checkpoint_path" "$val_manifest" "$prompt_len" "$decode_len" "$val_tag" 4
  run_oracle_group "$checkpoint_path" "$test_manifest" "$prompt_len" "$decode_len" "$test_tag" 4

  if [[ -f "$ROOT/results/bank_hygiene/${bank_tag}_summary.csv" ]]; then
    echo "[v6-bank-audit] skip bank hygiene ${bank_tag}"
    return
  fi

  echo "[v6-bank-audit] bank hygiene ${bank_tag}"
  "$PY" "$ROOT/scripts/evaluate_train_only_bank_hygiene.py" \
    --calib-tags "$val_tag" \
    --eval-tags "$test_tag" \
    --output-tag "$bank_tag" \
    --bank-sizes "${BANK_SIZES[@]}"
}

main() {
  cd "$ROOT"
  build_manifest_if_missing "$MAIN_TAG" "$MAIN_PROMPT_LEN" "$MAIN_DECODE_LEN" "$MAIN_STRIDE" "$MAIN_TRAIN_TARGET" "$MAIN_VAL_TARGET" "$MAIN_TEST_TARGET"
  if [[ ${#BASH_AUX_STEPS[@]} -gt 0 ]]; then
    build_manifest_if_missing "$AUX_TAG" "$AUX_PROMPT_LEN" "$AUX_DECODE_LEN" "$AUX_STRIDE" "$AUX_TRAIN_TARGET" "$AUX_VAL_TARGET" "$AUX_TEST_TARGET"
  fi

  echo "[v6-bank-audit] main setting: ${MAIN_TAG} (${MAIN_PROMPT_LEN}/${MAIN_DECODE_LEN})"
  echo "[v6-bank-audit] aux setting: ${AUX_TAG} (${AUX_PROMPT_LEN}/${AUX_DECODE_LEN})"
  echo "[v6-bank-audit] steps: ${BASH_CHECKPOINT_STEPS[*]:-<none>}"
  echo "[v6-bank-audit] aux steps: ${BASH_AUX_STEPS[*]:-<none>}"

  for step in "${BASH_CHECKPOINT_STEPS[@]}"; do
    run_bank_audit "$MAIN_TAG" "$step"
  done

  summary_tags=()
  for step in "${BASH_CHECKPOINT_STEPS[@]}"; do
    summary_tags+=("${MAIN_TAG}_step${step}_bank")
  done

  for step in "${BASH_AUX_STEPS[@]}"; do
    run_bank_audit "$AUX_TAG" "$step"
    summary_tags+=("${AUX_TAG}_step${step}_bank")
  done

  if [[ ${#summary_tags[@]} -gt 0 ]]; then
    "$PY" "$ROOT/scripts/summarize_hetero_bank_audit_v6.py" \
      --bank-tags "${summary_tags[@]}" \
      --output-dir "$ROOT/results/hetero_bank_hygiene_v6"
  fi
}

main "$@"
