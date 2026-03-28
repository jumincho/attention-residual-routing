#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

WAIT_PID="${WAIT_PID:-}"
if [[ -n "$WAIT_PID" ]]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 60
  done
fi

PY="$ROOT/.venv/bin/python"
mkdir -p "$ROOT/results/readiness_v8"

for seed in 43 44; do
  case "$seed" in
    43)
      EXPERIMENT_DIR="$ROOT/results/scale24x512_ccnews_attnres_seed43_v8"
      TAG_PREFIX="v8_ccnews_seed43_fast"
      ;;
    44)
      EXPERIMENT_DIR="$ROOT/results/scale24x512_ccnews_attnres_seed44_v8"
      TAG_PREFIX="v8_ccnews_seed44_fast"
      ;;
    *)
      echo "unsupported seed ${seed}" >&2
      exit 1
      ;;
  esac

  if [[ ! -f "$ROOT/results/bank_hygiene/${TAG_PREFIX}_step6000_bank_summary.csv" ]]; then
    EXPERIMENT_DIR="$EXPERIMENT_DIR" \
    CHECKPOINT_STEPS="2500 3000 3500 4000 4500 5000 5500 6000" \
    TAG_PREFIX="$TAG_PREFIX" \
    BANK_SIZES="32" \
    BATCH_SIZE="${BATCH_SIZE:-128}" \
    bash "$ROOT/scripts/run_ccnews_fast_bank_audit_v8.sh"
  fi

  TAGS=()
  for step in 2500 3000 3500 4000 4500 5000 5500 6000; do
    TAGS+=("${TAG_PREFIX}_step${step}_val")
  done
  "$PY" "$ROOT/scripts/compile_routing_readiness_v8.py" \
    --tags "${TAGS[@]}" \
    --experiment-dir "$EXPERIMENT_DIR" \
    --bank-size 32 \
    --bank-skip 1 \
    --feature-mode attnres \
    --selector-prefix "__missing__step{step}_dev2048_b32" \
    --output-tag "${TAG_PREFIX}_main"

  CHECKPOINT_STEPS="$("$PY" - <<PY
import pandas as pd
df = pd.read_csv("$ROOT/results/readiness_v8/${TAG_PREFIX}_main_routing_readiness_v4.csv")
steps = [3000]
alts = [int(x) for x in df.sort_values("routing_readiness_v4", ascending=False)["step"].tolist() if int(x) != 3000]
steps.extend(alts[: max(1, int(${TOP_N:-2}) - 1)])
dedup = []
for step in steps:
    if step not in dedup:
        dedup.append(step)
print(" ".join(str(step) for step in dedup))
PY
)"
  echo "[ccnews-followup-after-seed42-v8] seed=${seed} shortlist=${CHECKPOINT_STEPS}"
  EXPERIMENT_DIR="$EXPERIMENT_DIR" \
  CHECKPOINT_STEPS="$CHECKPOINT_STEPS" \
  TRAIN_MANIFEST="$ROOT/results/lockbox_manifests_v8/v8_ccnews_p256d64_lockbox_train2048.jsonl" \
  VAL_MANIFEST="$ROOT/results/lockbox_manifests_v8/v8_ccnews_p256d64_lockbox_validation.jsonl" \
  EVAL_MANIFEST="$ROOT/results/lockbox_manifests_v8/v8_ccnews_p256d64_lockbox_dev_select2048.jsonl" \
  TAG_PREFIX="${TAG_PREFIX/fast/full}" \
  OUTPUT_PREFIX_BASE="${TAG_PREFIX/fast/full}" \
  BANK_SIZES="${BANK_SIZES:-32 64}" \
  OUTPUT_SUBDIR="regret_reduction_v8" \
  PLOT_PREFIX="regret_reduction_v8" \
  BATCH_SIZE="${BATCH_SIZE:-128}" \
  bash "$ROOT/scripts/run_ccnews_shortlist_full_scorer_v8.sh"
done

echo "[ccnews-followup-after-seed42-v8] complete"
