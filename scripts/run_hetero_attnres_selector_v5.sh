#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
READINESS_CSV="${READINESS_CSV:-$ROOT/results/routing_checkpoint_selection_v5/v5_ccnews_attnres_routing_readiness.csv}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$ROOT/results/scale24x512_ccnews_attnres_v5}"
BEST_CKPT_PATH="${BEST_CKPT_PATH:-$EXPERIMENT_DIR/best_checkpoint.pt}"
TRAIN_FULL="${TRAIN_FULL:-$ROOT/results/selector_data_scale/v5_ccnews_p256d64_stride160_train.jsonl}"
VAL256_MANIFEST="${VAL256_MANIFEST:-$ROOT/results/selector_data_scale/v5_ccnews_p256d64_stride160_validation256.jsonl}"
TEST512_MANIFEST="${TEST512_MANIFEST:-$ROOT/results/selector_data_scale/v5_ccnews_p256d64_stride160_test512.jsonl}"
TRAIN2048_MANIFEST="${TRAIN2048_MANIFEST:-$ROOT/results/selector_data_scale/v5_ccnews_p256d64_stride160_train2048.jsonl}"
LOG_DIR="$ROOT/results/scale_heterogeneity_v5/logs"
mkdir -p "$LOG_DIR"
MARKER_FILE="$ROOT/results/scale_heterogeneity_v5/v5_ccnews_routebest_selector_complete.txt"
ROLE_CSV="$ROOT/results/scale_heterogeneity_v5/checkpoint_roles.csv"

PROMPT_LEN="${PROMPT_LEN:-256}"
DECODE_LEN="${DECODE_LEN:-64}"
BATCH_SIZE="${BATCH_SIZE:-64}"
BANK_SIZE="${BANK_SIZE:-32}"

if [[ ! -f "$TRAIN2048_MANIFEST" ]]; then
  "$PY" scripts/subset_manifest.py --input "$TRAIN_FULL" --output "$TRAIN2048_MANIFEST" --target-count 2048 --mode head
fi

read -r ROUTEBEST_STEP LMBEST_STEP <<<"$("$PY" - <<'PY'
import pandas as pd
import torch
from pathlib import Path
readiness_path = Path("results/routing_checkpoint_selection_v5/v5_ccnews_attnres_routing_readiness.csv")
best_ckpt_path = Path("results/scale24x512_ccnews_attnres_v5/best_checkpoint.pt")
df = pd.read_csv(readiness_path)
routebest = int(df.sort_values("routing_readiness_score", ascending=False).iloc[0]["step"])
lmbest = int(torch.load(best_ckpt_path, map_location="cpu").get("step", -1))
print(routebest, lmbest)
PY
)"

echo "[hetero-v5] routing-best step=${ROUTEBEST_STEP} lm-best step=${LMBEST_STEP}"
echo "[hetero-v5] estimated test512 oracle time per checkpoint: ~12-20m on 4 GPUs"
echo "[hetero-v5] estimated routebest train2048 oracle + hidden + ranker: ~45-70m total"

run_oracle_group() {
  local ckpt="$1"
  local manifest_path="$2"
  local prefix="$3"
  local total_shards="$4"
  local logfile="$LOG_DIR/${prefix}_oracle.log"
  if [[ -f "results/oracles/${prefix}_oracle_mask_alignment_summary.csv" ]]; then
    echo "[hetero-v5] skip oracle $prefix"
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
    --skip-counts 1 2 3 \
    >"$logfile" 2>&1
  mapfile -t shard_tags < <(seq -f "${prefix}_s%02g" 0 $((total_shards - 1)))
  "$PY" scripts/merge_oracle_shards.py --tags "${shard_tags[@]}" --output-tag "$prefix"
}

checkpoint_for_step() {
  local step="$1"
  local candidate="$EXPERIMENT_DIR/checkpoint_$(printf 'step_%06d.pt' "$step")"
  if [[ -f "$candidate" ]]; then
    echo "$candidate"
  else
    echo "$BEST_CKPT_PATH"
  fi
}

run_hidden_group() {
  local ckpt="$1"
  local manifest_path="$2"
  local prefix="$3"
  local total_shards="$4"
  local logfile="$LOG_DIR/${prefix}_hidden.log"
  if [[ -f "results/rich_features/${prefix}_hidden_prompt_features.csv" ]]; then
    echo "[hetero-v5] skip hidden $prefix"
    return
  fi
  "$PY" scripts/run_sharded_hidden_extract.py \
    --checkpoint "$ckpt" \
    --manifest-path "$manifest_path" \
    --prompt-len "$PROMPT_LEN" \
    --decode-len "$DECODE_LEN" \
    --tag-prefix "$prefix" \
    --total-shards "$total_shards" \
    --gpus 0 1 2 3 \
    >"$logfile" 2>&1
  mapfile -t shard_tags < <(seq -f "${prefix}_s%02g" 0 $((total_shards - 1)))
  "$PY" scripts/merge_hidden_shards.py --tags "${shard_tags[@]}" --output-tag "$prefix"
}

run_test_bank() {
  local step="$1"
  local ckpt="$2"
  local step_tag
  step_tag="$(printf 'step%04d' "$step")"
  local val_tag="v5_ccnews_${step_tag}_cal256"
  local test_tag="v5_ccnews_${step_tag}_test512"
  local bank_tag="v5_ccnews_${step_tag}_bank_test"
  if [[ ! -f "results/oracles/${val_tag}_oracle_mask_alignment_summary.csv" ]]; then
    run_oracle_group "$ckpt" "$VAL256_MANIFEST" "$val_tag" 4
  fi
  run_oracle_group "$ckpt" "$TEST512_MANIFEST" "$test_tag" 4
  if [[ ! -f "results/bank_hygiene/${bank_tag}_summary.csv" ]]; then
    "$PY" scripts/evaluate_train_only_bank_hygiene.py \
      --calib-tags "$val_tag" \
      --eval-tags "$test_tag" \
      --output-tag "$bank_tag" \
      --bank-sizes 16 32 64
  fi
}

LMBEST_CKPT="$(checkpoint_for_step "$LMBEST_STEP")"
ROUTEBEST_CKPT="$(checkpoint_for_step "$ROUTEBEST_STEP")"

cat > "$ROLE_CSV" <<EOF
role,step,checkpoint_path
lm_best,$LMBEST_STEP,$LMBEST_CKPT
routebest,$ROUTEBEST_STEP,$ROUTEBEST_CKPT
EOF

run_test_bank "$LMBEST_STEP" "$LMBEST_CKPT"
if [[ "$ROUTEBEST_STEP" != "$LMBEST_STEP" ]]; then
  run_test_bank "$ROUTEBEST_STEP" "$ROUTEBEST_CKPT"
fi

ROUTEBEST_TAG="$(printf 'v5_ccnews_step%04d' "$ROUTEBEST_STEP")"
ROUTEBEST_TRAIN_TAG="${ROUTEBEST_TAG}_train2048"
ROUTEBEST_TEST_TAG="${ROUTEBEST_TAG}_test512"
ROUTEBEST_BANK_TAG="${ROUTEBEST_TAG}_bank_test"
ROUTEBEST_RANKER_TAG="${ROUTEBEST_TAG}_test512_b32"

run_oracle_group "$ROUTEBEST_CKPT" "$TRAIN2048_MANIFEST" "$ROUTEBEST_TRAIN_TAG" 8
run_hidden_group "$ROUTEBEST_CKPT" "$TRAIN2048_MANIFEST" "$ROUTEBEST_TRAIN_TAG" 8
run_hidden_group "$ROUTEBEST_CKPT" "$TEST512_MANIFEST" "$ROUTEBEST_TEST_TAG" 4

if [[ ! -f "results/ranker_v5/${ROUTEBEST_RANKER_TAG}_attnres_summary.csv" ]]; then
  "$PY" scripts/train_candidate_conditioned_ranker_v5.py \
    --bank-tag "$ROUTEBEST_BANK_TAG" \
    --bank-size "$BANK_SIZE" \
    --train-tags "$ROUTEBEST_TRAIN_TAG" \
    --eval-tags "$ROUTEBEST_TEST_TAG" \
    --hidden-train-tags "$ROUTEBEST_TRAIN_TAG" \
    --hidden-eval-tags "$ROUTEBEST_TEST_TAG" \
    --output-tag "$ROUTEBEST_RANKER_TAG" \
    --feature-mode attnres
fi

if [[ ! -f "results/ranker_v5/${ROUTEBEST_RANKER_TAG}_hidden_summary.csv" ]]; then
  "$PY" scripts/train_candidate_conditioned_ranker_v5.py \
    --bank-tag "$ROUTEBEST_BANK_TAG" \
    --bank-size "$BANK_SIZE" \
    --train-tags "$ROUTEBEST_TRAIN_TAG" \
    --eval-tags "$ROUTEBEST_TEST_TAG" \
    --hidden-train-tags "$ROUTEBEST_TRAIN_TAG" \
    --hidden-eval-tags "$ROUTEBEST_TEST_TAG" \
    --output-tag "$ROUTEBEST_RANKER_TAG" \
    --feature-mode hidden
fi

printf 'routebest_step=%s\nlmbest_step=%s\n' "$ROUTEBEST_STEP" "$LMBEST_STEP" > "$MARKER_FILE"
echo "[hetero-v5] complete"
