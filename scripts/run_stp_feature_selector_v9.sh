#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
BANK_TAG="${BANK_TAG:?set BANK_TAG=/bank_tag/}"
BANK_SIZE="${BANK_SIZE:-32}"
TRAIN_TAGS_RAW="${TRAIN_TAGS:?set TRAIN_TAGS='train_tag'}"
EVAL_TAGS_RAW="${EVAL_TAGS:?set EVAL_TAGS='eval_tag'}"
HIDDEN_TRAIN_TAGS_RAW="${HIDDEN_TRAIN_TAGS:-$TRAIN_TAGS_RAW}"
HIDDEN_EVAL_TAGS_RAW="${HIDDEN_EVAL_TAGS:-$EVAL_TAGS_RAW}"
OUTPUT_TAG="${OUTPUT_TAG:?set OUTPUT_TAG=v9_stp_selector_run}"
FEATURE_MODES_RAW="${FEATURE_MODES:-attnres stp_scalar attnres_stp_scalar attnres_stp_diff hidden hidden_stp_diff}"
SKIP_COUNTS_RAW="${SKIP_COUNTS:-1}"
FAST_MODE="${FAST_MODE:-1}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-stp_feature_selector_v9}"
PLOT_PREFIX="${PLOT_PREFIX:-stp_feature_selector_v9}"
HIDDEN_PCA_DIM="${HIDDEN_PCA_DIM:-64}"
STP_PCA_DIM="${STP_PCA_DIM:-64}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

read -r -a TRAIN_TAGS <<<"$TRAIN_TAGS_RAW"
read -r -a EVAL_TAGS <<<"$EVAL_TAGS_RAW"
read -r -a HIDDEN_TRAIN_TAGS <<<"$HIDDEN_TRAIN_TAGS_RAW"
read -r -a HIDDEN_EVAL_TAGS <<<"$HIDDEN_EVAL_TAGS_RAW"
read -r -a FEATURE_MODES <<<"$FEATURE_MODES_RAW"
read -r -a SKIP_COUNTS <<<"$SKIP_COUNTS_RAW"

for feature_mode in "${FEATURE_MODES[@]}"; do
  summary_path="$ROOT/results/${OUTPUT_SUBDIR}/${OUTPUT_TAG}_${feature_mode}_summary.csv"
  if [[ "$SKIP_EXISTING" == "1" && -f "$summary_path" ]]; then
    echo "[stp-feature-v9] skip ${feature_mode} (${summary_path})"
    continue
  fi

  cmd=(
    "$PY" "$ROOT/scripts/train_candidate_conditioned_ranker_v7.py"
    --bank-tag "$BANK_TAG"
    --bank-size "$BANK_SIZE"
    --train-tags "${TRAIN_TAGS[@]}"
    --eval-tags "${EVAL_TAGS[@]}"
    --hidden-train-tags "${HIDDEN_TRAIN_TAGS[@]}"
    --hidden-eval-tags "${HIDDEN_EVAL_TAGS[@]}"
    --output-tag "$OUTPUT_TAG"
    --feature-mode "$feature_mode"
    --skip-counts "${SKIP_COUNTS[@]}"
    --hidden-pca-dim "$HIDDEN_PCA_DIM"
    --stp-pca-dim "$STP_PCA_DIM"
    --output-subdir "$OUTPUT_SUBDIR"
    --plot-prefix "$PLOT_PREFIX"
  )

  if [[ "$FAST_MODE" == "1" ]]; then
    cmd+=(--fast-mode)
  fi

  echo "[stp-feature-v9] run feature_mode=${feature_mode} output=${OUTPUT_TAG}"
  "${cmd[@]}"
done

echo "[stp-feature-v9] complete"
