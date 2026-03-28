#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/torchrun}"
TEMPLATE_CONFIG="${TEMPLATE_CONFIG:-$ROOT/configs/scale_heterogeneity_v9/attnres_24x512_ccnews_stp_continuation_template.yaml}"
SEED="${SEED:-44}"
RESUME_STEP="${RESUME_STEP:-3000}"
RESUME_FROM="${RESUME_FROM:-$ROOT/results/scale24x512_ccnews_attnres_seed${SEED}_v8/checkpoint_step_$(printf '%06d' "$RESUME_STEP").pt}"
MAX_STEPS="${MAX_STEPS:-4500}"
SAVE_STEPS_RAW="${SAVE_STEPS:-3500 4000 4500}"
LAMBDAS_RAW="${LAMBDAS:-0.0 0.005 0.01 0.02 0.04}"
STP_TRIPLETS="${STP_TRIPLETS:-2}"
CONFIG_OUT_DIR="${CONFIG_OUT_DIR:-$ROOT/results/attnres_stp_v9/generated_configs}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-scale24x512_ccnews_attnres_stpcont_seed${SEED}_from${RESUME_STEP}_v9}"

mkdir -p "$CONFIG_OUT_DIR"

save_steps_yaml=""
for step in $SAVE_STEPS_RAW; do
  if [[ -n "$save_steps_yaml" ]]; then
    save_steps_yaml="${save_steps_yaml}, "
  fi
  save_steps_yaml="${save_steps_yaml}${step}"
done

lambda_tag() {
  local lambda_value="$1"
  if [[ "$lambda_value" == "0.0" || "$lambda_value" == "0" ]]; then
    echo "plain"
    return
  fi
  echo "lam$(echo "$lambda_value" | tr '.' 'p')"
}

for lambda_value in $LAMBDAS_RAW; do
  tag="$(lambda_tag "$lambda_value")"
  experiment_name="${EXPERIMENT_PREFIX}_${tag}"
  config_path="$CONFIG_OUT_DIR/${experiment_name}.yaml"

  sed \
    -e "s#__EXPERIMENT_NAME__#${experiment_name}#g" \
    -e "s#__SEED__#${SEED}#g" \
    -e "s#__MAX_STEPS__#${MAX_STEPS}#g" \
    -e "s#__SAVE_STEPS__#${save_steps_yaml}#g" \
    -e "s#__RESUME_FROM__#${RESUME_FROM}#g" \
    -e "s#__STP_WEIGHT__#${lambda_value}#g" \
    -e "s#__STP_TRIPLETS__#${STP_TRIPLETS}#g" \
    "$TEMPLATE_CONFIG" >"$config_path"

  echo "[attnres-stp-v9] launch experiment=${experiment_name} lambda=${lambda_value}"
  "$PYTHON_BIN" --standalone --nproc_per_node=4 scripts/train_lm.py --config "$config_path"
done

echo "[attnres-stp-v9] complete"
