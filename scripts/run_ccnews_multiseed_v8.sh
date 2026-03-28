#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

SEEDS_RAW="${SEEDS:-43 44}"
read -r -a SEEDS <<<"$SEEDS_RAW"

for seed in "${SEEDS[@]}"; do
  config_path="$ROOT/configs/scale_heterogeneity_v8/attnres_24x512_ccnews_seed${seed}.yaml"
  out_dir="$ROOT/results/scale24x512_ccnews_attnres_seed${seed}_v8"
  if [[ -f "$out_dir/checkpoint_step_006000.pt" ]]; then
    echo "[ccnews-multiseed-v8] skip completed seed ${seed}"
    continue
  fi
  echo "[ccnews-multiseed-v8] launch seed ${seed}"
  CONFIG_PATH="$config_path" CUDA_VISIBLE_DEVICES=0,1,2,3 "$ROOT/scripts/run_ccnews_seed_v8.sh"
done

echo "[ccnews-multiseed-v8] complete"
