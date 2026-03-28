#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
PROMPT_LEN="${PROMPT_LEN:-256}"
DECODE_LEN="${DECODE_LEN:-64}"
DEV_TOTAL_SHARDS="${DEV_TOTAL_SHARDS:-16}"
VAL_TOTAL_SHARDS="${VAL_TOTAL_SHARDS:-4}"
GPUS_RAW="${GPUS:-0 1 2 3}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

read -r -a GPUS <<<"$GPUS_RAW"

checkpoint_path() {
  local seed="$1"
  local role="$2"
  local step
  local base
  case "$seed:$role" in
    42:route) step=5500; base="$ROOT/results/scale24x512_ccnews_attnres_dense_v7" ;;
    42:lm)    step=3000; base="$ROOT/results/scale24x512_ccnews_attnres_dense_v7" ;;
    43:route) step=6000; base="$ROOT/results/scale24x512_ccnews_attnres_seed43_v8" ;;
    43:lm)    step=3000; base="$ROOT/results/scale24x512_ccnews_attnres_seed43_v8" ;;
    44:route) step=3500; base="$ROOT/results/scale24x512_ccnews_attnres_seed44_v8" ;;
    44:lm)    step=3000; base="$ROOT/results/scale24x512_ccnews_attnres_seed44_v8" ;;
    *)
      echo "unsupported seed/role: $seed $role" >&2
      exit 1
      ;;
  esac
  printf "%s/checkpoint_step_%06d.pt\n" "$base" "$step"
}

tag_base() {
  local seed="$1"
  local role="$2"
  local split_tag="$3"
  local step
  case "$seed:$role" in
    42:route) step=5500 ;;
    42:lm)    step=3000 ;;
    43:route) step=6000 ;;
    43:lm)    step=3000 ;;
    44:route) step=3500 ;;
    44:lm)    step=3000 ;;
    *)
      echo "unsupported seed/role: $seed $role" >&2
      exit 1
      ;;
  esac
  echo "v9_stp_seed${seed}_${role}_step${step}_${split_tag}"
}

run_hidden_group() {
  local checkpoint="$1"
  local manifest_path="$2"
  local split_tag="$3"
  local total_shards="$4"
  local output_tag="$5"

  if [[ "$SKIP_EXISTING" == "1" && -f "$ROOT/results/rich_features/${output_tag}_hidden_prompt_features.csv" ]]; then
    echo "[stp-hidden-v9] skip ${output_tag} (exists)"
    return
  fi

  echo "[stp-hidden-v9] extract ${output_tag}"
  "$PY" "$ROOT/scripts/run_sharded_hidden_extract.py" \
    --checkpoint "$checkpoint" \
    --manifest-path "$manifest_path" \
    --prompt-len "$PROMPT_LEN" \
    --decode-len "$DECODE_LEN" \
    --tag-prefix "$output_tag" \
    --total-shards "$total_shards" \
    --gpus "${GPUS[@]}" \
    --python-bin "$PY"

  mapfile -t shard_tags < <(seq -f "${output_tag}_s%02g" 0 $((total_shards - 1)))
  "$PY" "$ROOT/scripts/merge_hidden_shards.py" --tags "${shard_tags[@]}" --output-tag "$output_tag"
}

for seed in 42 43 44; do
  for role in route lm; do
    checkpoint="$(checkpoint_path "$seed" "$role")"
    dev_tag="$(tag_base "$seed" "$role" "devselect2048")"
    val_tag="$(tag_base "$seed" "$role" "validation512")"

    run_hidden_group \
      "$checkpoint" \
      "$ROOT/results/lockbox_manifests_v8/v8_ccnews_p256d64_lockbox_dev_select2048.jsonl" \
      "devselect2048" \
      "$DEV_TOTAL_SHARDS" \
      "$dev_tag"

    run_hidden_group \
      "$checkpoint" \
      "$ROOT/results/lockbox_manifests_v8/v8_ccnews_p256d64_lockbox_validation512.jsonl" \
      "validation512" \
      "$VAL_TOTAL_SHARDS" \
      "$val_tag"
  done
done

echo "[stp-hidden-v9] complete"
