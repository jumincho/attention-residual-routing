#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
LOG_DIR="$ROOT/results/selector_data_scale/logs"
mkdir -p "$LOG_DIR"

log() {
  printf '[resume-v7] %s %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" | tee -a "$LOG_DIR/resume_v7_pipeline.log"
}

wait_until_no_match() {
  local pattern="$1"
  while pgrep -f "$pattern" >/dev/null 2>&1; do
    sleep 15
  done
}

run_direct_oracle_group() {
  local checkpoint="$1"
  local manifest="$2"
  local tag_prefix="$3"
  local total_shards="$4"
  local prompt_len="$5"
  local decode_len="$6"

  if [[ -f "$ROOT/results/oracles/${tag_prefix}_oracle_mask_alignment_summary.csv" ]]; then
    log "skip existing oracle ${tag_prefix}"
    return
  fi

  log "direct oracle start ${tag_prefix}"
  local pids=()
  local shard
  for ((shard=0; shard<total_shards; shard++)); do
    local gpu=$(( shard % 4 ))
    local tag
    tag="$(printf '%s_s%02d' "$tag_prefix" "$shard")"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT/src" "$PY" scripts/evaluate_functional_oracles.py \
      --checkpoint "$checkpoint" \
      --manifest-path "$manifest" \
      --prompt-len "$prompt_len" \
      --decode-len "$decode_len" \
      --num-chunks 4 \
      --batch-size 64 \
      --score-mode utility_over_variance \
      --skip-counts 1 2 3 \
      --num-sequences -1 \
      --num-shards "$total_shards" \
      --shard-index "$shard" \
      --tag "$tag" \
      --skip-plots \
      --skip-docs \
      >"$LOG_DIR/${tag}.log" 2>&1 &
    pids+=("$!")
  done

  local fail=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      fail=1
    fi
  done
  if [[ "$fail" -ne 0 ]]; then
    log "direct oracle failed ${tag_prefix}"
    return 1
  fi

  local shard_tags=()
  for ((shard=0; shard<total_shards; shard++)); do
    shard_tags+=("$(printf '%s_s%02d' "$tag_prefix" "$shard")")
  done
  "$PY" scripts/merge_oracle_shards.py --tags "${shard_tags[@]}" --output-tag "$tag_prefix" >>"$LOG_DIR/resume_v7_pipeline.log" 2>&1
  log "direct oracle complete ${tag_prefix}"
}

run_third_corpus_eval_direct() {
  local exp_dir="$ROOT/results/scale24x512_finewebedu_sample10bt_attnres_v7"
  local lockbox_tag="v7_fineweb_p256d64_lockbox"
  local train_manifest="$ROOT/results/lockbox_manifests_v7/${lockbox_tag}_train.jsonl"
  local val_manifest="$ROOT/results/lockbox_manifests_v7/${lockbox_tag}_validation.jsonl"
  local dev_manifest="$ROOT/results/lockbox_manifests_v7/${lockbox_tag}_dev_test.jsonl"

  # Ensure step2000 dev is complete even if the wrapper failed earlier.
  run_direct_oracle_group \
    "$exp_dir/checkpoint_step_002000.pt" \
    "$dev_manifest" \
    "${lockbox_tag}_step2000_dev" \
    4 256 64

  if [[ ! -f "$ROOT/results/bank_hygiene/${lockbox_tag}_step2000_bank_summary.csv" ]]; then
    "$PY" scripts/evaluate_train_only_bank_hygiene.py \
      --calib-tags "${lockbox_tag}_step2000_val" \
      --eval-tags "${lockbox_tag}_step2000_dev" \
      --output-tag "${lockbox_tag}_step2000_bank" \
      --bank-sizes 32 >>"$LOG_DIR/resume_v7_pipeline.log" 2>&1
    log "bank hygiene complete ${lockbox_tag}_step2000_bank"
  fi

  if [[ ! -f "$ROOT/results/oracles/${lockbox_tag}_step3000_val_oracle_mask_alignment_summary.csv" ]]; then
    run_direct_oracle_group \
      "$exp_dir/checkpoint_step_003000.pt" \
      "$val_manifest" \
      "${lockbox_tag}_step3000_val" \
      4 256 64
  fi
  if [[ ! -f "$ROOT/results/oracles/${lockbox_tag}_step3000_dev_oracle_mask_alignment_summary.csv" ]]; then
    run_direct_oracle_group \
      "$exp_dir/checkpoint_step_003000.pt" \
      "$dev_manifest" \
      "${lockbox_tag}_step3000_dev" \
      4 256 64
  fi
  if [[ ! -f "$ROOT/results/oracles/${lockbox_tag}_step3000_train4096_oracle_mask_alignment_summary.csv" ]]; then
    run_direct_oracle_group \
      "$exp_dir/checkpoint_step_003000.pt" \
      "$train_manifest" \
      "${lockbox_tag}_step3000_train4096" \
      8 256 64
  fi

  if [[ ! -f "$ROOT/results/bank_hygiene/${lockbox_tag}_step3000_bank_summary.csv" ]]; then
    "$PY" scripts/evaluate_train_only_bank_hygiene.py \
      --calib-tags "${lockbox_tag}_step3000_val" \
      --eval-tags "${lockbox_tag}_step3000_dev" \
      --output-tag "${lockbox_tag}_step3000_bank" \
      --bank-sizes 32 >>"$LOG_DIR/resume_v7_pipeline.log" 2>&1
    log "bank hygiene complete ${lockbox_tag}_step3000_bank"
  fi

  for step in 1000 2000 3000; do
    local train_tag="${lockbox_tag}_step${step}_train4096"
    local eval_tag="${lockbox_tag}_step${step}_dev"
    local bank_tag="${lockbox_tag}_step${step}_bank"
    local out_tag="v7_fineweb_step${step}_dev1024_b32"
    if [[ ! -f "$ROOT/results/regret_reduction_v7/${out_tag}_attnres_summary.csv" ]]; then
      "$PY" scripts/train_candidate_conditioned_ranker_v7.py \
        --bank-tag "$bank_tag" \
        --bank-size 32 \
        --train-tags "$train_tag" \
        --eval-tags "$eval_tag" \
        --output-tag "$out_tag" \
        --feature-mode attnres \
        --fast-mode >>"$LOG_DIR/resume_v7_pipeline.log" 2>&1
      log "fineweb fast selector complete ${out_tag}"
    fi
  done

  if [[ ! -f "$ROOT/results/readiness_v7/v7_fineweb_main_routing_readiness_v3.csv" ]]; then
    "$PY" scripts/compile_routing_readiness_v7.py \
      --tags \
      "${lockbox_tag}_step1000_val" \
      "${lockbox_tag}_step2000_val" \
      "${lockbox_tag}_step3000_val" \
      --experiment-dir "$exp_dir" \
      --bank-size 32 \
      --bank-skip 1 \
      --feature-mode attnres \
      --selector-prefix 'v7_fineweb_step{step}_dev1024_b32' \
      --selector-dir results/regret_reduction_v7 \
      --output-tag v7_fineweb_main >>"$LOG_DIR/resume_v7_pipeline.log" 2>&1
    log "fineweb readiness complete"
  fi
}

run_systems_if_missing() {
  if [[ ! -f "$ROOT/results/systems_routing_v7/v7_ccnews_step3000_skip1_rf_pair_tpl8_summary.csv" ]]; then
    bash "$ROOT/scripts/run_systems_routing_v7.sh" >>"$LOG_DIR/resume_v7_pipeline.log" 2>&1
    log "systems routing complete"
  fi
}

log "watcher started"

# Wait for any currently running fineweb shard jobs to finish first.
wait_until_no_match 'v7_fineweb_p256d64_lockbox_step2000_dev_s0[0-3]'
wait_until_no_match 'v7_fineweb_p256d64_lockbox_step3000_'

run_third_corpus_eval_direct
run_systems_if_missing

log "watcher finished"
