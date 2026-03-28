#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
cd "$ROOT"

EXPERIMENT_DIR="${EXPERIMENT_DIR:?set EXPERIMENT_DIR=/abs/path/to/experiment_dir}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-2500 3000 3500 4000 4500 5000 5500 6000}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-$ROOT/results/lockbox_manifests_v8/v8_ccnews_p256d64_lockbox_train512.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-$ROOT/results/lockbox_manifests_v8/v8_ccnews_p256d64_lockbox_validation512.jsonl}"
DEV_MANIFEST="${DEV_MANIFEST:-$ROOT/results/lockbox_manifests_v8/v8_ccnews_p256d64_lockbox_dev_select512.jsonl}"
TAG_PREFIX="${TAG_PREFIX:-v8_ccnews_fast}"
BANK_SIZES="${BANK_SIZES:-32}"
PROMPT_LEN="${PROMPT_LEN:-256}"
DECODE_LEN="${DECODE_LEN:-64}"
SUMMARY_DIR="${SUMMARY_DIR:-$ROOT/results/ccnews_multiseed_multisplit_v8}"
BATCH_SIZE="${BATCH_SIZE:-128}"

EXPERIMENT_DIR="$EXPERIMENT_DIR" \
CHECKPOINT_STEPS="$CHECKPOINT_STEPS" \
TRAIN_MANIFEST="$TRAIN_MANIFEST" \
VAL_MANIFEST="$VAL_MANIFEST" \
DEV_MANIFEST="$DEV_MANIFEST" \
TAG_PREFIX="$TAG_PREFIX" \
PROMPT_LEN="$PROMPT_LEN" \
DECODE_LEN="$DECODE_LEN" \
BANK_SIZES="$BANK_SIZES" \
SUMMARY_DIR="$SUMMARY_DIR" \
BATCH_SIZE="$BATCH_SIZE" \
bash "$ROOT/scripts/run_hetero_bank_audit_v7.sh"

echo "[ccnews-fast-bank-audit-v8] complete"
