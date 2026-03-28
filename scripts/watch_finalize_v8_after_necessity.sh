#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid2/chojm/attnres-routing-research"
PID="${1:?usage: watch_finalize_v8_after_necessity.sh <necessity-pid>}"

while kill -0 "$PID" 2>/dev/null; do
  sleep 20
done

source "$ROOT/.venv/bin/activate"

python "$ROOT/scripts/aggregate_ccnews_necessity_v8.py" \
  --winners-csv "$ROOT/results/ccnews_multiseed_multisplit_v8/v8_ccnews_dev_frozen_selection_winners.csv" \
  --final-splits final_A final_B final_C \
  --output-dir "$ROOT/results/ccnews_necessity_multiseed_v8"

python "$ROOT/scripts/aggregate_ccnews_systems_v8.py" \
  --winners-csv "$ROOT/results/ccnews_multiseed_multisplit_v8/v8_ccnews_dev_frozen_selection_winners.csv" \
  --final-splits final_A final_B final_C \
  --template-limits 0 2 4 \
  --output-dir "$ROOT/results/systems_speedup_v8"

python "$ROOT/scripts/compile_summary_v8.py" \
  --repo-root "$ROOT" \
  --output "$ROOT/results/summary_v8.csv"

python "$ROOT/scripts/write_v8_docs.py" \
  --repo-root "$ROOT"
