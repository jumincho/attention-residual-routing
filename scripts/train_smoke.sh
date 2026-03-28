#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source .venv/bin/activate

PYTHONPATH=src python scripts/train_lm.py --config configs/smoke_tinystories_standard.yaml
PYTHONPATH=src python scripts/train_lm.py --config configs/smoke_tinystories_attnres.yaml
