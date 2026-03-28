#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attnres_routing.train import train_experiment
from attnres_routing.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    if not config:
        raise ValueError(f"Empty config: {args.config}")
    train_experiment(config)


if __name__ == "__main__":
    main()
