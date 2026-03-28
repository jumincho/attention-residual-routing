#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", type=str, nargs="+", required=True)
    parser.add_argument("--output-tag", type=str, required=True)
    args = parser.parse_args()

    base_dir = ROOT / "results" / "rich_features"
    frames = [pd.read_csv(base_dir / f"{tag}_hidden_prompt_features.csv") for tag in args.tags]
    merged = pd.concat(frames, ignore_index=True).sort_values(["split", "document_idx", "window_idx", "sequence_idx"])
    merged.to_csv(base_dir / f"{args.output_tag}_hidden_prompt_features.csv", index=False)
    print(f"[merge_hidden] output_tag={args.output_tag} rows={len(merged)}", flush=True)


if __name__ == "__main__":
    main()
