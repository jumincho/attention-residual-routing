#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def merge_family(
    base_dir: Path,
    tags: list[str],
    output_tag: str,
    suffix: str,
    dedupe_on: list[str] | None = None,
) -> None:
    frames = [pd.read_csv(base_dir / f"{tag}_{suffix}.csv") for tag in tags]
    merged = pd.concat(frames, ignore_index=True)
    if dedupe_on:
        merged = merged.drop_duplicates(subset=dedupe_on)
    merged.to_csv(base_dir / f"{output_tag}_{suffix}.csv", index=False)
    print(f"[merge_sublayer] suffix={suffix} rows={len(merged)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", type=str, nargs="+", required=True)
    parser.add_argument("--output-tag", type=str, required=True)
    args = parser.parse_args()

    base_dir = ROOT / "results" / "latency_budgeted_sublayer_v5"
    merge_family(base_dir, args.tags, args.output_tag, "candidate_losses")
    merge_family(base_dir, args.tags, args.output_tag, "candidate_summary", dedupe_on=["candidate_id"])
    merge_family(base_dir, args.tags, args.output_tag, "candidate_defs", dedupe_on=["candidate_id"])


if __name__ == "__main__":
    main()
