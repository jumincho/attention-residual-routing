#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attnres_routing.analysis import bootstrap_mean_ci  # noqa: E402


def concat_tables(base_dir: Path, tags: list[str], suffix: str) -> pd.DataFrame:
    frames = [pd.read_csv(base_dir / f"{tag}_{suffix}.csv") for tag in tags]
    return pd.concat(frames, ignore_index=True)


def summarize(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []
    metric_cols = [
        "stability_spearman",
        "stability_top3_jaccard",
        "prompt_margin",
        "prompt_depth_entropy",
        "prompt_support_size",
    ]
    for skip_count, group in df.groupby("skip_count", sort=False):
        for metric in metric_cols:
            ci = bootstrap_mean_ci(group[metric].to_numpy(dtype=float), seed=seed)
            rows.append({"skip_count": skip_count, "metric": metric, **ci})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", type=str, nargs="+", required=True)
    parser.add_argument("--output-tag", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_dir = ROOT / "results" / "oracles"
    feature_df = concat_tables(base_dir, args.tags, "sequence_features")
    oracle_df = concat_tables(base_dir, args.tags, "oracle_mask_alignment")
    oracle_summary = summarize(oracle_df, seed=args.seed)

    feature_df.to_csv(base_dir / f"{args.output_tag}_sequence_features.csv", index=False)
    oracle_df.to_csv(base_dir / f"{args.output_tag}_oracle_mask_alignment.csv", index=False)
    oracle_summary.to_csv(base_dir / f"{args.output_tag}_oracle_mask_alignment_summary.csv", index=False)
    print(
        f"[merge_prompt_features] output_tag={args.output_tag} feature_rows={len(feature_df)} oracle_rows={len(oracle_df)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
