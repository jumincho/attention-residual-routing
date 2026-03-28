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


def summarize(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str], seed: int) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = {col: value for col, value in zip(group_cols, keys)}
        for metric in metric_cols:
            ci = bootstrap_mean_ci(group[metric].to_numpy(dtype=float), seed=seed)
            rows.append({**key_map, "metric": metric, **ci})
    return pd.DataFrame(rows)


def maybe_concat(base_dir: Path, tags: list[str], suffix: str) -> pd.DataFrame | None:
    paths = [base_dir / f"{tag}_{suffix}.csv" for tag in tags]
    if not all(path.exists() for path in paths):
        return None
    return concat_tables(base_dir, tags, suffix)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", type=str, nargs="+", required=True)
    parser.add_argument("--output-tag", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    oracles_dir = ROOT / "results" / "oracles"
    routing_dir = ROOT / "results" / "routing"

    loo_df = concat_tables(oracles_dir, args.tags, "leave_one_out_alignment")
    feature_df = concat_tables(oracles_dir, args.tags, "sequence_features")
    mask_df = concat_tables(oracles_dir, args.tags, "exhaustive_mask_losses")
    oracle_df = concat_tables(oracles_dir, args.tags, "oracle_mask_alignment")
    routing_df = maybe_concat(routing_dir, args.tags, "routing_eval_per_sequence")

    loo_summary = summarize(
        loo_df.assign(split="all"),
        group_cols=["split"],
        metric_cols=["spearman", "kendall", "topk_jaccard_1", "recall_at_3", "ndcg_at_3"],
        seed=args.seed,
    ).drop(columns=["split"])
    oracle_summary = summarize(
        oracle_df,
        group_cols=["skip_count"],
        metric_cols=[
            "oracle_exact_match",
            "oracle_mask_jaccard",
            "delta_to_oracle",
            "delta_to_global_static",
            "stability_spearman",
            "stability_top3_jaccard",
            "prompt_margin",
        ],
        seed=args.seed,
    )

    loo_df.to_csv(oracles_dir / f"{args.output_tag}_leave_one_out_alignment.csv", index=False)
    loo_summary.to_csv(oracles_dir / f"{args.output_tag}_leave_one_out_alignment_summary.csv", index=False)
    feature_df.to_csv(oracles_dir / f"{args.output_tag}_sequence_features.csv", index=False)
    mask_df.to_csv(oracles_dir / f"{args.output_tag}_exhaustive_mask_losses.csv", index=False)
    oracle_df.to_csv(oracles_dir / f"{args.output_tag}_oracle_mask_alignment.csv", index=False)
    oracle_summary.to_csv(oracles_dir / f"{args.output_tag}_oracle_mask_alignment_summary.csv", index=False)

    if routing_df is not None:
        routing_summary = summarize(
            routing_df,
            group_cols=["skip_count", "method"],
            metric_cols=[
                "continuation_loss",
                "decode_tokens_per_sec",
                "prefill_seconds",
                "decode_seconds",
                "routing_overhead_seconds",
                "end_to_end_seconds",
                "active_blocks",
            ],
            seed=args.seed,
        )
        routing_df.to_csv(routing_dir / f"{args.output_tag}_routing_eval_per_sequence.csv", index=False)
        routing_summary.to_csv(routing_dir / f"{args.output_tag}_routing_eval_summary.csv", index=False)

    print(
        f"[merge] output_tag={args.output_tag} "
        f"rows={{'loo': {len(loo_df)}, 'features': {len(feature_df)}, 'mask': {len(mask_df)}, 'oracle': {len(oracle_df)}}}",
        flush=True,
    )


if __name__ == "__main__":
    main()
