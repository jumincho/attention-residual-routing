#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT / "src"))
from attnres_routing.analysis import bootstrap_mean_ci  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-csv", required=True)
    parser.add_argument("--final-splits", nargs="+", default=["final_A", "final_B", "final_C"])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    winners = pd.read_csv(args.selection_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_seed_rows = []
    pooled_seq_rows = []
    deploy_rows = []

    for _, row in winners.iterrows():
        seed = int(row["seed"])
        step = int(row["step"])
        bank_size = int(row["bank_size"])
        feature_mode = str(row["feature_mode"])
        model_name = str(row["model_name"])
        for split in args.final_splits:
            prefix = f"v8_locked_seed{seed}_{split}_step{step}_b{bank_size}_{model_name}_{feature_mode}"
            summary_path = ROOT / "results" / "regret_reduction_v8" / f"{prefix}_{feature_mode}_summary.csv"
            per_seq_path = ROOT / "results" / "regret_reduction_v8" / f"{prefix}_{feature_mode}_per_sequence.csv"
            deploy_path = ROOT / "results" / "systems_routing_v7" / f"{prefix}_deploy_summary.csv"
            if not summary_path.exists() or not per_seq_path.exists():
                continue
            summary_df = pd.read_csv(summary_path)
            selected = summary_df[
                (summary_df["feature_mode"] == feature_mode)
                & (summary_df["skip_count"] == 1)
                & (summary_df["bank_size"] == bank_size)
                & (summary_df["model_name"] == model_name)
            ].copy()
            if selected.empty:
                continue
            delta_row = selected[selected["metric"] == "delta_to_static"].iloc[0]
            regret_row = selected[selected["metric"] == "delta_to_bank_upper_bound"].iloc[0]
            frac_row = selected[selected["metric"] == "fraction_improved"].iloc[0]
            top1_row = selected[selected["metric"] == "oracle_in_bank_match"].iloc[0]
            per_seed_rows.append(
                {
                    "seed": seed,
                    "final_split": split,
                    "step": step,
                    "bank_size": bank_size,
                    "feature_mode": feature_mode,
                    "model_name": model_name,
                    "delta_to_static": float(delta_row["mean"]),
                    "delta_ci_low": float(delta_row["ci_low"]),
                    "delta_ci_high": float(delta_row["ci_high"]),
                    "regret_to_bank": float(regret_row["mean"]),
                    "fraction_improved": float(frac_row["mean"]),
                    "oracle_in_bank_match": float(top1_row["mean"]),
                    "n": int(delta_row["n"]),
                }
            )
            per_seq_df = pd.read_csv(per_seq_path)
            per_seq_df = per_seq_df[
                (per_seq_df["feature_mode"] == feature_mode)
                & (per_seq_df["skip_count"] == 1)
                & (per_seq_df["bank_size"] == bank_size)
                & (per_seq_df["model_name"] == model_name)
            ].copy()
            per_seq_df["seed"] = seed
            per_seq_df["final_split"] = split
            per_seq_df["step"] = step
            pooled_seq_rows.append(per_seq_df)
            if deploy_path.exists():
                deploy_df = pd.read_csv(deploy_path)
                deploy_df["seed"] = seed
                deploy_df["final_split"] = split
                deploy_df["step"] = step
                deploy_rows.append(deploy_df)

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_df.to_csv(output_dir / "ccnews_v8_locked_per_seed_split.csv", index=False)

    pooled_df = pd.concat(pooled_seq_rows, ignore_index=True) if pooled_seq_rows else pd.DataFrame()
    if not pooled_df.empty:
        delta = bootstrap_mean_ci(pooled_df["actual_delta_to_static"].to_numpy())
        regret = bootstrap_mean_ci(pooled_df["delta_to_bank_upper_bound"].to_numpy())
        frac = bootstrap_mean_ci(pooled_df["improved_over_static"].to_numpy())
        pooled_summary = pd.DataFrame(
            [
                {
                    "metric": "pooled_delta_to_static",
                    "mean": float(delta["mean"]),
                    "ci_low": float(delta["ci_low"]),
                    "ci_high": float(delta["ci_high"]),
                    "n": int(delta["n"]),
                },
                {
                    "metric": "pooled_regret_to_bank",
                    "mean": float(regret["mean"]),
                    "ci_low": float(regret["ci_low"]),
                    "ci_high": float(regret["ci_high"]),
                    "n": int(regret["n"]),
                },
                {
                    "metric": "pooled_fraction_improved",
                    "mean": float(frac["mean"]),
                    "ci_low": float(frac["ci_low"]),
                    "ci_high": float(frac["ci_high"]),
                    "n": int(frac["n"]),
                },
            ]
        )
        pooled_summary.to_csv(output_dir / "ccnews_v8_locked_pooled_summary.csv", index=False)
        pooled_df.to_csv(output_dir / "ccnews_v8_locked_pooled_per_sequence.csv", index=False)

    if deploy_rows:
        deploy_df = pd.concat(deploy_rows, ignore_index=True)
        deploy_df.to_csv(output_dir / "ccnews_v8_locked_deployment_rows.csv", index=False)

    print(per_seed_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
