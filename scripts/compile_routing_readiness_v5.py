#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def zscore(values: np.ndarray, larger_is_better: bool = True) -> np.ndarray:
    values = values.astype(np.float64)
    if not larger_is_better:
        values = -values
    mean = values.mean()
    std = values.std()
    if std < 1e-8:
        return np.zeros_like(values)
    return (values - mean) / std


def step_from_tag(tag: str) -> int:
    match = re.search(r"step(\d+)", tag)
    if not match:
        raise ValueError(f"Cannot parse step from tag: {tag}")
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", type=str, nargs="+", required=True)
    parser.add_argument("--experiment-dir", type=str, required=True)
    parser.add_argument("--output-tag", type=str, required=True)
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    oracles_dir = ROOT / "results" / "oracles"
    out_dir = ROOT / "results" / "routing_checkpoint_selection_v5"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.read_csv(experiment_dir / "metrics.csv")
    rows = []
    for tag in args.tags:
        step = step_from_tag(tag)
        loo_path = oracles_dir / f"{tag}_leave_one_out_alignment_summary.csv"
        oracle_path = oracles_dir / f"{tag}_oracle_mask_alignment_summary.csv"
        if not loo_path.exists() or not oracle_path.exists():
            continue
        loo_df = pd.read_csv(loo_path)
        oracle_df = pd.read_csv(oracle_path)
        metric_row = metrics_df[metrics_df["step"] == step].iloc[-1]
        val_loss = float(metric_row["val_loss"]) if pd.notna(metric_row["val_loss"]) and metric_row["val_loss"] != "" else np.nan
        row = {
            "tag": tag,
            "step": step,
            "val_loss": val_loss,
            "loo_spearman": float(loo_df[loo_df["metric"] == "spearman"]["mean"].iloc[0]),
            "loo_kendall": float(loo_df[loo_df["metric"] == "kendall"]["mean"].iloc[0]),
        }
        for skip_count in sorted(oracle_df["skip_count"].unique().tolist()):
            subset = oracle_df[oracle_df["skip_count"] == skip_count]
            for metric_name in ["delta_to_global_static", "stability_spearman", "oracle_exact_match", "prompt_margin"]:
                metric_subset = subset[subset["metric"] == metric_name]
                if metric_subset.empty:
                    continue
                row[f"skip{skip_count}_{metric_name}"] = float(metric_subset["mean"].iloc[0])
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
    if df.empty:
        raise ValueError("No routing checkpoint rows found.")

    components = {
        "loo_spearman_z": zscore(df["loo_spearman"].to_numpy(), larger_is_better=True),
        "loo_kendall_z": zscore(df["loo_kendall"].to_numpy(), larger_is_better=True),
        "val_loss_z": zscore(df["val_loss"].fillna(df["val_loss"].max()).to_numpy(), larger_is_better=False),
    }
    for key, values in components.items():
        df[key] = values
    df["routing_readiness_score"] = df["loo_spearman_z"] + df["loo_kendall_z"] + 0.5 * df["val_loss_z"]
    df.to_csv(out_dir / f"{args.output_tag}_routing_readiness.csv", index=False)
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
