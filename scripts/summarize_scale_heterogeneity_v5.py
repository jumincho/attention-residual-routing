#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def step_from_name(text: str) -> int:
    match = re.search(r"step(\d+)", text)
    if not match:
        raise ValueError(f"Could not parse step from {text}")
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default=str(ROOT / "results" / "scale24x512_ccnews_attnres_v5"),
    )
    parser.add_argument(
        "--readiness-csv",
        type=str,
        default=str(ROOT / "results" / "routing_checkpoint_selection_v5" / "v5_ccnews_attnres_routing_readiness.csv"),
    )
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    readiness_csv = Path(args.readiness_csv)
    out_dir = ROOT / "results" / "scale_heterogeneity_v5"
    plot_dir = ROOT / "results" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(experiment_dir / "metrics.csv")
    trajectory = metrics[metrics["val_loss"].notna()].copy()
    trajectory.to_csv(out_dir / "trajectory.csv", index=False)

    plt.figure(figsize=(6, 4))
    plt.plot(trajectory["step"], trajectory["val_loss"], marker="o")
    plt.xlabel("Step")
    plt.ylabel("Val Loss")
    plt.title("Scale Heterogeneity V5 Val Loss")
    plt.tight_layout()
    plt.savefig(plot_dir / "scale_heterogeneity_v5_val_loss.png", dpi=160)
    plt.close()

    if readiness_csv.exists():
        readiness = pd.read_csv(readiness_csv).sort_values("step").reset_index(drop=True)
        readiness.to_csv(out_dir / "checkpoint_readiness.csv", index=False)
        plt.figure(figsize=(6, 4))
        plt.plot(readiness["step"], readiness["routing_readiness_score"], marker="o", label="routing_readiness")
        if "val_loss" in readiness.columns:
            val_scaled = (readiness["val_loss"] - readiness["val_loss"].mean()) / max(readiness["val_loss"].std(), 1e-8)
            plt.plot(readiness["step"], -val_scaled, marker="s", label="-z(val_loss)")
        plt.xlabel("Step")
        plt.ylabel("Score")
        plt.title("Scale Heterogeneity V5 Routing Readiness")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "scale_heterogeneity_v5_routing_readiness.png", dpi=160)
        plt.close()

    bank_rows = []
    for path in sorted((ROOT / "results" / "bank_hygiene").glob("v5_ccnews_step*_bank_test_summary.csv")):
        step = step_from_name(path.name)
        df = pd.read_csv(path)
        df["step"] = step
        bank_rows.append(df)
    if bank_rows:
        bank_df = pd.concat(bank_rows, ignore_index=True)
        bank_df.to_csv(out_dir / "bank_summary.csv", index=False)
        subset = bank_df[
            (bank_df["metric"] == "delta_bank_to_calib_global")
            & (bank_df["bank_size"] == 32)
        ].copy()
        if not subset.empty:
            plt.figure(figsize=(6, 4))
            for skip_count in sorted(subset["skip_count"].unique().tolist()):
                group = subset[subset["skip_count"] == skip_count].sort_values("step")
                plt.plot(group["step"], group["mean"], marker="o", label=f"skip {skip_count}")
            plt.xlabel("Step")
            plt.ylabel("Delta To Calib Global")
            plt.title("Scale Heterogeneity V5 Bank Upper Bound")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_dir / "scale_heterogeneity_v5_bank_upper_bound.png", dpi=160)
            plt.close()

    ranker_rows = []
    for path in sorted((ROOT / "results" / "ranker_v5").glob("v5_ccnews_step*_test512_b32_attnres_summary.csv")):
        step = step_from_name(path.name)
        df = pd.read_csv(path)
        df["step"] = step
        ranker_rows.append(df)
    if ranker_rows:
        ranker_df = pd.concat(ranker_rows, ignore_index=True)
        ranker_df.to_csv(out_dir / "ranker_summary.csv", index=False)
        subset = ranker_df[(ranker_df["metric"] == "delta_to_static")].copy()
        if not subset.empty:
            plt.figure(figsize=(6, 4))
            for skip_count in sorted(subset["skip_count"].unique().tolist()):
                group = subset[subset["skip_count"] == skip_count].sort_values("step")
                best_by_step = group.groupby("step", as_index=False)["mean"].min()
                plt.plot(best_by_step["step"], best_by_step["mean"], marker="o", label=f"skip {skip_count}")
            plt.xlabel("Step")
            plt.ylabel("Best Ranker Delta To Static")
            plt.title("Scale Heterogeneity V5 Selector")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_dir / "scale_heterogeneity_v5_ranker_delta.png", dpi=160)
            plt.close()


if __name__ == "__main__":
    main()
