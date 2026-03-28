#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_metric_table(path: Path, extra: dict[str, str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    for key, value in extra.items():
        df[key] = value
    return df


def main() -> None:
    results_dir = ROOT / "results"
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows: list[pd.DataFrame] = []

    control_df = pd.read_csv(results_dir / "controls" / "transfer_controls_summary.csv")
    control_df["section"] = "controls"
    rows.append(control_df)

    trajectory_df = pd.read_csv(results_dir / "trajectory" / "trajectory24x384_wikitext_attnres_trajectory.csv")
    trajectory_df["section"] = "trajectory_attnres"
    rows.append(trajectory_df)
    trajectory_std_df = pd.read_csv(results_dir / "trajectory" / "trajectory24x384_wikitext_standard_trajectory.csv")
    trajectory_std_df["section"] = "trajectory_standard"
    rows.append(trajectory_std_df)

    oracle_tags = [
        "step50",
        "step100",
        "best200",
        "late1000",
        "step100_64",
        "best200_64",
        "late1000_64",
        "best200_p128d128_64",
    ]
    oracle_summary_rows = []
    routing_summary_rows = []
    gated_summary_rows = []
    for tag in oracle_tags:
        oracle_path = results_dir / "oracles" / f"{tag}_oracle_mask_alignment_summary.csv"
        loo_path = results_dir / "oracles" / f"{tag}_leave_one_out_alignment_summary.csv"
        routing_path = results_dir / "routing" / f"{tag}_routing_eval_summary.csv"
        if loo_path.exists():
            df = load_metric_table(loo_path, {"section": "loo_summary", "tag": tag})
            rows.append(df)
            tmp = df[df["metric"].isin(["spearman", "kendall", "recall_at_3", "ndcg_at_3"])][
                ["tag", "metric", "mean", "ci_low", "ci_high"]
            ].copy()
            oracle_summary_rows.append(tmp)
        if oracle_path.exists():
            df = load_metric_table(oracle_path, {"section": "oracle_summary", "tag": tag})
            rows.append(df)
        if routing_path.exists():
            df = load_metric_table(routing_path, {"section": "routing_summary", "tag": tag})
            rows.append(df)
            keep = df[
                (df["metric"] == "continuation_loss")
                & (df["method"].isin(["no_skip", "global_static", "prompt_fixed", "oracle_sequence", "balanced"]))
            ][["tag", "skip_count", "method", "mean", "ci_low", "ci_high"]].copy()
            routing_summary_rows.append(keep)
        gated_path = results_dir / "gated" / f"{tag}_gated_policy_sweep_summary.csv"
        if gated_path.exists():
            df = load_metric_table(gated_path, {"section": "gated_summary", "tag": tag})
            rows.append(df)
            gated_keep = df[
                (df["metric"].isin(["continuation_loss", "end_to_end_seconds"]))
                & (df["policy"].isin(["gated_to_full", "gated_to_global_static"]))
            ][["tag", "skip_count", "policy", "threshold", "route_fraction", "metric", "mean"]].copy()
            gated_summary_rows.append(gated_keep)

    summary_df = pd.concat(rows, ignore_index=True, sort=False)
    summary_df.to_csv(results_dir / "summary_v2.csv", index=False)

    # Plot 1: functional alignment across checkpoints.
    checkpoint_points = {
        "step50": 50,
        "step100": 100,
        "best200": 200,
        "late1000": 1000,
        "step100_64": 100,
        "best200_64": 200,
        "late1000_64": 1000,
    }
    loo_plot_rows = []
    oracle_plot_rows = []
    for tag, step in checkpoint_points.items():
        loo_path = results_dir / "oracles" / f"{tag}_leave_one_out_alignment_summary.csv"
        oracle_path = results_dir / "oracles" / f"{tag}_oracle_mask_alignment_summary.csv"
        if loo_path.exists():
            df = pd.read_csv(loo_path)
            spearman = df[df["metric"] == "spearman"].iloc[0]
            loo_plot_rows.append(
                {
                    "tag": tag,
                    "step": step,
                    "mean": spearman["mean"],
                    "ci_low": spearman["ci_low"],
                    "ci_high": spearman["ci_high"],
                }
            )
        if oracle_path.exists():
            df = pd.read_csv(oracle_path)
            sub = df[(df["skip_count"] == 1) & (df["metric"] == "delta_to_global_static")].iloc[0]
            oracle_plot_rows.append(
                {
                    "tag": tag,
                    "step": step,
                    "mean": sub["mean"],
                    "ci_low": sub["ci_low"],
                    "ci_high": sub["ci_high"],
                }
            )

    if loo_plot_rows and oracle_plot_rows:
        loo_plot_df = pd.DataFrame(loo_plot_rows).sort_values("step")
        oracle_plot_df = pd.DataFrame(oracle_plot_rows).sort_values("step")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(loo_plot_df["step"], loo_plot_df["mean"], marker="o")
        axes[0].fill_between(loo_plot_df["step"], loo_plot_df["ci_low"], loo_plot_df["ci_high"], alpha=0.2)
        axes[0].set_xlabel("checkpoint step")
        axes[0].set_ylabel("LOO spearman")
        axes[0].set_title("Functional Alignment")

        axes[1].plot(oracle_plot_df["step"], oracle_plot_df["mean"], marker="o", color="tab:red")
        axes[1].fill_between(oracle_plot_df["step"], oracle_plot_df["ci_low"], oracle_plot_df["ci_high"], alpha=0.2)
        axes[1].axhline(0.0, color="black", linewidth=1, linestyle="--")
        axes[1].set_xlabel("checkpoint step")
        axes[1].set_ylabel("Prompt - Global Static (skip=1)")
        axes[1].set_title("Static Baseline Gap")
        fig.tight_layout()
        fig.savefig(plots_dir / "v2_functional_alignment_vs_step.png", dpi=160)
        plt.close(fig)

    # Plot 2: routing baseline comparison for the strongest checkpoint.
    best_routing_path = results_dir / "routing" / "best200_64_routing_eval_summary.csv"
    if best_routing_path.exists():
        df = pd.read_csv(best_routing_path)
        sub = df[
            (df["metric"] == "continuation_loss")
            & (df["method"].isin(["no_skip", "balanced", "global_static", "prompt_fixed", "oracle_sequence"]))
        ].copy()
        fig, ax = plt.subplots(figsize=(7, 4))
        for method in ["no_skip", "balanced", "global_static", "prompt_fixed", "oracle_sequence"]:
            method_df = sub[sub["method"] == method].sort_values("skip_count")
            ax.plot(method_df["skip_count"], method_df["mean"], marker="o", label=method)
        ax.set_xlabel("skip count")
        ax.set_ylabel("continuation loss")
        ax.set_title("Best200 Routing Baselines")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "v2_best200_routing_baselines.png", dpi=160)
        plt.close(fig)

    # Plot 3: gated vs global static for best200_64.
    gated_path = results_dir / "gated" / "best200_64_gated_policy_sweep_summary.csv"
    if gated_path.exists() and best_routing_path.exists():
        gated_df = pd.read_csv(gated_path)
        routing_df = pd.read_csv(best_routing_path)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for idx, skip_count in enumerate([1, 2, 3]):
            ax = axes[idx]
            sub = gated_df[
                (gated_df["skip_count"] == skip_count)
                & (gated_df["policy"].isin(["gated_to_full", "gated_to_global_static"]))
                & (gated_df["metric"].isin(["continuation_loss", "end_to_end_seconds"]))
            ]
            pivot = sub.pivot_table(
                index=["policy", "threshold", "route_fraction"],
                columns="metric",
                values="mean",
            ).reset_index()
            for policy, policy_df in pivot.groupby("policy", sort=False):
                ax.plot(policy_df["end_to_end_seconds"], policy_df["continuation_loss"], marker="o", label=policy)
            static = routing_df[
                (routing_df["skip_count"] == skip_count)
                & (routing_df["method"] == "global_static")
                & (routing_df["metric"].isin(["continuation_loss", "end_to_end_seconds"]))
            ].pivot(index="method", columns="metric", values="mean")
            ax.scatter(
                static["end_to_end_seconds"],
                static["continuation_loss"],
                marker="X",
                s=90,
                color="black",
                label="global_static" if idx == 0 else None,
            )
            ax.set_title(f"skip={skip_count}")
            ax.set_xlabel("end-to-end seconds")
            ax.set_ylabel("continuation loss")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(plots_dir / "v2_best200_gated_vs_static.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
