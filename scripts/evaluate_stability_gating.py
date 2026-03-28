#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attnres_routing.analysis import bootstrap_mean_ci  # noqa: E402


def bootstrap_summary(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str], seed: int) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = {col: value for col, value in zip(group_cols, keys)}
        for metric in metric_cols:
            ci = bootstrap_mean_ci(group[metric].to_numpy(), seed=seed)
            rows.append({**key_map, "metric": metric, **ci})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routing-csv", type=str, required=True)
    parser.add_argument("--oracle-csv", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-thresholds", type=int, default=9)
    parser.add_argument("--tag", type=str, default="main")
    args = parser.parse_args()

    routing_df = pd.read_csv(args.routing_csv)
    oracle_df = pd.read_csv(args.oracle_csv)

    gated_dir = ROOT / "results" / "gated"
    gated_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = ROOT / "results" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    tag_prefix = f"{args.tag}_" if args.tag else ""

    pivot = routing_df.pivot_table(
        index=["sequence_idx", "skip_count"],
        columns="method",
        values=["continuation_loss", "decode_tokens_per_sec", "end_to_end_seconds", "active_blocks"],
    )
    pivot.columns = ["_".join(col).strip() for col in pivot.columns.to_flat_index()]
    merged = pivot.reset_index().merge(oracle_df, on=["sequence_idx", "skip_count"], how="inner")
    merged["combined_score"] = merged["stability_spearman"].clip(lower=0.0) * merged["prompt_margin"].clip(lower=0.0)
    merged["prompt_penalty_vs_full"] = merged["continuation_loss_prompt_fixed"] - merged["continuation_loss_no_skip"]

    rows = []
    for skip_count, subset in merged.groupby("skip_count", sort=True):
        thresholds = np.quantile(subset["combined_score"], np.linspace(0.0, 1.0, args.num_thresholds))
        thresholds = np.unique(thresholds)
        for threshold in thresholds:
            use_prompt = subset["combined_score"] >= threshold
            policies = {
                "always_full": pd.Series(False, index=subset.index),
                "always_prompt_fixed": pd.Series(True, index=subset.index),
                "gated_to_full": use_prompt,
                "gated_to_global_static": use_prompt,
            }
            for policy, gate in policies.items():
                if policy == "always_full":
                    loss = subset["continuation_loss_no_skip"]
                    tok_s = subset["decode_tokens_per_sec_no_skip"]
                    end_to_end = subset["end_to_end_seconds_no_skip"]
                    active = subset["active_blocks_no_skip"]
                elif policy == "always_prompt_fixed":
                    loss = subset["continuation_loss_prompt_fixed"]
                    tok_s = subset["decode_tokens_per_sec_prompt_fixed"]
                    end_to_end = subset["end_to_end_seconds_prompt_fixed"]
                    active = subset["active_blocks_prompt_fixed"]
                elif policy == "gated_to_full":
                    loss = np.where(gate, subset["continuation_loss_prompt_fixed"], subset["continuation_loss_no_skip"])
                    tok_s = np.where(gate, subset["decode_tokens_per_sec_prompt_fixed"], subset["decode_tokens_per_sec_no_skip"])
                    end_to_end = np.where(gate, subset["end_to_end_seconds_prompt_fixed"], subset["end_to_end_seconds_no_skip"])
                    active = np.where(gate, subset["active_blocks_prompt_fixed"], subset["active_blocks_no_skip"])
                else:
                    loss = np.where(
                        gate,
                        subset["continuation_loss_prompt_fixed"],
                        subset["continuation_loss_global_static"],
                    )
                    tok_s = np.where(
                        gate,
                        subset["decode_tokens_per_sec_prompt_fixed"],
                        subset["decode_tokens_per_sec_global_static"],
                    )
                    end_to_end = np.where(
                        gate,
                        subset["end_to_end_seconds_prompt_fixed"],
                        subset["end_to_end_seconds_global_static"],
                    )
                    active = np.where(gate, subset["active_blocks_prompt_fixed"], subset["active_blocks_global_static"])

                for seq_idx, lval, sval, eval_s, aval in zip(subset["sequence_idx"], loss, tok_s, end_to_end, active):
                    rows.append(
                        {
                            "sequence_idx": seq_idx,
                            "skip_count": skip_count,
                            "policy": policy,
                            "threshold": float(threshold),
                            "route_fraction": float(gate.mean()) if policy.startswith("gated") else float(policy == "always_prompt_fixed"),
                            "continuation_loss": float(lval),
                            "decode_tokens_per_sec": float(sval),
                            "end_to_end_seconds": float(eval_s),
                            "active_blocks": float(aval),
                        }
                    )

    gated_df = pd.DataFrame(rows)
    summary_df = bootstrap_summary(
        gated_df,
        group_cols=["skip_count", "policy", "threshold", "route_fraction"],
        metric_cols=["continuation_loss", "decode_tokens_per_sec", "end_to_end_seconds", "active_blocks"],
        seed=args.seed,
    )
    gated_df.to_csv(gated_dir / f"{tag_prefix}gated_policy_sweep.csv", index=False)
    summary_df.to_csv(gated_dir / f"{tag_prefix}gated_policy_sweep_summary.csv", index=False)

    calibration_rows = []
    for skip_count, subset in merged.groupby("skip_count", sort=True):
        bins = np.quantile(subset["combined_score"], np.linspace(0.0, 1.0, 6))
        bins = np.unique(bins)
        if len(bins) < 2:
            continue
        bucket_ids = pd.cut(subset["combined_score"], bins=bins, include_lowest=True, duplicates="drop")
        for bucket, bucket_df in subset.groupby(bucket_ids, observed=False):
            if bucket_df.empty:
                continue
            calibration_rows.append(
                {
                    "skip_count": skip_count,
                    "bucket": str(bucket),
                    "mean_score": float(bucket_df["combined_score"].mean()),
                    "mean_penalty_vs_full": float(bucket_df["prompt_penalty_vs_full"].mean()),
                    "count": int(len(bucket_df)),
                }
            )
    calibration_df = pd.DataFrame(calibration_rows)
    calibration_df.to_csv(gated_dir / f"{tag_prefix}gated_calibration.csv", index=False)

    for skip_count in sorted(gated_df["skip_count"].unique().tolist()):
        subset = summary_df[
            (summary_df["skip_count"] == skip_count)
            & (summary_df["metric"].isin(["continuation_loss", "decode_tokens_per_sec"]))
            & (summary_df["policy"].isin(["always_full", "always_prompt_fixed", "gated_to_full", "gated_to_global_static"]))
        ]
        if subset.empty:
            continue
        loss_df = subset[subset["metric"] == "continuation_loss"]
        speed_df = subset[subset["metric"] == "decode_tokens_per_sec"]
        merged_plot = loss_df.merge(
            speed_df,
            on=["skip_count", "policy", "threshold", "route_fraction"],
            suffixes=("_loss", "_speed"),
        )
        plt.figure(figsize=(6, 4))
        for policy, policy_df in merged_plot.groupby("policy", sort=False):
            plt.plot(
                policy_df["mean_speed"],
                policy_df["mean_loss"],
                marker="o",
                label=policy,
            )
        plt.xlabel("decode tokens/s")
        plt.ylabel("continuation loss")
        plt.tight_layout()
        plt.legend()
        plt.savefig(plot_dir / f"{tag_prefix}gated_pareto_skip{skip_count}.png", dpi=160)
        plt.close()

    if not calibration_df.empty:
        plt.figure(figsize=(6, 4))
        for skip_count, subset in calibration_df.groupby("skip_count", sort=True):
            plt.plot(subset["mean_score"], subset["mean_penalty_vs_full"], marker="o", label=f"skip={skip_count}")
        plt.xlabel("combined stability-margin score")
        plt.ylabel("prompt-fixed penalty vs full")
        plt.tight_layout()
        plt.legend()
        plt.savefig(plot_dir / f"{tag_prefix}gated_penalty_vs_score.png", dpi=160)
        plt.close()

    doc_name = f"stability_gated_routing_{args.tag}.md" if args.tag else "stability_gated_routing.md"
    with open(ROOT / "docs" / doc_name, "w", encoding="utf-8") as f:
        f.write("# Stability-Gated Routing\n\n")
        f.write("- score: `combined_score = max(stability_spearman, 0) * max(prompt_margin, 0)`\n")
        f.write("- policies evaluated:\n")
        f.write("  - always full\n")
        f.write("  - always prompt-fixed\n")
        f.write("  - gated prompt-fixed with fallback to full\n")
        f.write("  - gated prompt-fixed with fallback to best global static\n\n")
        if not summary_df.empty:
            for skip_count in sorted(summary_df["skip_count"].unique().tolist()):
                f.write(f"## Skip {skip_count}\n\n")
                subset = summary_df[
                    (summary_df["skip_count"] == skip_count)
                    & (summary_df["policy"].isin(["always_full", "always_prompt_fixed", "gated_to_full", "gated_to_global_static"]))
                    & (summary_df["metric"].isin(["continuation_loss", "decode_tokens_per_sec", "end_to_end_seconds"]))
                ]
                for policy in ["always_full", "always_prompt_fixed", "gated_to_full", "gated_to_global_static"]:
                    policy_df = subset[subset["policy"] == policy]
                    if policy_df.empty:
                        continue
                    f.write(f"### {policy}\n\n")
                    for metric in ["continuation_loss", "decode_tokens_per_sec", "end_to_end_seconds"]:
                        row = policy_df[policy_df["metric"] == metric].sort_values("mean")
                        if row.empty:
                            continue
                        best = row.iloc[0]
                        f.write(f"- best {metric}: {best['mean']:.4f} [{best['ci_low']:.4f}, {best['ci_high']:.4f}]\n")
                    f.write("\n")


if __name__ == "__main__":
    main()
