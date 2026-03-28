#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attnres_routing.analysis import bootstrap_mean_ci  # noqa: E402
from attnres_routing.routing import select_prompt_fixed_route  # noqa: E402


def parse_mask_id(mask_id: str, num_blocks: int) -> np.ndarray:
    mask = np.zeros(num_blocks, dtype=np.bool_)
    _, kept = mask_id.split(":", 1)
    if kept.strip():
        for token in kept.split(","):
            mask[int(token) - 1] = True
    return mask


def mask_to_id(mask: np.ndarray) -> str:
    kept = [str(idx + 1) for idx, value in enumerate(mask.tolist()) if value]
    return "keep:" + ",".join(kept)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(spearmanr(x, y).correlation)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, required=True)
    args = parser.parse_args()

    oracles_dir = ROOT / "results" / "oracles"
    routing_dir = ROOT / "results" / "routing"
    headroom_dir = ROOT / "results" / "headroom"
    headroom_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = ROOT / "results" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    oracle_df = pd.read_csv(oracles_dir / f"{args.tag}_oracle_mask_alignment.csv")
    routing_df = pd.read_csv(routing_dir / f"{args.tag}_routing_eval_per_sequence.csv")
    feature_df = pd.read_csv(oracles_dir / f"{args.tag}_sequence_features.csv")
    mask_df = pd.read_csv(oracles_dir / f"{args.tag}_exhaustive_mask_losses.csv")

    pivot = routing_df.pivot_table(
        index=["sequence_idx", "skip_count"],
        columns="method",
        values="continuation_loss",
    ).reset_index()
    merged = pivot.merge(oracle_df, on=["sequence_idx", "skip_count"], how="inner")
    feature_join = feature_df[
        [
            "sequence_idx",
            "split",
            "prompt_scores_json",
            "prompt_scores_attn_json",
            "prompt_scores_mlp_json",
            "prompt_chunk_utilities_json",
        ]
    ]
    merged = merged.merge(feature_join, on=["sequence_idx", "split"], how="left")
    merged["oracle_headroom_vs_global_static"] = merged["global_static"] - merged["oracle_sequence"]
    merged["prompt_penalty_vs_oracle"] = merged["prompt_fixed"] - merged["oracle_sequence"]
    merged["prompt_penalty_vs_global_static"] = merged["prompt_fixed"] - merged["global_static"]

    merged.to_csv(headroom_dir / f"{args.tag}_headroom_per_sequence.csv", index=False)

    threshold_rows = []
    for skip_count, subset in merged.groupby("skip_count", sort=True):
        values = subset["oracle_headroom_vs_global_static"].to_numpy()
        for threshold in [0.005, 0.01, 0.02, 0.05]:
            threshold_rows.append(
                {
                    "skip_count": skip_count,
                    "threshold": threshold,
                    "fraction": float((values > threshold).mean()),
                    "count": int((values > threshold).sum()),
                    "n": int(len(values)),
                }
            )
    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(headroom_dir / f"{args.tag}_headroom_thresholds.csv", index=False)

    corr_rows = []
    for skip_count, subset in merged.groupby("skip_count", sort=True):
        for feature in [
            "stability_spearman",
            "stability_top3_jaccard",
            "prompt_margin",
            "prompt_depth_entropy",
            "prompt_support_size",
        ]:
            corr_rows.append(
                {
                    "skip_count": skip_count,
                    "x": feature,
                    "y": "oracle_headroom_vs_global_static",
                    "spearman": safe_spearman(subset[feature].to_numpy(), subset["oracle_headroom_vs_global_static"].to_numpy()),
                }
            )
        corr_rows.append(
            {
                "skip_count": skip_count,
                "x": "oracle_headroom_vs_global_static",
                "y": "prompt_penalty_vs_oracle",
                "spearman": safe_spearman(
                    subset["oracle_headroom_vs_global_static"].to_numpy(),
                    subset["prompt_penalty_vs_oracle"].to_numpy(),
                ),
            }
        )
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(headroom_dir / f"{args.tag}_signal_correlations.csv", index=False)

    # Alternative scoring audit: combined vs attention-only vs mlp-only.
    example_prompt_scores = json.loads(feature_df.iloc[0]["prompt_scores_json"])
    num_blocks = len(example_prompt_scores) - 1
    exhaustive_pivot = mask_df[mask_df["method"] == "exhaustive_mask"].pivot_table(
        index=["sequence_idx", "skip_count"],
        columns="mask_id",
        values="continuation_loss",
    )
    alt_rows = []
    feature_indexed = feature_df.set_index("sequence_idx")
    global_static_losses = routing_df[routing_df["method"] == "global_static"].pivot_table(
        index=["sequence_idx", "skip_count"],
        values="continuation_loss",
    )
    oracle_losses = routing_df[routing_df["method"] == "oracle_sequence"].pivot_table(
        index=["sequence_idx", "skip_count"],
        values="continuation_loss",
    )
    for sequence_idx, feature_row in feature_indexed.iterrows():
        combined = np.asarray(json.loads(feature_row["prompt_scores_json"]))
        attn_only = np.asarray(json.loads(feature_row["prompt_scores_attn_json"]))
        mlp_only = np.asarray(json.loads(feature_row["prompt_scores_mlp_json"]))
        score_map = {"combined": combined, "attn_only": attn_only, "mlp_only": mlp_only}
        for skip_count in sorted(merged["skip_count"].unique().tolist()):
            skip_fraction = skip_count / num_blocks
            global_loss = float(global_static_losses.loc[(sequence_idx, skip_count)].iloc[0])
            oracle_loss = float(oracle_losses.loc[(sequence_idx, skip_count)].iloc[0])
            for score_name, scores in score_map.items():
                mask = select_prompt_fixed_route(scores, num_blocks, skip_fraction)
                mask_id = mask_to_id(mask)
                loss = float(exhaustive_pivot.loc[(sequence_idx, skip_count), mask_id])
                alt_rows.append(
                    {
                        "sequence_idx": sequence_idx,
                        "skip_count": skip_count,
                        "score_name": score_name,
                        "continuation_loss": loss,
                        "delta_to_global_static": loss - global_loss,
                        "delta_to_oracle": loss - oracle_loss,
                    }
                )
    alt_df = pd.DataFrame(alt_rows)
    alt_df.to_csv(headroom_dir / f"{args.tag}_alternative_score_audit.csv", index=False)
    alt_summary_rows = []
    for (skip_count, score_name), subset in alt_df.groupby(["skip_count", "score_name"], sort=True):
        for metric in ["continuation_loss", "delta_to_global_static", "delta_to_oracle"]:
            ci = bootstrap_mean_ci(subset[metric].to_numpy())
            alt_summary_rows.append(
                {
                    "skip_count": skip_count,
                    "score_name": score_name,
                    "metric": metric,
                    **ci,
                }
            )
    alt_summary_df = pd.DataFrame(alt_summary_rows)
    alt_summary_df.to_csv(headroom_dir / f"{args.tag}_alternative_score_audit_summary.csv", index=False)

    # Plots.
    for skip_count, subset in merged.groupby("skip_count", sort=True):
        values = np.sort(subset["oracle_headroom_vs_global_static"].to_numpy())
        ecdf = np.arange(1, len(values) + 1) / max(len(values), 1)
        plt.figure(figsize=(5, 4))
        plt.hist(subset["oracle_headroom_vs_global_static"], bins=20)
        plt.xlabel("oracle headroom over global static")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(plot_dir / f"headroom_hist_{args.tag}_skip{skip_count}.png", dpi=160)
        plt.close()

        plt.figure(figsize=(5, 4))
        plt.plot(values, ecdf)
        plt.xlabel("oracle headroom over global static")
        plt.ylabel("ECDF")
        plt.tight_layout()
        plt.savefig(plot_dir / f"headroom_ecdf_{args.tag}_skip{skip_count}.png", dpi=160)
        plt.close()

        plt.figure(figsize=(5, 4))
        plt.scatter(subset["stability_spearman"], subset["oracle_headroom_vs_global_static"], alpha=0.7)
        plt.xlabel("stability spearman")
        plt.ylabel("oracle headroom over global static")
        plt.tight_layout()
        plt.savefig(plot_dir / f"headroom_vs_stability_{args.tag}_skip{skip_count}.png", dpi=160)
        plt.close()

        plt.figure(figsize=(5, 4))
        plt.scatter(subset["oracle_headroom_vs_global_static"], subset["prompt_penalty_vs_oracle"], alpha=0.7)
        plt.xlabel("oracle headroom over global static")
        plt.ylabel("prompt-fixed penalty vs oracle")
        plt.tight_layout()
        plt.savefig(plot_dir / f"headroom_vs_prompt_penalty_{args.tag}_skip{skip_count}.png", dpi=160)
        plt.close()

    plt.figure(figsize=(7, 4))
    alt_plot = alt_summary_df[alt_summary_df["metric"] == "delta_to_global_static"]
    for score_name, subset in alt_plot.groupby("score_name", sort=False):
        subset = subset.sort_values("skip_count")
        plt.plot(subset["skip_count"], subset["mean"], marker="o", label=score_name)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("skip count")
    plt.ylabel("delta to global static")
    plt.tight_layout()
    plt.legend()
    plt.savefig(plot_dir / f"headroom_alt_score_audit_{args.tag}.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
