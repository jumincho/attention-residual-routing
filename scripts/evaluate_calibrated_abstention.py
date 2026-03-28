#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-tag", type=str, required=True)
    parser.add_argument("--method", type=str, default="best_model")
    parser.add_argument("--output-tag", type=str, required=True)
    parser.add_argument("--thresholds", type=float, nargs="*", default=None)
    args = parser.parse_args()

    delta_dir = ROOT / "results" / "delta_selector"
    out_dir = ROOT / "results" / "calibrated_abstention"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = ROOT / "results" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    eval_df = pd.read_csv(delta_dir / f"{args.input_tag}_eval_per_sequence.csv")
    method_df = eval_df[eval_df["method"] == args.method].copy()
    static_df = eval_df[eval_df["method"] == "calib_global_static"][
        ["sequence_idx", "skip_count", "continuation_loss", "delta_to_calib_global_static"]
    ].rename(
        columns={
            "continuation_loss": "static_continuation_loss",
            "delta_to_calib_global_static": "static_delta_to_calib_global_static",
        }
    )
    merged = method_df.merge(static_df, on=["sequence_idx", "skip_count"], how="inner")

    rows = []
    thresholds = args.thresholds
    for skip_count in sorted(merged["skip_count"].unique().tolist()):
        subset = merged[merged["skip_count"] == skip_count].copy()
        if thresholds is None:
            candidate_thresholds = np.unique(
                np.concatenate(
                    [
                        np.asarray([0.0], dtype=np.float64),
                        np.quantile(subset["predicted_best_delta"].to_numpy(dtype=np.float64), [0.05, 0.1, 0.2, 0.5]),
                    ]
                )
            )
        else:
            candidate_thresholds = np.asarray(thresholds, dtype=np.float64)
        for threshold in candidate_thresholds.tolist():
            routed_mask = (subset["selected_nonstatic"] > 0.5) & (subset["predicted_best_delta"] <= threshold)
            chosen_loss = np.where(
                routed_mask.to_numpy(),
                subset["continuation_loss"].to_numpy(dtype=np.float64),
                subset["static_continuation_loss"].to_numpy(dtype=np.float64),
            )
            static_loss = subset["static_continuation_loss"].to_numpy(dtype=np.float64)
            actual_gain = static_loss - chosen_loss
            rows.append(
                {
                    "input_tag": args.input_tag,
                    "method": args.method,
                    "skip_count": skip_count,
                    "threshold": float(threshold),
                    "mean_continuation_loss": float(chosen_loss.mean()),
                    "mean_delta_to_calib_global_static": float((chosen_loss - static_loss).mean()),
                    "route_rate": float(routed_mask.mean()),
                    "route_precision": float((actual_gain[routed_mask.to_numpy()] > 0.0).mean()) if routed_mask.any() else float("nan"),
                    "num_routed": int(routed_mask.sum()),
                    "num_total": int(len(subset)),
                }
            )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_dir / f"{args.output_tag}_threshold_sweep.csv", index=False)

    for metric in ["mean_delta_to_calib_global_static", "route_rate"]:
        plt.figure(figsize=(6, 4))
        for skip_count in sorted(out_df["skip_count"].unique().tolist()):
            subset = out_df[out_df["skip_count"] == skip_count].sort_values("threshold")
            plt.plot(subset["threshold"], subset[metric], marker="o", label=f"skip={skip_count}")
        plt.xlabel("predicted delta threshold")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.legend()
        plt.savefig(plot_dir / f"calibrated_abstention_{args.output_tag}_{metric}.png", dpi=160)
        plt.close()


if __name__ == "__main__":
    main()
