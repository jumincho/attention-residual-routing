#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def bootstrap_mean(values: np.ndarray, samples: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    values = values.astype(np.float64)
    means = []
    for _ in range(samples):
        idx = rng.integers(0, len(values), size=len(values))
        means.append(values[idx].mean())
    means = np.asarray(means, dtype=np.float64)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-sequence-csv", type=str, required=True)
    parser.add_argument("--feature-mode", type=str, required=True)
    parser.add_argument("--skip-count", type=int, required=True)
    parser.add_argument("--selected-model", type=str, required=True)
    parser.add_argument("--output-tag", type=str, required=True)
    parser.add_argument("--delta-thresholds", type=float, nargs="*", default=None)
    parser.add_argument("--uncertainty-thresholds", type=float, nargs="*", default=None)
    parser.add_argument("--agreement-thresholds", type=float, nargs="*", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.per_sequence_csv)
    subset = df[(df["feature_mode"] == args.feature_mode) & (df["skip_count"] == args.skip_count)].copy()
    if subset.empty:
        raise ValueError("No rows for requested feature_mode/skip_count.")

    model_names = sorted(subset["model_name"].unique().tolist())
    pivot_pred = subset.pivot(index="sequence_idx", columns="model_name", values="predicted_delta")
    pivot_mask = subset.pivot(index="sequence_idx", columns="model_name", values="selected_mask_id")
    pivot_actual = subset.pivot(index="sequence_idx", columns="model_name", values="actual_delta_to_static")

    base = subset[subset["model_name"] == args.selected_model].copy()
    base = base.set_index("sequence_idx")
    base["ensemble_pred_std"] = pivot_pred.std(axis=1)
    base["ensemble_pred_mean"] = pivot_pred.mean(axis=1)
    selected_mask = pivot_mask[args.selected_model]
    base["mask_agreement"] = (pivot_mask.eq(selected_mask, axis=0)).mean(axis=1)
    base["actual_delta_mean_models"] = pivot_actual.mean(axis=1)
    base = base.reset_index()

    delta_thresholds = (
        np.unique(
            np.concatenate(
                [
                    np.asarray([-0.01, -0.005, 0.0, 0.005], dtype=np.float64),
                    np.quantile(base["predicted_delta"].to_numpy(dtype=np.float64), [0.1, 0.25, 0.5]),
                ]
            )
        )
        if args.delta_thresholds is None
        else np.asarray(args.delta_thresholds, dtype=np.float64)
    )
    uncertainty_thresholds = (
        np.unique(np.quantile(base["ensemble_pred_std"].to_numpy(dtype=np.float64), [0.25, 0.5, 0.75]))
        if args.uncertainty_thresholds is None
        else np.asarray(args.uncertainty_thresholds, dtype=np.float64)
    )
    agreement_thresholds = (
        np.asarray([0.25, 0.5, 0.75, 1.0], dtype=np.float64)
        if args.agreement_thresholds is None
        else np.asarray(args.agreement_thresholds, dtype=np.float64)
    )

    rows = []
    for delta_thr in delta_thresholds.tolist():
        for unc_thr in uncertainty_thresholds.tolist():
            for agree_thr in agreement_thresholds.tolist():
                routed = (
                    (base["predicted_delta"] <= delta_thr)
                    & (base["ensemble_pred_std"] <= unc_thr)
                    & (base["mask_agreement"] >= agree_thr)
                )
                chosen_delta = np.where(
                    routed.to_numpy(),
                    base["actual_delta_to_static"].to_numpy(dtype=np.float64),
                    0.0,
                )
                precision = float((base.loc[routed, "actual_delta_to_static"] < 0.0).mean()) if routed.any() else float("nan")
                mean_delta, ci_low, ci_high = bootstrap_mean(chosen_delta)
                rows.append(
                    {
                        "output_tag": args.output_tag,
                        "feature_mode": args.feature_mode,
                        "skip_count": args.skip_count,
                        "selected_model": args.selected_model,
                        "delta_threshold": float(delta_thr),
                        "uncertainty_threshold": float(unc_thr),
                        "agreement_threshold": float(agree_thr),
                        "mean_delta_to_static": mean_delta,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "route_rate": float(routed.mean()),
                        "route_precision": precision,
                        "num_routed": int(routed.sum()),
                        "num_total": int(len(base)),
                    }
                )

    out_dir = ROOT / "results" / "calibrated_abstention_v5"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = ROOT / "results" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows).sort_values(
        ["mean_delta_to_static", "route_rate", "delta_threshold", "uncertainty_threshold", "agreement_threshold"]
    )
    out_df.to_csv(out_dir / f"{args.output_tag}_threshold_sweep.csv", index=False)

    best = out_df.nsmallest(20, "mean_delta_to_static")
    plt.figure(figsize=(6, 4))
    plt.scatter(out_df["route_rate"], out_df["mean_delta_to_static"], alpha=0.3)
    plt.scatter(best["route_rate"], best["mean_delta_to_static"], color="red", s=18)
    plt.xlabel("route rate")
    plt.ylabel("mean delta to static")
    plt.tight_layout()
    plt.savefig(plot_dir / f"calibrated_abstention_v5_{args.output_tag}_pareto.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.scatter(base["ensemble_pred_std"], base["actual_delta_to_static"], alpha=0.35)
    plt.xlabel("ensemble predicted-delta std")
    plt.ylabel("actual delta to static")
    plt.tight_layout()
    plt.savefig(plot_dir / f"calibrated_abstention_v5_{args.output_tag}_uncertainty_vs_actual.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
