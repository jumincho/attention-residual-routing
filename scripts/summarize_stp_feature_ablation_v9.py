#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def metric_ascending(metric: str) -> bool:
    return not metric.endswith("fraction_improved") and not metric.endswith("match")


def read_best_row(
    base_dir: Path,
    output_tag: str,
    feature_mode: str,
    skip_count: int,
    bank_size: int,
    selection_metric: str,
) -> dict[str, object] | None:
    sel_path = base_dir / f"{output_tag}_{feature_mode}_model_selection.csv"
    sum_path = base_dir / f"{output_tag}_{feature_mode}_summary.csv"
    if not sel_path.exists() or not sum_path.exists():
        return None

    sel_df = pd.read_csv(sel_path)
    sum_df = pd.read_csv(sum_path)
    dev_df = sel_df[
        (sel_df["metric"] == selection_metric)
        & (sel_df["skip_count"] == skip_count)
        & (sel_df["bank_size"] == bank_size)
    ].copy()
    if dev_df.empty:
        return None
    best_dev = dev_df.sort_values("mean", ascending=metric_ascending(selection_metric)).iloc[0]
    model_name = str(best_dev["model_name"])

    def lookup(metric: str) -> float:
        subset = sum_df[
            (sum_df["metric"] == metric)
            & (sum_df["model_name"] == model_name)
            & (sum_df["skip_count"] == skip_count)
            & (sum_df["bank_size"] == bank_size)
        ]
        if subset.empty:
            return float("nan")
        return float(subset.iloc[0]["mean"])

    return {
        "feature_mode": feature_mode,
        "dev_best_model": model_name,
        "selection_metric": selection_metric,
        "dev_metric_value": float(best_dev["mean"]),
        "eval_delta_to_static": lookup("delta_to_static"),
        "eval_delta_to_bank_upper_bound": lookup("delta_to_bank_upper_bound"),
        "eval_fraction_improved": lookup("fraction_improved"),
        "eval_oracle_in_bank_match": lookup("oracle_in_bank_match"),
    }


def plot_metric(df: pd.DataFrame, metric: str, output_path: Path, ylabel: str) -> None:
    if df.empty:
        return
    plt.figure(figsize=(max(7, 1.1 * len(df)), 4))
    plt.bar(df["feature_mode"], df[metric])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-subdir", type=str, default="stp_feature_selector_v9")
    parser.add_argument("--output-tag", type=str, required=True)
    parser.add_argument("--feature-modes", type=str, nargs="+", required=True)
    parser.add_argument("--skip-count", type=int, default=1)
    parser.add_argument("--bank-size", type=int, default=32)
    parser.add_argument("--selection-metric", type=str, default="dev_delta_to_static")
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--plot-prefix", type=str, default="")
    args = parser.parse_args()

    base_dir = ROOT / "results" / args.results_subdir
    rows = []
    for feature_mode in args.feature_modes:
        row = read_best_row(
            base_dir=base_dir,
            output_tag=args.output_tag,
            feature_mode=feature_mode,
            skip_count=args.skip_count,
            bank_size=args.bank_size,
            selection_metric=args.selection_metric,
        )
        if row is not None:
            rows.append(row)

    df = pd.DataFrame(rows)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    if args.plot_prefix and not df.empty:
        plot_dir = ROOT / "results" / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_metric(
            df,
            "eval_delta_to_static",
            plot_dir / f"{args.plot_prefix}_eval_delta_to_static.png",
            "eval delta_to_static",
        )
        plot_metric(
            df,
            "eval_delta_to_bank_upper_bound",
            plot_dir / f"{args.plot_prefix}_eval_delta_to_bank.png",
            "eval delta_to_bank_upper_bound",
        )
        plot_metric(
            df,
            "eval_fraction_improved",
            plot_dir / f"{args.plot_prefix}_eval_fraction_improved.png",
            "eval fraction_improved",
        )

    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
