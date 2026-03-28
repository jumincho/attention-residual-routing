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
    valid = np.isfinite(values)
    out = np.full_like(values, np.nan, dtype=np.float64)
    if valid.sum() <= 1:
        out[valid] = 0.0
        return out
    signed = values[valid] if larger_is_better else -values[valid]
    mean = signed.mean()
    std = signed.std()
    if std < 1e-8:
        out[valid] = 0.0
        return out
    out[valid] = (signed - mean) / std
    return out


def step_from_tag(tag: str) -> int:
    match = re.search(r"step(\d+)", tag)
    if not match:
        raise ValueError(f"Cannot parse step from tag: {tag}")
    return int(match.group(1))


def find_selector_dev_metric(
    selector_dir: Path,
    selector_prefix: str,
    step: int,
    feature_mode: str,
    metric: str,
    skip_count: int,
    bank_size: int,
) -> float:
    pattern = f"{selector_prefix.format(step=step)}_{feature_mode}_model_selection.csv"
    path = selector_dir / pattern
    if not path.exists():
        return float("nan")
    df = pd.read_csv(path)
    subset = df[
        (df["metric"] == metric)
        & (df["skip_count"] == skip_count)
        & (df["bank_size"] == bank_size)
    ]
    if subset.empty:
        return float("nan")
    if metric.endswith("fraction_improved"):
        return float(subset.sort_values("mean", ascending=False).iloc[0]["mean"])
    return float(subset.sort_values("mean", ascending=True).iloc[0]["mean"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", type=str, nargs="+", required=True)
    parser.add_argument("--experiment-dir", type=str, required=True)
    parser.add_argument("--bank-size", type=int, default=32)
    parser.add_argument("--bank-skip", type=int, default=1)
    parser.add_argument("--feature-mode", type=str, default="attnres")
    parser.add_argument("--selector-prefix", type=str, default="")
    parser.add_argument("--selector-dir", type=str, default="results/ranker_v5")
    parser.add_argument("--output-tag", type=str, required=True)
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    metrics_df = pd.read_csv(experiment_dir / "metrics.csv")
    oracles_dir = ROOT / "results" / "oracles"
    bank_dir = ROOT / "results" / "bank_hygiene"
    selector_dir = ROOT / args.selector_dir
    out_dir = ROOT / "results" / "routing_readiness_v6"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for tag in args.tags:
        step = step_from_tag(tag)
        oracle_path = oracles_dir / f"{tag}_oracle_mask_alignment_summary.csv"
        loo_path = oracles_dir / f"{tag}_leave_one_out_alignment_summary.csv"
        bank_path = bank_dir / f"{tag.replace('_val', '')}_bank_summary.csv"
        if not oracle_path.exists() or not loo_path.exists() or not bank_path.exists():
            continue
        oracle_df = pd.read_csv(oracle_path)
        loo_df = pd.read_csv(loo_path)
        bank_df = pd.read_csv(bank_path)

        metric_rows = metrics_df[metrics_df["step"] == step]
        val_loss = float(metric_rows.iloc[-1]["val_loss"]) if not metric_rows.empty and pd.notna(metric_rows.iloc[-1]["val_loss"]) else np.nan
        bank_subset = bank_df[(bank_df["bank_size"] == args.bank_size) & (bank_df["skip_count"] == args.bank_skip)]

        row = {
            "tag": tag,
            "step": step,
            "val_loss": val_loss,
            "loo_spearman": float(loo_df[loo_df["metric"] == "spearman"]["mean"].iloc[0]),
            "loo_kendall": float(loo_df[loo_df["metric"] == "kendall"]["mean"].iloc[0]),
            "bank_headroom": float(bank_subset[bank_subset["metric"] == "oracle_headroom_over_calib_global"]["mean"].iloc[0]),
            "bank_upper_gain": float(-bank_subset[bank_subset["metric"] == "delta_bank_to_calib_global"]["mean"].iloc[0]),
            "bank_tail_frac_005": float(bank_subset[bank_subset["metric"] == "fraction_oracle_headroom_gt_0.005"]["mean"].iloc[0]),
            "bank_tail_frac_020": float(bank_subset[bank_subset["metric"] == "fraction_oracle_headroom_gt_0.020"]["mean"].iloc[0]),
            "bank_tail_frac_050": float(bank_subset[bank_subset["metric"] == "fraction_oracle_headroom_gt_0.050"]["mean"].iloc[0]),
        }
        if args.selector_prefix:
            row["selector_dev_delta"] = find_selector_dev_metric(
                selector_dir=selector_dir,
                selector_prefix=args.selector_prefix,
                step=step,
                feature_mode=args.feature_mode,
                metric="dev_delta_to_static",
                skip_count=args.bank_skip,
                bank_size=args.bank_size,
            )
            row["selector_dev_fraction"] = find_selector_dev_metric(
                selector_dir=selector_dir,
                selector_prefix=args.selector_prefix,
                step=step,
                feature_mode=args.feature_mode,
                metric="dev_fraction_improved",
                skip_count=args.bank_skip,
                bank_size=args.bank_size,
            )
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
    if df.empty:
        raise ValueError("No readiness rows found.")

    components = {
        "loo_spearman_z": zscore(df["loo_spearman"].to_numpy(), larger_is_better=True),
        "loo_kendall_z": zscore(df["loo_kendall"].to_numpy(), larger_is_better=True),
        "val_loss_z": zscore(df["val_loss"].to_numpy(), larger_is_better=False),
        "bank_headroom_z": zscore(df["bank_headroom"].to_numpy(), larger_is_better=True),
        "bank_upper_gain_z": zscore(df["bank_upper_gain"].to_numpy(), larger_is_better=True),
        "bank_tail_frac_020_z": zscore(df["bank_tail_frac_020"].to_numpy(), larger_is_better=True),
    }
    if "selector_dev_delta" in df.columns:
        components["selector_dev_delta_z"] = zscore(df["selector_dev_delta"].to_numpy(), larger_is_better=False)
        components["selector_dev_fraction_z"] = zscore(df["selector_dev_fraction"].to_numpy(), larger_is_better=True)

    for key, values in components.items():
        df[key] = values

    weight_map = {
        "loo_spearman_z": 0.75,
        "loo_kendall_z": 0.50,
        "val_loss_z": 0.25,
        "bank_headroom_z": 1.25,
        "bank_upper_gain_z": 1.00,
        "bank_tail_frac_020_z": 0.75,
        "selector_dev_delta_z": 1.25,
        "selector_dev_fraction_z": 0.50,
    }
    scores = []
    for _, row in df.iterrows():
        total = 0.0
        total_weight = 0.0
        for key, weight in weight_map.items():
            value = row.get(key, np.nan)
            if pd.notna(value):
                total += weight * float(value)
                total_weight += weight
        scores.append(total / max(total_weight, 1e-8))
    df["routing_readiness_v2"] = scores
    df.to_csv(out_dir / f"{args.output_tag}_routing_readiness_v2.csv", index=False)
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
