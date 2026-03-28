#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_metric_rows(path: Path, feature_mode: str, model_name: str, bank_size: int, skip_count: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[
        (df["feature_mode"] == feature_mode)
        & (df["model_name"] == model_name)
        & (df["bank_size"] == bank_size)
        & (df["skip_count"] == skip_count)
    ].copy()


def compare_summary(
    original_path: Path,
    rerun_path: Path,
    feature_mode: str,
    model_name: str,
    bank_size: int,
    skip_count: int,
) -> list[dict]:
    orig = load_metric_rows(original_path, feature_mode, model_name, bank_size, skip_count)
    rerun = load_metric_rows(rerun_path, feature_mode, model_name, bank_size, skip_count)
    rows: list[dict] = []
    for metric in sorted(set(orig["metric"]).intersection(set(rerun["metric"]))):
        o = orig[orig["metric"] == metric].iloc[0]
        r = rerun[rerun["metric"] == metric].iloc[0]
        rows.append(
            {
                "section": "summary",
                "metric": metric,
                "original_mean": float(o["mean"]),
                "rerun_mean": float(r["mean"]),
                "abs_diff": abs(float(o["mean"]) - float(r["mean"])),
                "original_ci_low": float(o["ci_low"]) if "ci_low" in o else None,
                "rerun_ci_low": float(r["ci_low"]) if "ci_low" in r else None,
                "original_ci_high": float(o["ci_high"]) if "ci_high" in o else None,
                "rerun_ci_high": float(r["ci_high"]) if "ci_high" in r else None,
            }
        )
    return rows


def compare_per_sequence(
    original_path: Path,
    rerun_path: Path,
    feature_mode: str,
    model_name: str,
    bank_size: int,
    skip_count: int,
) -> list[dict]:
    orig = pd.read_csv(original_path)
    rerun = pd.read_csv(rerun_path)
    orig = orig[
        (orig["feature_mode"] == feature_mode)
        & (orig["model_name"] == model_name)
        & (orig["bank_size"] == bank_size)
        & (orig["skip_count"] == skip_count)
    ].copy()
    rerun = rerun[
        (rerun["feature_mode"] == feature_mode)
        & (rerun["model_name"] == model_name)
        & (rerun["bank_size"] == bank_size)
        & (rerun["skip_count"] == skip_count)
    ].copy()
    merged = orig.merge(
        rerun,
        on=["sequence_idx"],
        suffixes=("_orig", "_rerun"),
        how="inner",
    )
    rows: list[dict] = []
    for col in [
        "predicted_delta",
        "actual_delta_to_static",
        "improved_over_static",
        "delta_to_bank_upper_bound",
        "oracle_in_bank_match",
    ]:
        orig_col = f"{col}_orig"
        rerun_col = f"{col}_rerun"
        if orig_col not in merged.columns or rerun_col not in merged.columns:
            continue
        diff = (merged[orig_col] - merged[rerun_col]).abs()
        rows.append(
            {
                "section": "per_sequence",
                "metric": col,
                "matched_rows": int(len(merged)),
                "max_abs_diff": float(diff.max()) if len(diff) else None,
                "mean_abs_diff": float(diff.mean()) if len(diff) else None,
                "exact_match": bool(diff.eq(0).all()) if len(diff) else None,
            }
        )
    return rows


def compare_deploy(original_path: Path, rerun_path: Path) -> list[dict]:
    orig = pd.read_csv(original_path)
    rerun = pd.read_csv(rerun_path)
    shared = sorted(set(orig["method"]).intersection(set(rerun["method"])))
    rows: list[dict] = []
    for method in shared:
        o = orig[orig["method"] == method].iloc[0]
        r = rerun[rerun["method"] == method].iloc[0]
        for metric in [
            "mean_continuation_loss",
            "delta_to_global_static",
            "end_to_end_seconds_per_sequence",
            "decode_tokens_per_sec",
            "selector_overhead_seconds_total",
            "route_count",
        ]:
            if metric not in o or metric not in r:
                continue
            rows.append(
                {
                    "section": "deploy",
                    "method": method,
                    "metric": metric,
                    "original_value": float(o[metric]),
                    "rerun_value": float(r[metric]),
                    "abs_diff": abs(float(o[metric]) - float(r[metric])),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-prefix", required=True)
    parser.add_argument("--rerun-prefix", required=True)
    parser.add_argument("--feature-mode", default="attnres")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--bank-size", type=int, default=32)
    parser.add_argument("--skip-count", type=int, default=1)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    rows: list[dict] = []

    orig_summary = ROOT / "results" / "regret_reduction_v8" / f"{args.original_prefix}_{args.feature_mode}_summary.csv"
    rerun_summary = ROOT / "results" / "regret_reduction_v8" / f"{args.rerun_prefix}_{args.feature_mode}_summary.csv"
    if orig_summary.exists() and rerun_summary.exists():
        rows.extend(
            compare_summary(
                orig_summary,
                rerun_summary,
                args.feature_mode,
                args.model_name,
                args.bank_size,
                args.skip_count,
            )
        )

    orig_per_seq = ROOT / "results" / "regret_reduction_v8" / f"{args.original_prefix}_{args.feature_mode}_per_sequence.csv"
    rerun_per_seq = ROOT / "results" / "regret_reduction_v8" / f"{args.rerun_prefix}_{args.feature_mode}_per_sequence.csv"
    if orig_per_seq.exists() and rerun_per_seq.exists():
        rows.extend(
            compare_per_sequence(
                orig_per_seq,
                rerun_per_seq,
                args.feature_mode,
                args.model_name,
                args.bank_size,
                args.skip_count,
            )
        )

    orig_deploy = ROOT / "results" / "systems_routing_v7" / f"{args.original_prefix}_deploy_summary.csv"
    rerun_deploy = ROOT / "results" / "systems_routing_v7" / f"{args.rerun_prefix}_deploy_summary.csv"
    if orig_deploy.exists() and rerun_deploy.exists():
        rows.extend(compare_deploy(orig_deploy, rerun_deploy))

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
