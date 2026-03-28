#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT / "src"))
from attnres_routing.analysis import bootstrap_mean_ci  # noqa: E402


def best_model_from_selection(path: Path, feature_mode: str, bank_size: int, skip_count: int = 1) -> str | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    subset = df[
        (df["feature_mode"] == feature_mode)
        & (df["metric"] == "dev_delta_to_static")
        & (df["skip_count"] == skip_count)
        & (df["bank_size"] == bank_size)
    ].copy()
    if subset.empty:
        return None
    subset = subset.sort_values(["mean", "ci_high"], ascending=[True, True]).reset_index(drop=True)
    return str(subset.iloc[0]["model_name"])


def summary_row(path: Path, feature_mode: str, bank_size: int, model_name: str, metric: str, skip_count: int = 1) -> pd.Series | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    subset = df[
        (df["feature_mode"] == feature_mode)
        & (df["metric"] == metric)
        & (df["skip_count"] == skip_count)
        & (df["bank_size"] == bank_size)
        & (df["model_name"] == model_name)
    ]
    if subset.empty:
        return None
    return subset.iloc[0]


def per_seq_rows(path: Path, feature_mode: str, bank_size: int, model_name: str, skip_count: int = 1) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    subset = df[
        (df["feature_mode"] == feature_mode)
        & (df["skip_count"] == skip_count)
        & (df["bank_size"] == bank_size)
        & (df["model_name"] == model_name)
    ].copy()
    return subset


def first_matching(pattern: str) -> Path | None:
    matches = sorted((ROOT / "results" / "regret_reduction_v8").glob(pattern))
    return matches[0] if matches else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--winners-csv", required=True)
    parser.add_argument("--final-splits", nargs="+", default=["final_A", "final_B", "final_C"])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    winners = pd.read_csv(args.winners_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_seed_rows = []
    pooled_rows = []

    for _, winner in winners.iterrows():
        seed = int(winner["seed"])
        step = int(winner["step"])
        bank_size = int(winner["bank_size"])
        attn_model = str(winner["model_name"])
        attn_feature_mode = str(winner["feature_mode"])

        for split in args.final_splits:
            locked_prefix = f"v8_locked_seed{seed}_{split}_step{step}_b{bank_size}_{attn_model}_{attn_feature_mode}"
            locked_summary = ROOT / "results" / "regret_reduction_v8" / f"{locked_prefix}_{attn_feature_mode}_summary.csv"
            locked_per_seq = ROOT / "results" / "regret_reduction_v8" / f"{locked_prefix}_{attn_feature_mode}_per_sequence.csv"
            attn_delta = summary_row(locked_summary, attn_feature_mode, bank_size, attn_model, "delta_to_static")
            attn_regret = summary_row(locked_summary, attn_feature_mode, bank_size, attn_model, "delta_to_bank_upper_bound")

            attn_hidden_sel = ROOT / "results" / "regret_reduction_v8" / f"v8_necessity_attnres_seed{seed}_{split}_step{step}_b{bank_size}_hidden_model_selection.csv"
            attn_hidden_summary = ROOT / "results" / "regret_reduction_v8" / f"v8_necessity_attnres_seed{seed}_{split}_step{step}_b{bank_size}_hidden_summary.csv"
            attn_hidden_per_seq = ROOT / "results" / "regret_reduction_v8" / f"v8_necessity_attnres_seed{seed}_{split}_step{step}_b{bank_size}_hidden_per_sequence.csv"
            attn_hidden_model = best_model_from_selection(attn_hidden_sel, "hidden", bank_size)

            std_sel = first_matching(f"v8_necessity_standard_seed{seed}_{split}_step*_b{bank_size}_hidden_model_selection.csv")
            std_summary = first_matching(f"v8_necessity_standard_seed{seed}_{split}_step*_b{bank_size}_hidden_summary.csv")
            std_per_seq = first_matching(f"v8_necessity_standard_seed{seed}_{split}_step*_b{bank_size}_hidden_per_sequence.csv")
            std_hidden_model = best_model_from_selection(std_sel, "hidden", bank_size) if std_sel else None

            families = [
                ("attnres_dynamic", attn_model, locked_summary, locked_per_seq, attn_feature_mode),
                ("attnres_hidden", attn_hidden_model, attn_hidden_summary, attn_hidden_per_seq, "hidden"),
                ("standard_hidden", std_hidden_model, std_summary, std_per_seq, "hidden"),
            ]

            for family, model_name, summary_path, per_seq_path, feature_mode in families:
                if not model_name or summary_path is None:
                    continue
                delta = summary_row(summary_path, feature_mode, bank_size, model_name, "delta_to_static")
                regret = summary_row(summary_path, feature_mode, bank_size, model_name, "delta_to_bank_upper_bound")
                frac = summary_row(summary_path, feature_mode, bank_size, model_name, "fraction_improved")
                if delta is None or regret is None or frac is None:
                    continue
                per_seed_rows.append(
                    {
                        "seed": seed,
                        "final_split": split,
                        "family": family,
                        "step": step if family != "standard_hidden" else None,
                        "bank_size": bank_size,
                        "model_name": model_name,
                        "delta_to_static": float(delta["mean"]),
                        "delta_ci_low": float(delta["ci_low"]),
                        "delta_ci_high": float(delta["ci_high"]),
                        "regret_to_bank": float(regret["mean"]),
                        "fraction_improved": float(frac["mean"]),
                    }
                )
                per_df = per_seq_rows(per_seq_path, feature_mode, bank_size, model_name)
                if not per_df.empty:
                    per_df["seed"] = seed
                    per_df["final_split"] = split
                    per_df["family"] = family
                    pooled_rows.append(per_df)

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_df.to_csv(output_dir / "ccnews_v8_necessity_per_seed_split.csv", index=False)

    pooled_summary_rows = []
    if pooled_rows:
        pooled_df = pd.concat(pooled_rows, ignore_index=True)
        pooled_df.to_csv(output_dir / "ccnews_v8_necessity_pooled_per_sequence.csv", index=False)
        for family, group_df in pooled_df.groupby("family"):
            delta = bootstrap_mean_ci(group_df["actual_delta_to_static"].to_numpy())
            regret = bootstrap_mean_ci(group_df["delta_to_bank_upper_bound"].to_numpy())
            frac = bootstrap_mean_ci(group_df["improved_over_static"].to_numpy())
            pooled_summary_rows.extend(
                [
                    {
                        "family": family,
                        "metric": "delta_to_static",
                        "mean": float(delta["mean"]),
                        "ci_low": float(delta["ci_low"]),
                        "ci_high": float(delta["ci_high"]),
                        "n": int(delta["n"]),
                    },
                    {
                        "family": family,
                        "metric": "regret_to_bank",
                        "mean": float(regret["mean"]),
                        "ci_low": float(regret["ci_low"]),
                        "ci_high": float(regret["ci_high"]),
                        "n": int(regret["n"]),
                    },
                    {
                        "family": family,
                        "metric": "fraction_improved",
                        "mean": float(frac["mean"]),
                        "ci_low": float(frac["ci_low"]),
                        "ci_high": float(frac["ci_high"]),
                        "n": int(frac["n"]),
                    },
                ]
            )
    pd.DataFrame(pooled_summary_rows).to_csv(output_dir / "ccnews_v8_necessity_pooled_summary.csv", index=False)
    print(per_seed_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
