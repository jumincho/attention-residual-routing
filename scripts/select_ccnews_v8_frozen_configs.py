#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "regret_reduction_v8"
DEFAULT_MODELS = [
    "rf_pair",
    "hgb_pair",
    "dual_tower_rank",
    "dual_tower_listwise",
    "binary_gate_top1",
    "ternary_gate_top2",
    "retrieval_rerank_top2",
    "retrieval_rerank_top4",
]


def selection_path(seed: int, step: int, bank_size: int, feature_mode: str) -> Path:
    return RESULTS_DIR / f"v8_ccnews_seed{seed}_full_step{step}_dev2048_b{bank_size}_{feature_mode}_model_selection.csv"


def summary_path(seed: int, step: int, bank_size: int, feature_mode: str) -> Path:
    return RESULTS_DIR / f"v8_ccnews_seed{seed}_full_step{step}_dev2048_b{bank_size}_{feature_mode}_summary.csv"


def load_best_row(seed: int, step: int, bank_size: int, feature_mode: str, allowed_models: list[str]) -> dict | None:
    path = selection_path(seed, step, bank_size, feature_mode)
    summary = summary_path(seed, step, bank_size, feature_mode)
    if not path.exists() or not summary.exists():
        return None
    df = pd.read_csv(path)
    summary_df = pd.read_csv(summary)

    subset = df[
        (df["feature_mode"] == feature_mode)
        & (df["metric"] == "dev_delta_to_static")
        & (df["skip_count"] == 1)
        & (df["bank_size"] == bank_size)
        & (df["model_name"].isin(allowed_models))
    ].copy()
    if subset.empty:
        return None
    regret_df = df[
        (df["feature_mode"] == feature_mode)
        & (df["metric"] == "dev_delta_to_bank_upper_bound")
        & (df["skip_count"] == 1)
        & (df["bank_size"] == bank_size)
        & (df["model_name"].isin(allowed_models))
    ][["model_name", "mean"]].rename(columns={"mean": "dev_regret"})
    frac_df = df[
        (df["feature_mode"] == feature_mode)
        & (df["metric"] == "dev_fraction_improved")
        & (df["skip_count"] == 1)
        & (df["bank_size"] == bank_size)
        & (df["model_name"].isin(allowed_models))
    ][["model_name", "mean"]].rename(columns={"mean": "dev_fraction_improved"})
    top1_df = df[
        (df["feature_mode"] == feature_mode)
        & (df["metric"] == "dev_oracle_in_bank_match")
        & (df["skip_count"] == 1)
        & (df["bank_size"] == bank_size)
        & (df["model_name"].isin(allowed_models))
    ][["model_name", "mean"]].rename(columns={"mean": "dev_oracle_in_bank_match"})

    full_delta_df = summary_df[
        (summary_df["feature_mode"] == feature_mode)
        & (summary_df["metric"] == "delta_to_static")
        & (summary_df["skip_count"] == 1)
        & (summary_df["bank_size"] == bank_size)
        & (summary_df["model_name"].isin(allowed_models))
    ][["model_name", "mean", "ci_low", "ci_high"]].rename(
        columns={
            "mean": "full_delta_to_static",
            "ci_low": "full_delta_ci_low",
            "ci_high": "full_delta_ci_high",
        }
    )
    full_regret_df = summary_df[
        (summary_df["feature_mode"] == feature_mode)
        & (summary_df["metric"] == "delta_to_bank_upper_bound")
        & (summary_df["skip_count"] == 1)
        & (summary_df["bank_size"] == bank_size)
        & (summary_df["model_name"].isin(allowed_models))
    ][["model_name", "mean"]].rename(columns={"mean": "full_regret_to_bank"})
    full_frac_df = summary_df[
        (summary_df["feature_mode"] == feature_mode)
        & (summary_df["metric"] == "fraction_improved")
        & (summary_df["skip_count"] == 1)
        & (summary_df["bank_size"] == bank_size)
        & (summary_df["model_name"].isin(allowed_models))
    ][["model_name", "mean"]].rename(columns={"mean": "full_fraction_improved"})
    full_top1_df = summary_df[
        (summary_df["feature_mode"] == feature_mode)
        & (summary_df["metric"] == "oracle_in_bank_match")
        & (summary_df["skip_count"] == 1)
        & (summary_df["bank_size"] == bank_size)
        & (summary_df["model_name"].isin(allowed_models))
    ][["model_name", "mean"]].rename(columns={"mean": "full_oracle_in_bank_match"})

    subset = subset.merge(regret_df, on="model_name", how="left")
    subset = subset.merge(frac_df, on="model_name", how="left")
    subset = subset.merge(top1_df, on="model_name", how="left")
    subset = subset.merge(full_delta_df, on="model_name", how="inner")
    subset = subset.merge(full_regret_df, on="model_name", how="left")
    subset = subset.merge(full_frac_df, on="model_name", how="left")
    subset = subset.merge(full_top1_df, on="model_name", how="left")
    subset = subset.sort_values(
        [
            "full_delta_to_static",
            "full_regret_to_bank",
            "full_fraction_improved",
            "full_oracle_in_bank_match",
            "mean",
            "dev_regret",
            "dev_fraction_improved",
            "dev_oracle_in_bank_match",
        ],
        ascending=[True, True, False, False, True, True, False, False],
    ).reset_index(drop=True)
    best = subset.iloc[0].to_dict()
    best["seed"] = seed
    best["step"] = step
    best["bank_size"] = bank_size
    best["feature_mode"] = feature_mode
    best["selection_csv"] = str(path)
    best["summary_csv"] = str(summary)
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--steps", nargs="+", type=int, required=True)
    parser.add_argument("--bank-sizes", nargs="+", type=int, default=[32, 64])
    parser.add_argument("--feature-modes", nargs="+", default=["attnres"])
    parser.add_argument("--allowed-models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    rows: list[dict] = []
    winners: list[dict] = []
    for seed in args.seeds:
        seed_rows: list[dict] = []
        for step in args.steps:
            for bank_size in args.bank_sizes:
                for feature_mode in args.feature_modes:
                    best = load_best_row(seed, step, bank_size, feature_mode, args.allowed_models)
                    if best is None:
                        continue
                    seed_rows.append(best)
        if not seed_rows:
            continue
        seed_df = pd.DataFrame(seed_rows).sort_values(
            [
                "full_delta_to_static",
                "full_regret_to_bank",
                "full_fraction_improved",
                "full_oracle_in_bank_match",
                "mean",
                "dev_regret",
                "dev_fraction_improved",
                "dev_oracle_in_bank_match",
            ],
            ascending=[True, True, False, False, True, True, False, False],
        )
        seed_df["selected_for_seed"] = False
        seed_df.loc[seed_df.index[0], "selected_for_seed"] = True
        rows.extend(seed_df.to_dict("records"))
        winners.append(seed_df.iloc[0].to_dict())

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)

    if winners:
        winners_path = out_path.with_name(out_path.stem + "_winners.csv")
        pd.DataFrame(winners).to_csv(winners_path, index=False)
        print(pd.DataFrame(winners).to_string(index=False), flush=True)
    else:
        print("no winners found", flush=True)


if __name__ == "__main__":
    main()
