#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attnres_routing.analysis import bootstrap_mean_ci  # noqa: E402


DEFAULT_MODEL_NAMES = ["hgb_pair", "rf_pair", "retrieval_rerank_top4"]
STANDARD_HIDDEN_PATTERN = "v8_necessity_standard_seed{seed}_{split}_step{step}_b{bank_size}_hidden"


def parse_seed_step_overrides(items: list[str] | None) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for item in items or []:
        seed_text, step_text = item.split(":", 1)
        mapping[int(seed_text)] = int(step_text)
    return mapping


def parse_step_from_name(path: Path) -> int | None:
    match = re.search(r"_step(\d+)_", path.name)
    return int(match.group(1)) if match else None


def existing_standard_step(seed: int, split: str, bank_size: int, standard_dir: Path, prefix_template: str) -> int | None:
    summary_pattern = f"{prefix_template.format(seed=seed, split=split, step='*', bank_size=bank_size)}_summary.csv"
    matches = sorted(standard_dir.glob(summary_pattern))
    if not matches:
        return None
    step_values = [parse_step_from_name(path) for path in matches]
    step_values = [step for step in step_values if step is not None]
    return sorted(set(step_values))[0] if step_values else None


def summary_metric(
    path: Path,
    feature_mode: str,
    model_name: str,
    bank_size: int,
    skip_count: int,
    metric: str,
) -> pd.Series | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    subset = df[
        (df["feature_mode"] == feature_mode)
        & (df["model_name"] == model_name)
        & (df["bank_size"] == bank_size)
        & (df["skip_count"] == skip_count)
        & (df["metric"] == metric)
    ]
    if subset.empty:
        return None
    return subset.iloc[0]


def per_sequence_rows(
    path: Path,
    feature_mode: str,
    model_name: str,
    bank_size: int,
    skip_count: int,
) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df[
        (df["feature_mode"] == feature_mode)
        & (df["model_name"] == model_name)
        & (df["bank_size"] == bank_size)
        & (df["skip_count"] == skip_count)
    ].copy()


def build_family_sources(
    seed: int,
    split: str,
    bank_size: int,
    winner_model: str,
    winner_feature_mode: str,
    dynamic_step: int,
    standard_step: int | None,
    standard_dir: Path,
    standard_prefix_template: str,
) -> list[dict[str, object]]:
    regret_dir = ROOT / "results" / "regret_reduction_v8"
    dynamic_prefix = f"v8_locked_seed{seed}_{split}_step{dynamic_step}_b{bank_size}_{winner_model}_{winner_feature_mode}"
    families = [
        {
            "family": "attnres_dynamic",
            "feature_mode": winner_feature_mode,
            "summary_path": regret_dir / f"{dynamic_prefix}_{winner_feature_mode}_summary.csv",
            "per_sequence_path": regret_dir / f"{dynamic_prefix}_{winner_feature_mode}_per_sequence.csv",
            "step": dynamic_step,
            "step_policy": "winner_step",
        },
        {
            "family": "attnres_hidden",
            "feature_mode": "hidden",
            "summary_path": regret_dir / f"v8_necessity_attnres_seed{seed}_{split}_step{dynamic_step}_b{bank_size}_hidden_summary.csv",
            "per_sequence_path": regret_dir / f"v8_necessity_attnres_seed{seed}_{split}_step{dynamic_step}_b{bank_size}_hidden_per_sequence.csv",
            "step": dynamic_step,
            "step_policy": "winner_step",
        },
    ]
    if standard_step is not None:
        standard_prefix = standard_prefix_template.format(
            seed=seed,
            split=split,
            step=standard_step,
            bank_size=bank_size,
        )
        families.append(
            {
                "family": "standard_hidden",
                "feature_mode": "hidden",
                "summary_path": standard_dir / f"{standard_prefix}_summary.csv",
                "per_sequence_path": standard_dir / f"{standard_prefix}_per_sequence.csv",
                "step": standard_step,
                "step_policy": "standard_step",
            }
        )
    return families


def pooled_metric_rows(group_df: pd.DataFrame, family: str, model_name: str) -> list[dict[str, object]]:
    metrics = {
        "delta_to_static": group_df["actual_delta_to_static"].to_numpy(),
        "regret_to_bank": group_df["delta_to_bank_upper_bound"].to_numpy(),
        "fraction_improved": group_df["improved_over_static"].to_numpy(),
        "oracle_in_bank_match": group_df["oracle_in_bank_match"].to_numpy(),
    }
    rows: list[dict[str, object]] = []
    for metric, values in metrics.items():
        summary = bootstrap_mean_ci(values)
        rows.append(
            {
                "family": family,
                "model_name": model_name,
                "metric": metric,
                "mean": float(summary["mean"]),
                "ci_low": float(summary["ci_low"]),
                "ci_high": float(summary["ci_high"]),
                "n": int(summary["n"]),
            }
        )
    return rows


def plot_pooled_summary(pooled_summary_df: pd.DataFrame) -> None:
    plot_dir = ROOT / "results" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for metric in ["delta_to_static", "fraction_improved"]:
        subset = pooled_summary_df[pooled_summary_df["metric"] == metric].copy()
        if subset.empty:
            continue
        pivot = subset.pivot(index="model_name", columns="family", values="mean").reindex(DEFAULT_MODEL_NAMES)
        ax = pivot.plot(kind="bar", figsize=(8, 4))
        ax.set_ylabel(metric)
        ax.set_xlabel("model_name")
        ax.set_title(f"Matched necessity pooled {metric}")
        plt.tight_layout()
        plt.savefig(plot_dir / f"ccnews_matched_necessity_v9_{metric}.png", dpi=160)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--winners-csv",
        default=str(ROOT / "results" / "ccnews_multiseed_multisplit_v8" / "v8_ccnews_dev_frozen_selection_winners.csv"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[43, 44])
    parser.add_argument("--final-splits", nargs="+", default=["final_A", "final_B", "final_C"])
    parser.add_argument("--model-names", nargs="+", default=DEFAULT_MODEL_NAMES)
    parser.add_argument("--skip-count", type=int, default=1)
    parser.add_argument("--dynamic-step-overrides", nargs="*", default=None)
    parser.add_argument("--standard-step-overrides", nargs="*", default=None)
    parser.add_argument("--standard-step-source", choices=["existing", "explicit_only"], default="existing")
    parser.add_argument(
        "--standard-dir",
        default=str(ROOT / "results" / "regret_reduction_v8"),
    )
    parser.add_argument(
        "--standard-prefix-template",
        default=STANDARD_HIDDEN_PATTERN,
    )
    parser.add_argument("--require-complete-triples", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    winners_df = pd.read_csv(args.winners_csv)
    winners_df = winners_df[winners_df["seed"].isin(args.seeds)].copy()
    winners_by_seed = {int(row["seed"]): row for _, row in winners_df.iterrows()}
    dynamic_overrides = parse_seed_step_overrides(args.dynamic_step_overrides)
    standard_overrides = parse_seed_step_overrides(args.standard_step_overrides)
    standard_dir = Path(args.standard_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_seed_rows: list[dict[str, object]] = []
    pooled_rows: list[pd.DataFrame] = []
    missing_rows: list[dict[str, object]] = []

    for seed in args.seeds:
        winner = winners_by_seed.get(seed)
        if winner is None:
            missing_rows.append(
                {
                    "seed": seed,
                    "final_split": None,
                    "model_name": None,
                    "family": "winner_lookup",
                    "reason": "missing_winner_row",
                    "path": None,
                }
            )
            continue
        bank_size = int(winner["bank_size"])
        winner_model = str(winner["model_name"])
        winner_feature_mode = str(winner["feature_mode"])
        dynamic_step = dynamic_overrides.get(seed, int(winner["step"]))

        for split in args.final_splits:
            if seed in standard_overrides:
                standard_step = standard_overrides[seed]
            elif args.standard_step_source == "existing":
                standard_step = existing_standard_step(
                    seed=seed,
                    split=split,
                    bank_size=bank_size,
                    standard_dir=standard_dir,
                    prefix_template=args.standard_prefix_template,
                )
            else:
                standard_step = None

            family_sources = build_family_sources(
                seed=seed,
                split=split,
                bank_size=bank_size,
                winner_model=winner_model,
                winner_feature_mode=winner_feature_mode,
                dynamic_step=dynamic_step,
                standard_step=standard_step,
                standard_dir=standard_dir,
                standard_prefix_template=args.standard_prefix_template,
            )

            for family_info in family_sources:
                family = str(family_info["family"])
                feature_mode = str(family_info["feature_mode"])
                summary_path = Path(family_info["summary_path"])
                per_sequence_path = Path(family_info["per_sequence_path"])
                family_step = int(family_info["step"])
                step_policy = str(family_info["step_policy"])

                if not summary_path.exists() or not per_sequence_path.exists():
                    missing_rows.append(
                        {
                            "seed": seed,
                            "final_split": split,
                            "model_name": None,
                            "family": family,
                            "reason": "missing_artifact",
                            "path": str(summary_path if not summary_path.exists() else per_sequence_path),
                        }
                    )
                    continue

                for model_name in args.model_names:
                    delta_row = summary_metric(
                        summary_path,
                        feature_mode,
                        model_name,
                        bank_size,
                        args.skip_count,
                        "delta_to_static",
                    )
                    regret_row = summary_metric(
                        summary_path,
                        feature_mode,
                        model_name,
                        bank_size,
                        args.skip_count,
                        "delta_to_bank_upper_bound",
                    )
                    fraction_row = summary_metric(
                        summary_path,
                        feature_mode,
                        model_name,
                        bank_size,
                        args.skip_count,
                        "fraction_improved",
                    )
                    top1_row = summary_metric(
                        summary_path,
                        feature_mode,
                        model_name,
                        bank_size,
                        args.skip_count,
                        "oracle_in_bank_match",
                    )
                    if any(row is None for row in [delta_row, regret_row, fraction_row, top1_row]):
                        missing_rows.append(
                            {
                                "seed": seed,
                                "final_split": split,
                                "model_name": model_name,
                                "family": family,
                                "reason": "missing_model_rows",
                                "path": str(summary_path),
                            }
                        )
                        continue

                    per_seed_rows.append(
                        {
                            "seed": seed,
                            "final_split": split,
                            "family": family,
                            "model_name": model_name,
                            "feature_mode": feature_mode,
                            "bank_size": bank_size,
                            "skip_count": args.skip_count,
                            "winner_model_prefix": winner_model,
                            "step": family_step,
                            "step_policy": step_policy,
                            "delta_to_static": float(delta_row["mean"]),
                            "delta_ci_low": float(delta_row["ci_low"]),
                            "delta_ci_high": float(delta_row["ci_high"]),
                            "regret_to_bank": float(regret_row["mean"]),
                            "regret_ci_low": float(regret_row["ci_low"]),
                            "regret_ci_high": float(regret_row["ci_high"]),
                            "fraction_improved": float(fraction_row["mean"]),
                            "fraction_ci_low": float(fraction_row["ci_low"]),
                            "fraction_ci_high": float(fraction_row["ci_high"]),
                            "oracle_in_bank_match": float(top1_row["mean"]),
                            "top1_ci_low": float(top1_row["ci_low"]),
                            "top1_ci_high": float(top1_row["ci_high"]),
                            "summary_path": str(summary_path),
                            "per_sequence_path": str(per_sequence_path),
                        }
                    )

                    per_df = per_sequence_rows(
                        per_sequence_path,
                        feature_mode,
                        model_name,
                        bank_size,
                        args.skip_count,
                    )
                    if per_df.empty:
                        missing_rows.append(
                            {
                                "seed": seed,
                                "final_split": split,
                                "model_name": model_name,
                                "family": family,
                                "reason": "empty_per_sequence_filter",
                                "path": str(per_sequence_path),
                            }
                        )
                        continue
                    per_df["seed"] = seed
                    per_df["final_split"] = split
                    per_df["family"] = family
                    per_df["step"] = family_step
                    per_df["step_policy"] = step_policy
                    pooled_rows.append(per_df)

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_df.to_csv(output_dir / "ccnews_matched_necessity_v9_per_seed_split.csv", index=False)

    missing_df = pd.DataFrame(missing_rows)
    missing_df.to_csv(output_dir / "ccnews_matched_necessity_v9_missing_artifacts.csv", index=False)

    if per_seed_df.empty:
        print("No matched-necessity rows were found.", flush=True)
        return

    coverage_df = (
        per_seed_df.groupby(["seed", "final_split", "model_name"], as_index=False)
        .agg(
            family_count=("family", "nunique"),
            families=("family", lambda items: ",".join(sorted(set(str(x) for x in items)))),
        )
        .sort_values(["seed", "final_split", "model_name"])
        .reset_index(drop=True)
    )
    coverage_df["is_complete_triple"] = coverage_df["family_count"] == 3
    coverage_df.to_csv(output_dir / "ccnews_matched_necessity_v9_coverage.csv", index=False)

    pooled_df = pd.concat(pooled_rows, ignore_index=True) if pooled_rows else pd.DataFrame()
    if not pooled_df.empty:
        pooled_df = pooled_df.merge(
            coverage_df[["seed", "final_split", "model_name", "is_complete_triple"]],
            on=["seed", "final_split", "model_name"],
            how="left",
        )
        pooled_df.to_csv(output_dir / "ccnews_matched_necessity_v9_pooled_per_sequence.csv", index=False)
        if args.require_complete_triples:
            pooled_use_df = pooled_df[pooled_df["is_complete_triple"].fillna(False)].copy()
        else:
            pooled_use_df = pooled_df.copy()

        pooled_summary_rows: list[dict[str, object]] = []
        if not pooled_use_df.empty:
            for (family, model_name), group_df in pooled_use_df.groupby(["family", "model_name"]):
                pooled_summary_rows.extend(pooled_metric_rows(group_df, family, model_name))
        pooled_summary_df = pd.DataFrame(pooled_summary_rows)
        pooled_summary_df.to_csv(output_dir / "ccnews_matched_necessity_v9_pooled_summary.csv", index=False)
        if not pooled_summary_df.empty:
            plot_pooled_summary(pooled_summary_df)

    wide_df = per_seed_df.pivot_table(
        index=["seed", "final_split", "model_name"],
        columns="family",
        values=["delta_to_static", "regret_to_bank", "fraction_improved", "oracle_in_bank_match"],
    )
    if not wide_df.empty:
        wide_df.columns = [f"{metric}__{family}" for metric, family in wide_df.columns]
        wide_df = wide_df.reset_index()
        if "delta_to_static__attnres_dynamic" in wide_df and "delta_to_static__attnres_hidden" in wide_df:
            wide_df["delta_gap_hidden_minus_dynamic"] = (
                wide_df["delta_to_static__attnres_hidden"] - wide_df["delta_to_static__attnres_dynamic"]
            )
        if "delta_to_static__standard_hidden" in wide_df and "delta_to_static__attnres_dynamic" in wide_df:
            wide_df["delta_gap_standard_minus_dynamic"] = (
                wide_df["delta_to_static__standard_hidden"] - wide_df["delta_to_static__attnres_dynamic"]
            )
        wide_df.to_csv(output_dir / "ccnews_matched_necessity_v9_pairwise_gaps.csv", index=False)

    print(per_seed_df.to_string(index=False), flush=True)
    print(coverage_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
