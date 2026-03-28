#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


SEED_CONFIGS = {
    42: {
        "experiment_dir": ROOT / "results" / "scale24x512_ccnews_attnres_dense_v7",
        "route_best_step": 5500,
        "lm_best_step": 3000,
    },
    43: {
        "experiment_dir": ROOT / "results" / "scale24x512_ccnews_attnres_seed43_v8",
        "route_best_step": 6000,
        "lm_best_step": 3000,
    },
    44: {
        "experiment_dir": ROOT / "results" / "scale24x512_ccnews_attnres_seed44_v8",
        "route_best_step": 3500,
        "lm_best_step": 3000,
    },
}
FAST_STEPS = [2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]
READINESS_WEIGHTS = {
    "oracle_spearman_z": 0.35,
    "loo_spearman_z": 0.50,
    "val_loss_z": 0.20,
    "bank_headroom_z": 1.20,
    "bank_upper_gain_z": 1.00,
    "bank_tail_frac_020_z": 0.75,
    "selector_dev_delta_z": 1.40,
    "selector_dev_regret_z": 1.25,
    "selector_dev_fraction_z": 0.45,
    "selector_dev_top1_z": 0.50,
}
STP_SCALARS = [
    "tube_loss_mean",
    "tube_loss_final",
    "tube_cos_mean",
    "tube_cos_final",
    "tube_stability_gap",
    "norm_b0_b4_mean",
    "norm_b4_final_mean",
    "norm_b0_final_mean",
    "norm_b0_b4_final",
    "norm_b4_final_final",
    "norm_b0_final_final",
    "transition_ratio_mean",
    "transition_ratio_final",
    "mean_final_direction_cos",
]
STP_TARGETS = [
    "actual_delta_to_static",
    "delta_to_bank_upper_bound",
    "improved_over_static",
    "oracle_in_bank_match",
]


def parse_json_vec(value: str | float | int | None) -> np.ndarray:
    if not isinstance(value, str) or not value:
        return np.zeros(0, dtype=np.float32)
    return np.asarray(json.loads(value), dtype=np.float32)


def safe_lookup_metric(df: pd.DataFrame, candidates: list[str]) -> float:
    for metric in candidates:
        subset = df[df["metric"] == metric]
        if not subset.empty:
            return float(subset.iloc[0]["mean"])
    return float("nan")


def zscore(values: np.ndarray, larger_is_better: bool = True) -> np.ndarray:
    values = values.astype(np.float64)
    valid = np.isfinite(values)
    out = np.full_like(values, np.nan, dtype=np.float64)
    if valid.sum() <= 1:
        out[valid] = 0.0
        return out
    signed = values[valid] if larger_is_better else -values[valid]
    std = signed.std()
    if std < 1e-8:
        out[valid] = 0.0
        return out
    out[valid] = (signed - signed.mean()) / std
    return out


def safe_cos(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return float("nan")
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def safe_ratio(num: float, den: float) -> float:
    if not math.isfinite(num) or not math.isfinite(den) or abs(den) <= 1e-12:
        return float("nan")
    return float(num / den)


def safe_corr(values_x: pd.Series, values_y: pd.Series, fn) -> float:
    mask = np.isfinite(values_x.to_numpy(dtype=float)) & np.isfinite(values_y.to_numpy(dtype=float))
    if int(mask.sum()) < 3:
        return float("nan")
    x = values_x.to_numpy(dtype=float)[mask]
    y = values_y.to_numpy(dtype=float)[mask]
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(fn(x, y)[0])


def load_checkpoint_proxy_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    oracles_dir = ROOT / "results" / "oracles"
    bank_dir = ROOT / "results" / "bank_hygiene"
    for seed, config in SEED_CONFIGS.items():
        metrics_path = Path(config["experiment_dir"]) / "metrics.csv"
        metrics_df = pd.read_csv(metrics_path)
        for step in FAST_STEPS:
            tag = f"v8_ccnews_seed{seed}_fast_step{step}_val"
            oracle_path = oracles_dir / f"{tag}_oracle_mask_alignment_summary.csv"
            loo_path = oracles_dir / f"{tag}_leave_one_out_alignment_summary.csv"
            bank_path = bank_dir / f"v8_ccnews_seed{seed}_fast_step{step}_bank_summary.csv"
            sequence_path = oracles_dir / f"{tag}_sequence_features.csv"
            if not oracle_path.exists() or not loo_path.exists() or not bank_path.exists() or not sequence_path.exists():
                continue
            oracle_df = pd.read_csv(oracle_path)
            loo_df = pd.read_csv(loo_path)
            bank_df = pd.read_csv(bank_path)
            sequence_df = pd.read_csv(sequence_path)
            metric_rows = metrics_df[metrics_df["step"] == step]
            bank_subset = bank_df[(bank_df["bank_size"] == 32) & (bank_df["skip_count"] == 1)]
            row = {
                "seed": seed,
                "step": step,
                "tag": tag,
                "checkpoint_role": (
                    "route_best"
                    if step == int(config["route_best_step"])
                    else "lm_best"
                    if step == int(config["lm_best_step"])
                    else "other"
                ),
                "val_loss": float(metric_rows.iloc[-1]["val_loss"]) if not metric_rows.empty else float("nan"),
                "oracle_spearman": safe_lookup_metric(oracle_df, ["spearman", "stability_spearman"]),
                "oracle_kendall": safe_lookup_metric(oracle_df, ["kendall", "stability_kendall"]),
                "oracle_topk_jaccard_1": safe_lookup_metric(oracle_df, ["topk_jaccard_1"]),
                "oracle_recall_at_3": safe_lookup_metric(oracle_df, ["recall_at_3"]),
                "oracle_ndcg_at_3": safe_lookup_metric(oracle_df, ["ndcg_at_3"]),
                "loo_spearman": safe_lookup_metric(loo_df, ["spearman", "stability_spearman"]),
                "loo_kendall": safe_lookup_metric(loo_df, ["kendall", "stability_kendall"]),
                "bank_headroom": float(
                    bank_subset[bank_subset["metric"] == "oracle_headroom_over_calib_global"]["mean"].iloc[0]
                ),
                "bank_upper_gain": float(
                    -bank_subset[bank_subset["metric"] == "delta_bank_to_calib_global"]["mean"].iloc[0]
                ),
                "bank_tail_frac_020": float(
                    bank_subset[bank_subset["metric"] == "fraction_oracle_headroom_gt_0.020"]["mean"].iloc[0]
                ),
                "prompt_depth_entropy_mean": float(sequence_df["prompt_depth_entropy"].mean()),
                "prompt_depth_entropy_std": float(sequence_df["prompt_depth_entropy"].std(ddof=0)),
                "prompt_support_size_mean": float(sequence_df["prompt_support_size"].mean()),
                "selector_dev_delta": float("nan"),
                "selector_dev_regret": float("nan"),
                "selector_dev_fraction": float("nan"),
                "selector_dev_top1": float("nan"),
            }
            rows.append(row)

    df = pd.DataFrame(rows).sort_values(["seed", "step"]).reset_index(drop=True)
    if df.empty:
        return df

    readiness_rows: list[pd.DataFrame] = []
    for seed, group_df in df.groupby("seed"):
        working = group_df.copy()
        components = {
            "oracle_spearman_z": zscore(working["oracle_spearman"].to_numpy(), larger_is_better=True),
            "loo_spearman_z": zscore(working["loo_spearman"].to_numpy(), larger_is_better=True),
            "val_loss_z": zscore(working["val_loss"].to_numpy(), larger_is_better=False),
            "bank_headroom_z": zscore(working["bank_headroom"].to_numpy(), larger_is_better=True),
            "bank_upper_gain_z": zscore(working["bank_upper_gain"].to_numpy(), larger_is_better=True),
            "bank_tail_frac_020_z": zscore(working["bank_tail_frac_020"].to_numpy(), larger_is_better=True),
            "selector_dev_delta_z": zscore(working["selector_dev_delta"].to_numpy(), larger_is_better=False),
            "selector_dev_regret_z": zscore(working["selector_dev_regret"].to_numpy(), larger_is_better=False),
            "selector_dev_fraction_z": zscore(working["selector_dev_fraction"].to_numpy(), larger_is_better=True),
            "selector_dev_top1_z": zscore(working["selector_dev_top1"].to_numpy(), larger_is_better=True),
        }
        for name, values in components.items():
            working[name] = values

        scores = []
        for _, row in working.iterrows():
            total = 0.0
            total_weight = 0.0
            for key, weight in READINESS_WEIGHTS.items():
                value = row.get(key, np.nan)
                if pd.notna(value):
                    total += weight * float(value)
                    total_weight += weight
            scores.append(total / max(total_weight, 1e-8))
        working["routing_readiness_v4_proxy"] = scores
        readiness_rows.append(working)
    return pd.concat(readiness_rows, ignore_index=True)


def compute_stp_features(hidden_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in hidden_df.iterrows():
        b0_mean = parse_json_vec(row.get("block_0_mean_json"))
        b4_mean = parse_json_vec(row.get("block_4_mean_json"))
        final_mean = parse_json_vec(row.get("final_mean_json"))
        b0_final = parse_json_vec(row.get("block_0_final_json"))
        b4_final = parse_json_vec(row.get("block_4_final_json"))
        final_final = parse_json_vec(row.get("final_final_json"))

        mean_early_mid = b4_mean - b0_mean
        mean_mid_late = final_mean - b4_mean
        mean_early_late = final_mean - b0_mean
        final_early_mid = b4_final - b0_final
        final_mid_late = final_final - b4_final
        final_early_late = final_final - b0_final

        tube_cos_mean = safe_cos(mean_mid_late, mean_early_mid)
        tube_cos_final = safe_cos(final_mid_late, final_early_mid)
        row_out = {
            "sequence_idx": int(row["sequence_idx"]),
            "document_idx": int(row["document_idx"]) if "document_idx" in row else int(row["sequence_idx"]),
            "window_idx": int(row.get("window_idx", 0)),
            "prompt_surprisal_mean": float(row.get("prompt_surprisal_mean", np.nan)),
            "prompt_surprisal_std": float(row.get("prompt_surprisal_std", np.nan)),
            "prompt_surprisal_max": float(row.get("prompt_surprisal_max", np.nan)),
            "prompt_ppl": float(row.get("prompt_ppl", np.nan)),
            "unique_token_ratio": float(row.get("unique_token_ratio", np.nan)),
            "adjacent_repeat_fraction": float(row.get("adjacent_repeat_fraction", np.nan)),
            "max_repeat_run_fraction": float(row.get("max_repeat_run_fraction", np.nan)),
            "tube_cos_mean": tube_cos_mean,
            "tube_cos_final": tube_cos_final,
            "tube_loss_mean": float(1.0 - tube_cos_mean) if math.isfinite(tube_cos_mean) else float("nan"),
            "tube_loss_final": float(1.0 - tube_cos_final) if math.isfinite(tube_cos_final) else float("nan"),
            "tube_stability_gap": (
                abs(float((1.0 - tube_cos_mean) - (1.0 - tube_cos_final)))
                if math.isfinite(tube_cos_mean) and math.isfinite(tube_cos_final)
                else float("nan")
            ),
            "norm_b0_b4_mean": float(np.linalg.norm(mean_early_mid)),
            "norm_b4_final_mean": float(np.linalg.norm(mean_mid_late)),
            "norm_b0_final_mean": float(np.linalg.norm(mean_early_late)),
            "norm_b0_b4_final": float(np.linalg.norm(final_early_mid)),
            "norm_b4_final_final": float(np.linalg.norm(final_mid_late)),
            "norm_b0_final_final": float(np.linalg.norm(final_early_late)),
            "transition_ratio_mean": safe_ratio(
                float(np.linalg.norm(mean_mid_late)),
                float(np.linalg.norm(mean_early_mid)),
            ),
            "transition_ratio_final": safe_ratio(
                float(np.linalg.norm(final_mid_late)),
                float(np.linalg.norm(final_early_mid)),
            ),
            "mean_final_direction_cos": safe_cos(mean_early_late, final_early_late),
        }
        rows.append(row_out)
    return pd.DataFrame(rows)


def build_locked_sequence_table(winners_csv: Path) -> pd.DataFrame:
    winners_df = pd.read_csv(winners_csv)
    rows: list[pd.DataFrame] = []
    for _, winner in winners_df.iterrows():
        seed = int(winner["seed"])
        if seed not in {43, 44}:
            continue
        step = int(winner["step"])
        bank_size = int(winner["bank_size"])
        winner_model = str(winner["model_name"])
        feature_mode = str(winner["feature_mode"])
        for split in ["final_A", "final_B", "final_C"]:
            hidden_path = ROOT / "results" / "rich_features" / f"v8_ccnews_seed{seed}_locked_{split}_hidden_prompt_features.csv"
            routing_prefix = f"v8_locked_seed{seed}_{split}_step{step}_b{bank_size}_{winner_model}_{feature_mode}"
            routing_path = ROOT / "results" / "regret_reduction_v8" / f"{routing_prefix}_{feature_mode}_per_sequence.csv"
            if not hidden_path.exists() or not routing_path.exists():
                continue
            hidden_df = pd.read_csv(hidden_path)
            hidden_df = compute_stp_features(hidden_df)
            routing_df = pd.read_csv(routing_path)
            routing_df = routing_df[
                (routing_df["feature_mode"] == feature_mode)
                & (routing_df["model_name"] == winner_model)
                & (routing_df["bank_size"] == bank_size)
                & (routing_df["skip_count"] == 1)
            ].copy()
            merged = routing_df.merge(hidden_df, on="sequence_idx", how="inner")
            merged["seed"] = seed
            merged["final_split"] = split
            merged["route_model_name"] = winner_model
            merged["route_step"] = step
            rows.append(merged)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def correlation_rows(df: pd.DataFrame, scope: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for stp_metric in STP_SCALARS:
        for target in STP_TARGETS:
            if stp_metric not in df.columns or target not in df.columns:
                continue
            subset = df[[stp_metric, target]].dropna()
            if subset.empty:
                continue
            rows.append(
                {
                    "scope": scope,
                    "stp_metric": stp_metric,
                    "target": target,
                    "n": int(len(subset)),
                    "spearman_r": safe_corr(subset[stp_metric], subset[target], spearmanr),
                    "pearson_r": safe_corr(subset[stp_metric], subset[target], pearsonr),
                    "mean_stp_metric": float(subset[stp_metric].mean()),
                    "mean_target": float(subset[target].mean()),
                }
            )
    return rows


def pair_rows(proxy_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed, group_df in proxy_df.groupby("seed"):
        route_row = group_df[group_df["checkpoint_role"] == "route_best"]
        lm_row = group_df[group_df["checkpoint_role"] == "lm_best"]
        if route_row.empty or lm_row.empty:
            continue
        route_row = route_row.iloc[0]
        lm_row = lm_row.iloc[0]
        rows.append(
            {
                "seed": seed,
                "route_best_step": int(route_row["step"]),
                "lm_best_step": int(lm_row["step"]),
                "route_readiness_proxy": float(route_row["routing_readiness_v4_proxy"]),
                "lm_readiness_proxy": float(lm_row["routing_readiness_v4_proxy"]),
                "route_minus_lm_readiness_proxy": float(route_row["routing_readiness_v4_proxy"] - lm_row["routing_readiness_v4_proxy"]),
                "route_minus_lm_bank_headroom": float(route_row["bank_headroom"] - lm_row["bank_headroom"]),
                "route_minus_lm_bank_upper_gain": float(route_row["bank_upper_gain"] - lm_row["bank_upper_gain"]),
                "route_minus_lm_loo_spearman": float(route_row["loo_spearman"] - lm_row["loo_spearman"]),
                "route_minus_lm_oracle_spearman": float(route_row["oracle_spearman"] - lm_row["oracle_spearman"]),
                "route_minus_lm_prompt_entropy": float(route_row["prompt_depth_entropy_mean"] - lm_row["prompt_depth_entropy_mean"]),
                "route_minus_lm_val_loss": float(route_row["val_loss"] - lm_row["val_loss"]),
            }
        )
    return pd.DataFrame(rows)


def plot_checkpoint_proxy(proxy_df: pd.DataFrame, plot_dir: Path) -> None:
    for metric in ["routing_readiness_v4_proxy", "bank_headroom", "loo_spearman"]:
        plt.figure(figsize=(7, 4))
        for seed, group_df in proxy_df.groupby("seed"):
            plt.plot(group_df["step"], group_df[metric], marker="o", label=f"seed{seed}")
        plt.xlabel("step")
        plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"stp_offline_v9_checkpoint_{metric}.png", dpi=160)
        plt.close()


def plot_locked_sequence(stp_df: pd.DataFrame, plot_dir: Path) -> None:
    if stp_df.empty:
        return
    plt.figure(figsize=(6, 4))
    for seed, group_df in stp_df.groupby("seed"):
        plt.scatter(
            group_df["tube_loss_final"],
            group_df["actual_delta_to_static"],
            s=8,
            alpha=0.25,
            label=f"seed{seed}",
        )
    plt.xlabel("tube_loss_final")
    plt.ylabel("actual_delta_to_static")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "stp_offline_v9_tube_loss_vs_delta.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 4))
    for seed, group_df in stp_df.groupby("seed"):
        plt.scatter(
            group_df["tube_loss_final"],
            group_df["delta_to_bank_upper_bound"],
            s=8,
            alpha=0.25,
            label=f"seed{seed}",
        )
    plt.xlabel("tube_loss_final")
    plt.ylabel("delta_to_bank_upper_bound")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "stp_offline_v9_tube_loss_vs_regret.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 4))
    data = [
        stp_df.loc[stp_df["improved_over_static"] == 0, "tube_loss_final"].dropna().to_numpy(),
        stp_df.loc[stp_df["improved_over_static"] == 1, "tube_loss_final"].dropna().to_numpy(),
    ]
    plt.boxplot(data, tick_labels=["not_improved", "improved"])
    plt.ylabel("tube_loss_final")
    plt.tight_layout()
    plt.savefig(plot_dir / "stp_offline_v9_tube_loss_by_improvement.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--winners-csv",
        default=str(ROOT / "results" / "ccnews_multiseed_multisplit_v8" / "v8_ccnews_dev_frozen_selection_winners.csv"),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--plot-dir", default=str(ROOT / "results" / "plots"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    plot_dir = Path(args.plot_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    proxy_df = load_checkpoint_proxy_rows()
    proxy_df.to_csv(output_dir / "stp_checkpoint_proxy_metrics_v9.csv", index=False)
    if not proxy_df.empty:
        pair_df = pair_rows(proxy_df)
        pair_df.to_csv(output_dir / "stp_checkpoint_proxy_pairs_v9.csv", index=False)
        plot_checkpoint_proxy(proxy_df, plot_dir)

    locked_df = build_locked_sequence_table(Path(args.winners_csv))
    locked_df.to_csv(output_dir / "stp_lite_locked_sequence_features_v9.csv", index=False)

    corr_rows: list[dict[str, object]] = []
    if not locked_df.empty:
        corr_rows.extend(correlation_rows(locked_df, "overall"))
        for seed, group_df in locked_df.groupby("seed"):
            corr_rows.extend(correlation_rows(group_df, f"seed{seed}"))
        for split, group_df in locked_df.groupby("final_split"):
            corr_rows.extend(correlation_rows(group_df, split))
        plot_locked_sequence(locked_df, plot_dir)

        split_summary = (
            locked_df.groupby(["seed", "final_split"], as_index=False)[
                STP_SCALARS + STP_TARGETS
            ]
            .mean(numeric_only=True)
            .sort_values(["seed", "final_split"])
            .reset_index(drop=True)
        )
        split_summary.to_csv(output_dir / "stp_lite_locked_split_summary_v9.csv", index=False)

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(output_dir / "stp_lite_locked_correlations_v9.csv", index=False)

    print(proxy_df.to_string(index=False), flush=True)
    if not corr_df.empty:
        print(corr_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
