#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


FINAL_SPLITS = ["final_A", "final_B", "final_C"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_sequence_metadata(manifest_dir: Path, metadata_csv: Path) -> pd.DataFrame:
    seq_rows: list[dict] = []
    for split in FINAL_SPLITS:
        manifest_path = manifest_dir / f"v8_ccnews_p256d64_lockbox_{split}.jsonl"
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                seq_rows.append(
                    {
                        "final_split": split,
                        "sequence_idx": int(row["sequence_idx"]),
                        "document_uid": row["document_uid"],
                    }
                )
    seq_df = pd.DataFrame(seq_rows)
    meta_df = pd.read_csv(metadata_csv)
    merged = seq_df.merge(meta_df, on="document_uid", how="left")

    merged["year"] = (
        merged["date"].astype(str).str.extract(r"^(\d{4})", expand=False).fillna("unknown")
    )
    top_domains = merged["domain"].fillna("unknown").value_counts().head(10).index
    merged["domain_group"] = merged["domain"].fillna("unknown").where(
        merged["domain"].fillna("unknown").isin(top_domains),
        "other",
    )
    rank_source = merged["text_char_len"].fillna(merged["text_char_len"].median()).rank(method="first")
    merged["length_bin"] = pd.qcut(rank_source, 4, labels=["q1", "q2", "q3", "q4"])
    return merged


def cvar_tail(series: pd.Series, frac: float, largest: bool) -> float:
    if series.empty:
        return float("nan")
    k = max(1, int(math.ceil(len(series) * frac)))
    ordered = series.sort_values(ascending=not largest)
    return float(ordered.head(k).mean())


def summarize_group(df: pd.DataFrame, subgroup_name: str, subgroup_value: str) -> dict:
    delta = df["actual_delta_to_static"]
    return {
        "subgroup_name": subgroup_name,
        "subgroup_value": subgroup_value,
        "n": int(len(df)),
        "mean_delta_to_static": float(delta.mean()),
        "median_delta_to_static": float(delta.median()),
        "p10_delta_to_static": float(delta.quantile(0.10)),
        "p25_delta_to_static": float(delta.quantile(0.25)),
        "p50_delta_to_static": float(delta.quantile(0.50)),
        "p75_delta_to_static": float(delta.quantile(0.75)),
        "p90_delta_to_static": float(delta.quantile(0.90)),
        "fraction_improved": float(df["improved_over_static"].mean()),
        "frac_worse_gt_001": float((delta > 0.01).mean()),
        "frac_worse_gt_002": float((delta > 0.02).mean()),
        "frac_worse_gt_005": float((delta > 0.05).mean()),
        "frac_better_gt_001": float((delta < -0.01).mean()),
        "frac_better_gt_002": float((delta < -0.02).mean()),
        "frac_better_gt_005": float((delta < -0.05).mean()),
        "mean_positive_delta": float(delta[delta > 0].mean()) if (delta > 0).any() else 0.0,
        "mean_negative_delta": float(delta[delta < 0].mean()) if (delta < 0).any() else 0.0,
        "cvar_worst_10pct": cvar_tail(delta, 0.10, largest=True),
        "cvar_best_10pct": cvar_tail(delta, 0.10, largest=False),
        "mean_regret_to_bank": float(df["delta_to_bank_upper_bound"].mean()),
    }


def plot_histogram(df: pd.DataFrame, output_path: Path) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for seed, group in sorted(df.groupby("seed")):
        ax.hist(group["actual_delta_to_static"], bins=60, alpha=0.35, density=True, label=f"seed{seed}")
    ax.axvline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_title("V9 Risk Profile: Delta-to-Static Histogram by Seed")
    ax.set_xlabel("actual_delta_to_static")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_cdf(df: pd.DataFrame, output_path: Path) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for seed, group in sorted(df.groupby("seed")):
        vals = group["actual_delta_to_static"].sort_values().to_numpy()
        y = (pd.Series(range(1, len(vals) + 1)) / len(vals)).to_numpy()
        ax.plot(vals, y, label=f"seed{seed}")
    ax.axvline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_title("V9 Risk Profile: Empirical CDF by Seed")
    ax.set_xlabel("actual_delta_to_static")
    ax.set_ylabel("cdf")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_tail_bars(summary_df: pd.DataFrame, output_path: Path) -> None:
    if plt is None:
        return
    seed_df = summary_df[summary_df["subgroup_name"] == "seed"].copy()
    if seed_df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(seed_df))
    width = 0.25
    ax.bar([i - width for i in x], seed_df["frac_worse_gt_002"], width=width, label="worse > 0.02")
    ax.bar(x, seed_df["frac_better_gt_002"], width=width, label="better > 0.02")
    ax.bar([i + width for i in x], seed_df["fraction_improved"], width=width, label="fraction improved")
    ax.set_xticks(list(x))
    ax.set_xticklabels(seed_df["subgroup_value"].tolist())
    ax.set_ylim(0, 1)
    ax.set_title("V9 Risk Profile: Tail and Improvement Rates by Seed")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pooled-per-seq",
        default=str(ROOT / "results" / "ccnews_multiseed_multisplit_v8" / "ccnews_v8_locked_pooled_per_sequence.csv"),
    )
    parser.add_argument(
        "--manifest-dir",
        default=str(ROOT / "results" / "lockbox_manifests_v8"),
    )
    parser.add_argument(
        "--metadata-csv",
        default=str(ROOT / "results" / "boundary_analysis_v8" / "v8_ccnews_final_metadata.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "risk_profile_v9"),
    )
    parser.add_argument(
        "--plots-dir",
        default=str(ROOT / "results" / "plots"),
    )
    args = parser.parse_args()

    pooled_df = pd.read_csv(args.pooled_per_seq)
    seq_meta = build_sequence_metadata(Path(args.manifest_dir), Path(args.metadata_csv))
    merged = pooled_df.merge(seq_meta, on=["final_split", "sequence_idx"], how="left")

    output_dir = Path(args.output_dir)
    plots_dir = Path(args.plots_dir)
    ensure_dir(output_dir)
    ensure_dir(plots_dir)

    summary_rows = [summarize_group(merged, "overall", "all")]
    for seed, group in sorted(merged.groupby("seed")):
        summary_rows.append(summarize_group(group, "seed", str(seed)))
    for split, group in sorted(merged.groupby("final_split")):
        summary_rows.append(summarize_group(group, "final_split", str(split)))
    for (seed, split), group in sorted(merged.groupby(["seed", "final_split"])):
        summary_rows.append(summarize_group(group, "seed_final_split", f"{seed}:{split}"))
    for year, group in sorted(merged.groupby("year")):
        summary_rows.append(summarize_group(group, "year", str(year)))
    for length_bin, group in sorted(merged.groupby("length_bin")):
        summary_rows.append(summarize_group(group, "length_bin", str(length_bin)))
    for domain_group, group in sorted(merged.groupby("domain_group")):
        if len(group) < 128:
            continue
        summary_rows.append(summarize_group(group, "domain_group", str(domain_group)))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "risk_profile_summary.csv", index=False)
    merged.to_csv(output_dir / "risk_profile_per_sequence.csv", index=False)
    summary_df[summary_df["subgroup_name"] == "seed"].to_csv(output_dir / "risk_profile_by_seed.csv", index=False)
    summary_df[summary_df["subgroup_name"] == "seed_final_split"].to_csv(output_dir / "risk_profile_seed_split.csv", index=False)

    plot_histogram(merged, plots_dir / "risk_profile_v9_hist_by_seed.png")
    plot_cdf(merged, plots_dir / "risk_profile_v9_cdf_by_seed.png")
    plot_tail_bars(summary_df, plots_dir / "risk_profile_v9_tail_by_seed.png")

    print(summary_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
