#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attnres_routing.analysis import (  # noqa: E402
    bootstrap_mean_ci,
    compute_chunk_utility,
    compute_utility_from_records,
    save_json,
    summarize_prompt_decode_transfer,
)
from attnres_routing.data import DataConfig, prepare_lm_datasets  # noqa: E402
from attnres_routing.model import AttnResConfig, DecoderLM  # noqa: E402
from attnres_routing.utils import ensure_dir, set_seed  # noqa: E402


def load_checkpoint(path: Path, device: torch.device) -> tuple[DecoderLM, dict[str, Any], int]:
    payload = torch.load(path, map_location=device)
    config = payload["config"]
    model = DecoderLM(AttnResConfig.from_dict(config["model"])).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, config, int(payload.get("step", 0))


def init_step0_model(config: dict[str, Any], device: torch.device) -> DecoderLM:
    seed = int(config.get("train", {}).get("seed", 42))
    set_seed(seed)
    model = DecoderLM(AttnResConfig.from_dict(config["model"])).to(device)
    model.eval()
    return model


def collect_sequence_vectors(
    model: DecoderLM,
    val_ds,
    prompt_len: int,
    decode_len: int,
    num_sequences: int,
    num_chunks: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    num_sources = model.config.num_blocks + 1
    for seq_idx in range(min(num_sequences, len(val_ds))):
        tokens = val_ds[seq_idx]["input_ids"]
        if tokens.numel() < prompt_len + decode_len:
            continue
        sample = tokens[: prompt_len + decode_len].unsqueeze(0).to(next(model.parameters()).device)
        with torch.no_grad():
            outputs = model(sample, labels=None, record_mode="full")
        records = outputs["stats"]["records"]
        prompt_corr = compute_chunk_utility(
            records,
            num_sources=num_sources,
            prompt_len=prompt_len,
            num_chunks=num_chunks,
            normalize=True,
            center_uniform=True,
        )
        prompt_raw = compute_chunk_utility(
            records,
            num_sources=num_sources,
            prompt_len=prompt_len,
            num_chunks=num_chunks,
            normalize=True,
            center_uniform=False,
        )
        decode_corr = compute_utility_from_records(
            records,
            num_sources=num_sources,
            token_start=prompt_len,
            token_end=prompt_len + decode_len,
            normalize=True,
            center_uniform=True,
        )
        decode_raw = compute_utility_from_records(
            records,
            num_sources=num_sources,
            token_start=prompt_len,
            token_end=prompt_len + decode_len,
            normalize=True,
            center_uniform=False,
        )
        rows.append(
            {
                "sequence_idx": seq_idx,
                "prompt_corrected": prompt_corr.full_utility,
                "decode_corrected": decode_corr,
                "prompt_raw": prompt_raw.full_utility,
                "decode_raw": decode_raw,
                "chunk_variance_corrected": prompt_corr.chunk_variance,
                "chunk_variance_raw": prompt_raw.chunk_variance,
                "chunk_utilities_corrected": prompt_corr.chunk_utilities,
                "chunk_utilities_raw": prompt_raw.chunk_utilities,
            }
        )
    return rows


def shifted_indices(n: int) -> np.ndarray:
    if n <= 1:
        return np.arange(n)
    return np.roll(np.arange(n), -1)


def random_derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 1:
        return np.arange(n)
    while True:
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm


def permute_prompt_scores(scores: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    permuted = scores.copy()
    permuted[1:] = scores[1:][permutation]
    return permuted


def build_condition_rows(
    vectors: list[dict[str, Any]],
    model_label: str,
    condition: str,
    prompt_key: str,
    decode_key: str,
    permutation: np.ndarray | None = None,
    decode_indices: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not vectors:
        return rows
    if decode_indices is None:
        decode_indices = np.arange(len(vectors))
    for source_pos, decode_pos in enumerate(decode_indices.tolist()):
        source = vectors[source_pos]
        target = vectors[decode_pos]
        prompt_scores = np.asarray(source[prompt_key], dtype=np.float64)
        if permutation is not None:
            prompt_scores = permute_prompt_scores(prompt_scores, permutation)
        decode_scores = np.asarray(target[decode_key], dtype=np.float64)
        metrics = summarize_prompt_decode_transfer(prompt_scores, decode_scores)
        rows.append(
            {
                "model_label": model_label,
                "condition": condition,
                "sequence_idx": int(source["sequence_idx"]),
                "target_sequence_idx": int(target["sequence_idx"]),
                **metrics,
            }
        )
    return rows


def summarize_conditions(
    per_sequence_df: pd.DataFrame,
    metric_columns: list[str],
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = per_sequence_df.groupby(["model_label", "condition"], sort=False)
    for (model_label, condition), group in grouped:
        for metric in metric_columns:
            ci = bootstrap_mean_ci(
                group[metric].to_numpy(),
                num_bootstrap=bootstrap_samples,
                seed=seed,
            )
            rows.append(
                {
                    "model_label": model_label,
                    "condition": condition,
                    "metric": metric,
                    **ci,
                }
            )
    return pd.DataFrame(rows)


def plot_control_summary(summary_df: pd.DataFrame, plot_path: Path, model_label: str) -> None:
    metrics = ["spearman", "kendall", "topk_jaccard_1", "recall_at_3", "ndcg_at_3"]
    condition_order = [
        "matched_corrected",
        "matched_raw",
        "mismatched_corrected",
        "permuted_corrected",
    ]
    subset = summary_df[(summary_df["model_label"] == model_label) & (summary_df["metric"].isin(metrics))]
    if subset.empty:
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 4), sharey=False)
    for ax, metric in zip(axes, metrics):
        metric_df = subset[subset["metric"] == metric].copy()
        metric_df["condition"] = pd.Categorical(metric_df["condition"], categories=condition_order, ordered=True)
        metric_df = metric_df.sort_values("condition")
        x = np.arange(len(metric_df))
        means = metric_df["mean"].to_numpy()
        yerr = np.vstack(
            [
                means - metric_df["ci_low"].to_numpy(),
                metric_df["ci_high"].to_numpy() - means,
            ]
        )
        ax.bar(x, means, color=["#2962ff", "#6d4c41", "#d81b60", "#00897b"][: len(metric_df)])
        ax.errorbar(x, means, yerr=yerr, fmt="none", ecolor="black", capsize=3, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_df["condition"], rotation=35, ha="right")
        ax.set_title(metric)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def plot_learning_stage(summary_df: pd.DataFrame, plot_path: Path) -> None:
    metrics = ["spearman", "kendall", "topk_jaccard_1", "recall_at_3", "ndcg_at_3"]
    subset = summary_df[
        (summary_df["condition"] == "matched_corrected")
        & (summary_df["metric"].isin(metrics))
        & (summary_df["model_label"].isin(["step0_random", "step10_short"]))
    ]
    if subset.empty:
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(3 * len(metrics), 4), sharey=False)
    for ax, metric in zip(axes, metrics):
        metric_df = subset[subset["metric"] == metric].copy()
        metric_df["model_label"] = pd.Categorical(
            metric_df["model_label"],
            categories=["step0_random", "step10_short"],
            ordered=True,
        )
        metric_df = metric_df.sort_values("model_label")
        x = np.arange(len(metric_df))
        means = metric_df["mean"].to_numpy()
        yerr = np.vstack(
            [
                means - metric_df["ci_low"].to_numpy(),
                metric_df["ci_high"].to_numpy() - means,
            ]
        )
        ax.bar(x, means, color=["#9e9d24", "#3949ab"])
        ax.errorbar(x, means, yerr=yerr, fmt="none", ecolor="black", capsize=3, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_df["model_label"], rotation=25, ha="right")
        ax.set_title(metric)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def write_controls_report(
    path: Path,
    reproduction_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    def summary_line(model_label: str, condition: str, metric: str) -> str:
        row = summary_df[
            (summary_df["model_label"] == model_label)
            & (summary_df["condition"] == condition)
            & (summary_df["metric"] == metric)
        ].iloc[0]
        return f"{row['mean']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}]"

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Transfer Controls Report\n\n")
        if not reproduction_df.empty:
            f.write("## Legacy 8-sequence reproduction\n\n")
            f.write(f"- mean Spearman: {reproduction_df['spearman'].mean():.4f}\n")
            f.write(f"- mean Kendall: {reproduction_df['kendall'].mean():.4f}\n")
            f.write(f"- mean top-1 Jaccard: {reproduction_df['topk_jaccard_1'].mean():.4f}\n")
            f.write(f"- mean recall@3: {reproduction_df['recall_at_3'].mean():.4f}\n\n")

        f.write("## Main controls (bootstrap 95% CI)\n\n")
        for metric in ["spearman", "kendall", "topk_jaccard_1", "recall_at_3", "ndcg_at_3"]:
            f.write(f"### {metric}\n\n")
            f.write(f"- step-10 matched corrected: {summary_line('step10_short', 'matched_corrected', metric)}\n")
            f.write(f"- step-10 matched raw: {summary_line('step10_short', 'matched_raw', metric)}\n")
            f.write(f"- step-10 mismatched corrected: {summary_line('step10_short', 'mismatched_corrected', metric)}\n")
            f.write(f"- step-10 permuted corrected: {summary_line('step10_short', 'permuted_corrected', metric)}\n")
            f.write(f"- step-0 matched corrected: {summary_line('step0_random', 'matched_corrected', metric)}\n\n")

        f.write("## Interpretation\n\n")
        f.write("- A viable endogenous-routing story should keep corrected matched transfer well above the mismatched and permuted controls.\n")
        f.write("- If step-0 transfer is already close to step-10, the signal is mostly architectural rather than learned.\n")
        f.write("- The raw-vs-corrected gap quantifies how much the naive utility definition overstates transfer.\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-sequences", type=int, default=64)
    parser.add_argument("--repro-sequences", type=int, default=8)
    parser.add_argument("--prompt-len", type=int, default=256)
    parser.add_argument("--decode-len", type=int, default=64)
    parser.add_argument("--num-chunks", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    trained_model, config, checkpoint_step = load_checkpoint(Path(args.checkpoint), device)
    step0_model = init_step0_model(config, device)
    _, val_ds, _ = prepare_lm_datasets(DataConfig(**config["data"]))

    out_dir = ensure_dir(ROOT / "results" / "controls")
    plot_dir = ensure_dir(ROOT / "results" / "plots")

    trained_vectors = collect_sequence_vectors(
        trained_model,
        val_ds,
        prompt_len=args.prompt_len,
        decode_len=args.decode_len,
        num_sequences=args.num_sequences,
        num_chunks=args.num_chunks,
    )
    step0_vectors = collect_sequence_vectors(
        step0_model,
        val_ds,
        prompt_len=args.prompt_len,
        decode_len=args.decode_len,
        num_sequences=args.num_sequences,
        num_chunks=args.num_chunks,
    )

    if not trained_vectors:
        raise RuntimeError("No valid sequences collected for transfer controls.")

    permutation = rng.permutation(trained_model.config.num_blocks)
    mismatch_idx = random_derangement(len(trained_vectors), rng)

    rows: list[dict[str, Any]] = []
    rows.extend(
        build_condition_rows(
            trained_vectors,
            model_label=f"step{checkpoint_step}_short",
            condition="matched_corrected",
            prompt_key="prompt_corrected",
            decode_key="decode_corrected",
        )
    )
    rows.extend(
        build_condition_rows(
            trained_vectors,
            model_label=f"step{checkpoint_step}_short",
            condition="matched_raw",
            prompt_key="prompt_raw",
            decode_key="decode_raw",
        )
    )
    rows.extend(
        build_condition_rows(
            trained_vectors,
            model_label=f"step{checkpoint_step}_short",
            condition="mismatched_corrected",
            prompt_key="prompt_corrected",
            decode_key="decode_corrected",
            decode_indices=mismatch_idx,
        )
    )
    rows.extend(
        build_condition_rows(
            trained_vectors,
            model_label=f"step{checkpoint_step}_short",
            condition="permuted_corrected",
            prompt_key="prompt_corrected",
            decode_key="decode_corrected",
            permutation=permutation,
        )
    )
    rows.extend(
        build_condition_rows(
            step0_vectors,
            model_label="step0_random",
            condition="matched_corrected",
            prompt_key="prompt_corrected",
            decode_key="decode_corrected",
        )
    )
    rows.extend(
        build_condition_rows(
            step0_vectors,
            model_label="step0_random",
            condition="matched_raw",
            prompt_key="prompt_raw",
            decode_key="decode_raw",
        )
    )

    per_sequence_df = pd.DataFrame(rows)
    metric_columns = [
        "spearman",
        "kendall",
        "topk_jaccard_1",
        "recall_at_1",
        "topk_jaccard_2",
        "recall_at_2",
        "topk_jaccard_3",
        "recall_at_3",
        "ndcg_at_1",
        "ndcg_at_2",
        "ndcg_at_3",
    ]
    summary_df = summarize_conditions(
        per_sequence_df,
        metric_columns=metric_columns,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )

    trained_repro_rows = build_condition_rows(
        trained_vectors[: args.repro_sequences],
        model_label=f"step{checkpoint_step}_short",
        condition="matched_corrected",
        prompt_key="prompt_corrected",
        decode_key="decode_corrected",
    )
    reproduction_df = pd.DataFrame(trained_repro_rows)

    per_sequence_df.to_csv(out_dir / "transfer_controls_per_sequence.csv", index=False)
    summary_df.to_csv(out_dir / "transfer_controls_summary.csv", index=False)
    reproduction_df.to_csv(out_dir / "reproduced_transfer_legacy8.csv", index=False)
    save_json(
        out_dir / "transfer_control_vectors.json",
        {
            "prompt_len": args.prompt_len,
            "decode_len": args.decode_len,
            "num_sequences": args.num_sequences,
            "checkpoint_step": checkpoint_step,
            "trained_vectors": [
                {
                    "sequence_idx": int(row["sequence_idx"]),
                    "prompt_corrected": np.asarray(row["prompt_corrected"]).tolist(),
                    "decode_corrected": np.asarray(row["decode_corrected"]).tolist(),
                    "prompt_raw": np.asarray(row["prompt_raw"]).tolist(),
                    "decode_raw": np.asarray(row["decode_raw"]).tolist(),
                }
                for row in trained_vectors
            ],
        },
    )

    plot_control_summary(summary_df, plot_dir / "controls_step10_short_summary.png", model_label=f"step{checkpoint_step}_short")
    plot_learning_stage(summary_df, plot_dir / "controls_learning_stage.png")

    corrected = per_sequence_df[
        (per_sequence_df["model_label"] == f"step{checkpoint_step}_short")
        & (per_sequence_df["condition"] == "matched_corrected")
    ]["spearman"].to_numpy()
    raw = per_sequence_df[
        (per_sequence_df["model_label"] == f"step{checkpoint_step}_short")
        & (per_sequence_df["condition"] == "matched_raw")
    ]["spearman"].to_numpy()
    plt.figure(figsize=(5, 4))
    plt.scatter(raw, corrected, alpha=0.8)
    min_val = float(min(np.min(raw) if raw.size else 0.0, np.min(corrected) if corrected.size else 0.0))
    max_val = float(max(np.max(raw) if raw.size else 1.0, np.max(corrected) if corrected.size else 1.0))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black", linewidth=1)
    plt.xlabel("raw utility spearman")
    plt.ylabel("corrected utility spearman")
    plt.tight_layout()
    plt.savefig(plot_dir / "controls_raw_vs_corrected_spearman.png", dpi=160)
    plt.close()

    write_controls_report(ROOT / "docs" / "controls_report.md", reproduction_df, summary_df)


if __name__ == "__main__":
    main()
