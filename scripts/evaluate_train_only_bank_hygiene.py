#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attnres_routing.analysis import bootstrap_mean_ci  # noqa: E402


def parse_mask_id(mask_id: str, num_blocks: int) -> np.ndarray:
    mask = np.zeros(num_blocks, dtype=np.bool_)
    _, kept = mask_id.split(":", 1)
    if kept.strip():
        for token in kept.split(","):
            mask[int(token) - 1] = True
    return mask


def swap_distance(mask_a: np.ndarray, mask_b: np.ndarray) -> int:
    return int(np.logical_xor(mask_a, mask_b).sum() // 2)


def farthest_first_masks(mask_counts: Counter[str], num_blocks: int, limit: int) -> list[str]:
    if not mask_counts or limit <= 0:
        return []
    remaining = list(mask_counts.keys())
    ordered = [remaining.pop(np.argmax([mask_counts[mask_id] for mask_id in remaining]))]
    while remaining and len(ordered) < limit:
        best_idx = None
        best_score = None
        for idx, candidate in enumerate(remaining):
            candidate_mask = parse_mask_id(candidate, num_blocks)
            min_dist = min(
                swap_distance(candidate_mask, parse_mask_id(existing, num_blocks))
                for existing in ordered
            )
            score = (min_dist, mask_counts[candidate])
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx
        ordered.append(remaining.pop(int(best_idx)))
    return ordered


def bootstrap_summary(values: np.ndarray, seed: int) -> dict[str, float]:
    return bootstrap_mean_ci(values.astype(np.float64), seed=seed)


def concat_csvs(base_dir: Path, tags: list[str], suffix: str) -> pd.DataFrame:
    frames = [pd.read_csv(base_dir / f"{tag}_{suffix}.csv") for tag in tags]
    return pd.concat(frames, ignore_index=True)


def build_bank_order(
    exhaustive_df: pd.DataFrame,
    num_blocks: int,
    top_global: int,
    top_oracle: int,
    diverse_oracle: int,
) -> tuple[dict[int, list[tuple[str, str]]], dict[int, str], dict[int, dict[str, float]], dict[int, Counter[str]]]:
    bank_order_by_budget: dict[int, list[tuple[str, str]]] = {}
    global_static_by_budget: dict[int, str] = {}
    global_mean_lookup_by_budget: dict[int, dict[str, float]] = {}
    oracle_counter_by_budget: dict[int, Counter[str]] = {}

    for skip_count in sorted(exhaustive_df["skip_count"].unique().tolist()):
        subset = exhaustive_df[exhaustive_df["skip_count"] == skip_count].copy()
        global_means = (
            subset.groupby("mask_id", as_index=False)["continuation_loss"]
            .mean()
            .sort_values("continuation_loss")
            .reset_index(drop=True)
        )
        global_static = str(global_means.iloc[0]["mask_id"])
        global_static_mask = parse_mask_id(global_static, num_blocks)

        oracle_counter: Counter[str] = Counter()
        for _seq_id, group in subset.groupby("sequence_idx"):
            oracle_mask = str(group.sort_values("continuation_loss").iloc[0]["mask_id"])
            oracle_counter[oracle_mask] += 1

        bank_order: list[tuple[str, str]] = [(global_static, "calib_global_static")]
        for mask_id in global_means["mask_id"].head(top_global).tolist():
            bank_order.append((str(mask_id), "calib_top_global"))
        for mask_id, _count in oracle_counter.most_common(top_oracle):
            bank_order.append((str(mask_id), "calib_oracle_freq"))

        all_masks = global_means["mask_id"].tolist()
        for max_swaps in [1, 2]:
            for mask_id in all_masks:
                dist = swap_distance(parse_mask_id(str(mask_id), num_blocks), global_static_mask)
                if dist == max_swaps:
                    bank_order.append((str(mask_id), f"calib_swap_{max_swaps}"))

        for mask_id in farthest_first_masks(oracle_counter, num_blocks, diverse_oracle):
            bank_order.append((str(mask_id), "calib_oracle_diverse"))

        bank_order_by_budget[skip_count] = bank_order
        global_static_by_budget[skip_count] = global_static
        global_mean_lookup_by_budget[skip_count] = {
            str(mask_id): float(loss)
            for mask_id, loss in zip(global_means["mask_id"].tolist(), global_means["continuation_loss"].tolist())
        }
        oracle_counter_by_budget[skip_count] = oracle_counter
    return (
        bank_order_by_budget,
        global_static_by_budget,
        global_mean_lookup_by_budget,
        oracle_counter_by_budget,
    )


def truncate_bank(bank_order: list[tuple[str, str]], bank_size: int) -> tuple[list[str], dict[str, list[str]]]:
    bank_ids: list[str] = []
    reasons: dict[str, list[str]] = defaultdict(list)
    for mask_id, reason in bank_order:
        if mask_id not in bank_ids and len(bank_ids) < bank_size:
            bank_ids.append(mask_id)
        if mask_id in bank_ids:
            reasons[mask_id].append(reason)
    return bank_ids, reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib-tags", type=str, nargs="+", required=True)
    parser.add_argument("--eval-tags", type=str, nargs="+", required=True)
    parser.add_argument("--output-tag", type=str, required=True)
    parser.add_argument("--bank-sizes", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--top-global", type=int, default=12)
    parser.add_argument("--top-oracle", type=int, default=12)
    parser.add_argument("--diverse-oracle", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    bank_hygiene_dir = ROOT / "results" / "bank_hygiene"
    bank_hygiene_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = ROOT / "results" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    oracles_dir = ROOT / "results" / "oracles"

    calib_masks = concat_csvs(oracles_dir, args.calib_tags, "exhaustive_mask_losses")
    calib_features = concat_csvs(oracles_dir, args.calib_tags, "sequence_features")
    eval_masks = concat_csvs(oracles_dir, args.eval_tags, "exhaustive_mask_losses")
    eval_features = concat_csvs(oracles_dir, args.eval_tags, "sequence_features")

    num_blocks = len(json.loads(calib_features.iloc[0]["prompt_scores_json"])) - 1
    bank_sizes = sorted({int(size) for size in args.bank_sizes if int(size) > 0})

    (
        bank_order_by_budget,
        calib_global_static_by_budget,
        calib_global_mean_lookup_by_budget,
        calib_oracle_counter_by_budget,
    ) = build_bank_order(
        exhaustive_df=calib_masks[calib_masks["method"] == "exhaustive_mask"].copy(),
        num_blocks=num_blocks,
        top_global=args.top_global,
        top_oracle=args.top_oracle,
        diverse_oracle=args.diverse_oracle,
    )

    eval_exhaustive = eval_masks[eval_masks["method"] == "exhaustive_mask"].copy()
    candidate_rows: list[dict[str, Any]] = []
    per_sequence_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for skip_count in sorted(eval_exhaustive["skip_count"].unique().tolist()):
        eval_subset = eval_exhaustive[eval_exhaustive["skip_count"] == skip_count].copy()
        eval_pivot = eval_subset.pivot(index="sequence_idx", columns="mask_id", values="continuation_loss")
        eval_global_means = (
            eval_subset.groupby("mask_id", as_index=False)["continuation_loss"]
            .mean()
            .sort_values("continuation_loss")
            .reset_index(drop=True)
        )
        heldout_global_static = str(eval_global_means.iloc[0]["mask_id"])
        heldout_global_loss = eval_pivot[heldout_global_static]

        calib_global_static = calib_global_static_by_budget[skip_count]
        calib_global_loss = eval_pivot[calib_global_static]
        oracle_mask = eval_pivot.idxmin(axis=1)
        oracle_loss = eval_pivot.min(axis=1)
        oracle_headroom = calib_global_loss - oracle_loss

        bank_order = bank_order_by_budget[skip_count]
        global_mean_lookup = calib_global_mean_lookup_by_budget[skip_count]
        oracle_counter = calib_oracle_counter_by_budget[skip_count]
        calib_global_static_mask = parse_mask_id(calib_global_static, num_blocks)

        for bank_size in bank_sizes:
            bank_ids, bank_reasons = truncate_bank(bank_order, bank_size)
            bank_losses = eval_pivot[bank_ids]
            bank_best_mask = bank_losses.idxmin(axis=1)
            bank_best_loss = bank_losses.min(axis=1)
            recovered_headroom = calib_global_loss - bank_best_loss
            headroom_coverage = np.divide(
                recovered_headroom.to_numpy(),
                np.clip(oracle_headroom.to_numpy(), 1e-8, None),
            )

            for rank, mask_id in enumerate(bank_ids, start=1):
                candidate_rows.append(
                    {
                        "output_tag": args.output_tag,
                        "skip_count": skip_count,
                        "bank_size": bank_size,
                        "mask_id": mask_id,
                        "bank_rank": rank,
                        "reasons": ",".join(sorted(set(bank_reasons[mask_id]))),
                        "calib_global_mean_loss": float(global_mean_lookup.get(mask_id, float("nan"))),
                        "calib_oracle_frequency": int(oracle_counter.get(mask_id, 0)),
                        "swap_distance_to_calib_global_static": swap_distance(
                            parse_mask_id(mask_id, num_blocks), calib_global_static_mask
                        ),
                    }
                )

            for seq_idx in eval_pivot.index.tolist():
                seq_idx = int(seq_idx)
                per_sequence_rows.append(
                    {
                        "output_tag": args.output_tag,
                        "sequence_idx": seq_idx,
                        "skip_count": skip_count,
                        "bank_size": bank_size,
                        "bank_best_mask_id": str(bank_best_mask.loc[seq_idx]),
                        "oracle_mask_id": str(oracle_mask.loc[seq_idx]),
                        "calib_global_static_mask_id": calib_global_static,
                        "heldout_global_static_mask_id": heldout_global_static,
                        "bank_upper_bound_loss": float(bank_best_loss.loc[seq_idx]),
                        "oracle_loss": float(oracle_loss.loc[seq_idx]),
                        "calib_global_static_loss": float(calib_global_loss.loc[seq_idx]),
                        "heldout_global_static_loss": float(heldout_global_loss.loc[seq_idx]),
                        "delta_bank_to_calib_global": float(bank_best_loss.loc[seq_idx] - calib_global_loss.loc[seq_idx]),
                        "delta_bank_to_heldout_global": float(bank_best_loss.loc[seq_idx] - heldout_global_loss.loc[seq_idx]),
                        "delta_bank_to_oracle": float(bank_best_loss.loc[seq_idx] - oracle_loss.loc[seq_idx]),
                        "oracle_headroom_over_calib_global": float(oracle_headroom.loc[seq_idx]),
                        "oracle_headroom_over_heldout_global": float(heldout_global_loss.loc[seq_idx] - oracle_loss.loc[seq_idx]),
                        "recovered_headroom_over_calib_global": float(recovered_headroom.loc[seq_idx]),
                        "headroom_coverage_over_calib_global": float(headroom_coverage[list(eval_pivot.index).index(seq_idx)]),
                    }
                )

            for metric_name, values in [
                ("delta_bank_to_calib_global", bank_best_loss.to_numpy() - calib_global_loss.to_numpy()),
                ("delta_bank_to_heldout_global", bank_best_loss.to_numpy() - heldout_global_loss.to_numpy()),
                ("delta_bank_to_oracle", bank_best_loss.to_numpy() - oracle_loss.to_numpy()),
                ("oracle_headroom_over_calib_global", oracle_headroom.to_numpy()),
                ("oracle_headroom_over_heldout_global", heldout_global_loss.to_numpy() - oracle_loss.to_numpy()),
                ("headroom_coverage_over_calib_global", headroom_coverage),
                ("fraction_oracle_mask_in_bank", np.isin(oracle_mask.to_numpy(dtype=object), np.asarray(bank_ids, dtype=object)).astype(np.float64)),
                ("fraction_bank_matches_oracle", (bank_best_mask.to_numpy(dtype=object) == oracle_mask.to_numpy(dtype=object)).astype(np.float64)),
            ]:
                ci = bootstrap_summary(np.asarray(values, dtype=np.float64), seed=args.seed)
                summary_rows.append(
                    {
                        "output_tag": args.output_tag,
                        "skip_count": skip_count,
                        "bank_size": bank_size,
                        "metric": metric_name,
                        **ci,
                    }
                )

            seq_df = pd.DataFrame(
                {
                    "sequence_idx": eval_pivot.index.to_numpy(dtype=int),
                    "oracle_headroom": oracle_headroom.to_numpy(dtype=np.float64),
                    "bank_gap": (bank_best_loss.to_numpy() - calib_global_loss.to_numpy()).astype(np.float64),
                    "bank_gap_to_oracle": (bank_best_loss.to_numpy() - oracle_loss.to_numpy()).astype(np.float64),
                }
            )
            thresholds = [0.005, 0.01, 0.02, 0.05]
            for threshold in thresholds:
                headroom_mask = seq_df["oracle_headroom"] > threshold
                if headroom_mask.any():
                    ci = bootstrap_summary(headroom_mask.to_numpy(dtype=np.float64), seed=args.seed)
                    summary_rows.append(
                        {
                            "output_tag": args.output_tag,
                            "skip_count": skip_count,
                            "bank_size": bank_size,
                            "metric": f"fraction_oracle_headroom_gt_{threshold:.3f}",
                            **ci,
                        }
                    )

    candidate_df = pd.DataFrame(candidate_rows)
    per_sequence_df = pd.DataFrame(per_sequence_rows)
    summary_df = pd.DataFrame(summary_rows)

    candidate_df.to_csv(bank_hygiene_dir / f"{args.output_tag}_candidate_bank.csv", index=False)
    per_sequence_df.to_csv(bank_hygiene_dir / f"{args.output_tag}_per_sequence.csv", index=False)
    summary_df.to_csv(bank_hygiene_dir / f"{args.output_tag}_summary.csv", index=False)

    for skip_count in sorted(per_sequence_df["skip_count"].unique().tolist()):
        subset = per_sequence_df[per_sequence_df["skip_count"] == skip_count].copy()

        plt.figure(figsize=(6, 4))
        for bank_size in bank_sizes:
            bank_subset = subset[subset["bank_size"] == bank_size]
            values = np.sort(bank_subset["oracle_headroom_over_calib_global"].to_numpy(dtype=np.float64))
            if values.size == 0:
                continue
            y = np.arange(1, values.size + 1, dtype=np.float64) / values.size
            plt.plot(values, y, label=f"oracle headroom (bank={bank_size})")
            break
        plt.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
        plt.xlabel("oracle headroom over calibration global static")
        plt.ylabel("ECDF")
        plt.tight_layout()
        plt.savefig(plot_dir / f"bank_hygiene_{args.output_tag}_skip{skip_count}_oracle_headroom_ecdf.png", dpi=160)
        plt.close()

        plt.figure(figsize=(6, 4))
        for bank_size in bank_sizes:
            bank_subset = subset[subset["bank_size"] == bank_size]
            values = bank_subset["delta_bank_to_calib_global"].to_numpy(dtype=np.float64)
            plt.hist(values, bins=24, alpha=0.5, label=f"bank={bank_size}")
        plt.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
        plt.xlabel("bank upper bound delta to calibration global static")
        plt.ylabel("count")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(plot_dir / f"bank_hygiene_{args.output_tag}_skip{skip_count}_bank_delta_hist.png", dpi=160)
        plt.close()

        plt.figure(figsize=(6, 4))
        oracle_headroom_values = subset[subset["bank_size"] == bank_sizes[-1]]["oracle_headroom_over_calib_global"].to_numpy(dtype=np.float64)
        for bank_size in bank_sizes:
            bank_subset = subset[subset["bank_size"] == bank_size]
            plt.scatter(
                oracle_headroom_values,
                bank_subset["recovered_headroom_over_calib_global"].to_numpy(dtype=np.float64),
                alpha=0.35,
                s=12,
                label=f"bank={bank_size}",
            )
        plt.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
        plt.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
        plt.xlabel("oracle headroom over calibration global static")
        plt.ylabel("recovered headroom by train-only bank")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(plot_dir / f"bank_hygiene_{args.output_tag}_skip{skip_count}_recovered_vs_oracle.png", dpi=160)
        plt.close()


if __name__ == "__main__":
    main()
