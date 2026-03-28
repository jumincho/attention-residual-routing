#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
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


def mask_to_id(mask: np.ndarray) -> str:
    kept = [str(idx + 1) for idx, value in enumerate(mask.tolist()) if value]
    return "keep:" + ",".join(kept)


def swap_distance(mask_a: np.ndarray, mask_b: np.ndarray) -> int:
    return int(np.logical_xor(mask_a, mask_b).sum() // 2)


def standardized(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (train - mean) / std, (test - mean) / std


def euclidean_score_mask(mask: np.ndarray, scores_mid: np.ndarray) -> float:
    return float(scores_mid[mask[:-1]].sum())


def weighted_vote(labels: list[str], distances: np.ndarray) -> str:
    votes: dict[str, float] = defaultdict(float)
    for label, dist in zip(labels, distances.tolist()):
        votes[label] += 1.0 / max(dist, 1e-6)
    return max(votes.items(), key=lambda item: item[1])[0]


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


def build_feature_table(feature_df: pd.DataFrame, oracle_df: pd.DataFrame) -> pd.DataFrame:
    merged = feature_df.merge(
        oracle_df[
            [
                "sequence_idx",
                "split",
                "skip_count",
                "stability_spearman",
                "stability_top3_jaccard",
                "prompt_margin",
            ]
        ].drop_duplicates(),
        on=["sequence_idx", "split"],
        how="left",
    )
    return merged.drop_duplicates(subset=["sequence_idx", "split"]).reset_index(drop=True)


def make_feature_vector(row: pd.Series, mode: str) -> np.ndarray:
    combined = np.asarray(json.loads(row["prompt_scores_json"]), dtype=np.float64)[1:-1]
    attn = np.asarray(json.loads(row["prompt_scores_attn_json"]), dtype=np.float64)[1:-1]
    mlp = np.asarray(json.loads(row["prompt_scores_mlp_json"]), dtype=np.float64)[1:-1]
    scalars = np.asarray(
        [
            float(row["stability_spearman"]),
            float(row["stability_top3_jaccard"]),
            float(row["prompt_margin"]),
            float(row["prompt_depth_entropy"]),
        ],
        dtype=np.float64,
    )
    if mode == "combined":
        return np.concatenate([combined, scalars], axis=0)
    if mode == "sublayer":
        return np.concatenate([combined, attn, mlp, scalars], axis=0)
    raise ValueError(f"Unknown feature mode: {mode}")


def summarize_metric(values: np.ndarray, seed: int) -> dict[str, float]:
    return bootstrap_mean_ci(values, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--top-global", type=int, default=8)
    parser.add_argument("--top-oracle", type=int, default=8)
    parser.add_argument("--diverse-oracle", type=int, default=4)
    parser.add_argument("--max-bank-size", type=int, default=16)
    parser.add_argument("--local-swap-radius", type=int, default=1)
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    oracles_dir = ROOT / "results" / "oracles"
    routing_dir = ROOT / "results" / "routing"
    out_dir = ROOT / "results" / "mask_bank"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = ROOT / "results" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    mask_df = pd.read_csv(oracles_dir / f"{args.tag}_exhaustive_mask_losses.csv")
    feature_df = pd.read_csv(oracles_dir / f"{args.tag}_sequence_features.csv")
    oracle_df = pd.read_csv(oracles_dir / f"{args.tag}_oracle_mask_alignment.csv")
    routing_df = pd.read_csv(routing_dir / f"{args.tag}_routing_eval_per_sequence.csv")

    num_blocks = len(json.loads(feature_df.iloc[0]["prompt_scores_json"])) - 1
    feature_table = build_feature_table(feature_df, oracle_df).set_index("sequence_idx")

    candidate_rows: list[dict[str, Any]] = []
    selector_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    exhaustive_df = mask_df[mask_df["method"] == "exhaustive_mask"].copy()

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
        for _, grp in subset.groupby("sequence_idx"):
            oracle_mask = str(grp.sort_values("continuation_loss").iloc[0]["mask_id"])
            oracle_counter[oracle_mask] += 1

        bank_order: list[tuple[str, str]] = []
        bank_order.append((global_static, "global_static"))
        for mask_id in global_means["mask_id"].head(args.top_global).tolist():
            bank_order.append((str(mask_id), "top_global"))
        for mask_id, _count in oracle_counter.most_common(args.top_oracle):
            bank_order.append((str(mask_id), "oracle_freq"))
        for max_swaps in [1, 2]:
            for mask_id in global_means["mask_id"].tolist():
                dist = swap_distance(parse_mask_id(str(mask_id), num_blocks), global_static_mask)
                if dist == max_swaps:
                    bank_order.append((str(mask_id), f"swap_{max_swaps}"))
        for mask_id in farthest_first_masks(oracle_counter, num_blocks, args.diverse_oracle):
            bank_order.append((str(mask_id), "oracle_diverse"))

        bank_ids: list[str] = []
        bank_reasons: dict[str, list[str]] = defaultdict(list)
        for mask_id, reason in bank_order:
            if mask_id not in bank_ids and len(bank_ids) < args.max_bank_size:
                bank_ids.append(mask_id)
            if mask_id in bank_ids:
                bank_reasons[mask_id].append(reason)

        global_mean_lookup = dict(zip(global_means["mask_id"], global_means["continuation_loss"]))
        for rank, mask_id in enumerate(bank_ids, start=1):
            candidate_rows.append(
                {
                    "tag": args.tag,
                    "skip_count": skip_count,
                    "mask_id": mask_id,
                    "bank_rank": rank,
                    "reasons": ",".join(sorted(set(bank_reasons[mask_id]))),
                    "global_mean_loss": float(global_mean_lookup[mask_id]),
                    "oracle_frequency": int(oracle_counter.get(mask_id, 0)),
                    "swap_distance_to_global_static": swap_distance(parse_mask_id(mask_id, num_blocks), global_static_mask),
                }
            )

        loss_pivot = subset.pivot(index="sequence_idx", columns="mask_id", values="continuation_loss")
        bank_losses = loss_pivot[bank_ids].copy()
        bank_best_mask = bank_losses.idxmin(axis=1)
        bank_best_loss = bank_losses.min(axis=1)

        routing_subset = routing_df[routing_df["skip_count"] == skip_count].copy()
        routing_pivot = routing_subset.pivot(index="sequence_idx", columns="method", values="continuation_loss")

        feature_vectors_combined = {
            seq_idx: make_feature_vector(feature_table.loc[seq_idx], "combined")
            for seq_idx in bank_losses.index.tolist()
        }
        feature_vectors_sublayer = {
            seq_idx: make_feature_vector(feature_table.loc[seq_idx], "sublayer")
            for seq_idx in bank_losses.index.tolist()
        }

        local_candidates = [
            mask_id
            for mask_id in bank_ids
            if swap_distance(parse_mask_id(mask_id, num_blocks), global_static_mask) <= args.local_swap_radius
        ]
        if global_static not in local_candidates:
            local_candidates = [global_static] + local_candidates

        for seq_idx in bank_losses.index.tolist():
            seq_idx = int(seq_idx)
            oracle_mask = str(loss_pivot.loc[seq_idx].idxmin())
            bank_oracle_mask = str(bank_best_mask.loc[seq_idx])
            prompt_scores_combined = np.asarray(json.loads(feature_table.loc[seq_idx]["prompt_scores_json"]), dtype=np.float64)
            prompt_scores_attn = np.asarray(json.loads(feature_table.loc[seq_idx]["prompt_scores_attn_json"]), dtype=np.float64)
            scores_mid_combined = prompt_scores_combined[1:-1]
            scores_mid_attn = prompt_scores_attn[1:-1]

            baseline_rows = {
                "bank_upper_bound": bank_oracle_mask,
                "global_static": global_static,
                "prompt_fixed": None,
                "mismatched_prompt": None,
                "balanced": None,
                "oracle_sequence": oracle_mask,
                "no_skip": None,
            }
            for method in ["prompt_fixed", "mismatched_prompt", "balanced", "oracle_sequence", "global_static", "no_skip"]:
                if method in routing_pivot.columns:
                    selector_rows.append(
                        {
                            "tag": args.tag,
                            "sequence_idx": seq_idx,
                            "skip_count": skip_count,
                            "method": method,
                            "selected_mask_id": baseline_rows.get(method, ""),
                            "continuation_loss": float(routing_pivot.loc[seq_idx, method]),
                            "delta_to_global_static": float(routing_pivot.loc[seq_idx, method] - routing_pivot.loc[seq_idx, "global_static"]),
                            "delta_to_bank_upper_bound": float(routing_pivot.loc[seq_idx, method] - bank_best_loss.loc[seq_idx]),
                            "delta_to_oracle_sequence": float(routing_pivot.loc[seq_idx, method] - routing_pivot.loc[seq_idx, "oracle_sequence"]),
                            "selected_from_bank": False,
                        }
                    )
            selector_rows.append(
                {
                    "tag": args.tag,
                    "sequence_idx": seq_idx,
                    "skip_count": skip_count,
                    "method": "bank_upper_bound",
                    "selected_mask_id": bank_oracle_mask,
                    "continuation_loss": float(bank_best_loss.loc[seq_idx]),
                    "delta_to_global_static": float(bank_best_loss.loc[seq_idx] - routing_pivot.loc[seq_idx, "global_static"]),
                    "delta_to_bank_upper_bound": 0.0,
                    "delta_to_oracle_sequence": float(bank_best_loss.loc[seq_idx] - routing_pivot.loc[seq_idx, "oracle_sequence"]),
                    "selected_from_bank": True,
                }
            )

            def add_bank_method(method: str, mask_id: str) -> None:
                loss = float(loss_pivot.loc[seq_idx, mask_id])
                selector_rows.append(
                    {
                        "tag": args.tag,
                        "sequence_idx": seq_idx,
                        "skip_count": skip_count,
                        "method": method,
                        "selected_mask_id": mask_id,
                        "continuation_loss": loss,
                        "delta_to_global_static": float(loss - routing_pivot.loc[seq_idx, "global_static"]),
                        "delta_to_bank_upper_bound": float(loss - bank_best_loss.loc[seq_idx]),
                        "delta_to_oracle_sequence": float(loss - routing_pivot.loc[seq_idx, "oracle_sequence"]),
                        "selected_from_bank": True,
                    }
                )

            score_rank_combined = max(
                bank_ids,
                key=lambda mask_id: euclidean_score_mask(parse_mask_id(mask_id, num_blocks), scores_mid_combined),
            )
            score_rank_attn = max(
                bank_ids,
                key=lambda mask_id: euclidean_score_mask(parse_mask_id(mask_id, num_blocks), scores_mid_attn),
            )
            add_bank_method("bank_score_combined", score_rank_combined)
            add_bank_method("bank_score_attn", score_rank_attn)

            local_combined = max(
                local_candidates,
                key=lambda mask_id: euclidean_score_mask(parse_mask_id(mask_id, num_blocks), scores_mid_combined),
            )
            local_attn = max(
                local_candidates,
                key=lambda mask_id: euclidean_score_mask(parse_mask_id(mask_id, num_blocks), scores_mid_attn),
            )
            add_bank_method("local_edit_combined", local_combined)
            add_bank_method("local_edit_attn", local_attn)

            train_ids = [other for other in bank_losses.index.tolist() if int(other) != seq_idx]

            for feature_mode, feature_map, proto_name, knn_name in [
                ("combined", feature_vectors_combined, "prototype_combined", "knn_combined"),
                ("sublayer", feature_vectors_sublayer, "prototype_sublayer", "knn_sublayer"),
            ]:
                train_x = np.stack([feature_map[int(other)] for other in train_ids], axis=0)
                test_x = feature_map[seq_idx][None, :]
                train_x_std, test_x_std = standardized(train_x, test_x)
                train_labels = [str(bank_best_mask.loc[int(other)]) for other in train_ids]

                prototypes: dict[str, np.ndarray] = {}
                for mask_id in bank_ids:
                    members = [idx for idx, label in enumerate(train_labels) if label == mask_id]
                    if members:
                        prototypes[mask_id] = train_x_std[members].mean(axis=0)
                proto_choice = min(
                    prototypes.items(),
                    key=lambda item: float(np.square(test_x_std[0] - item[1]).sum()),
                )[0]
                add_bank_method(proto_name, proto_choice)

                distances = np.linalg.norm(train_x_std - test_x_std[0], axis=1)
                order = np.argsort(distances)[: min(args.knn_k, len(distances))]
                knn_choice = weighted_vote([train_labels[idx] for idx in order], distances[order])
                add_bank_method(knn_name, knn_choice)

    candidate_df = pd.DataFrame(candidate_rows)
    selector_df = pd.DataFrame(selector_rows)
    candidate_df.to_csv(out_dir / f"{args.tag}_candidate_bank.csv", index=False)
    selector_df.to_csv(out_dir / f"{args.tag}_selectors_per_sequence.csv", index=False)

    for (skip_count, method), subset in selector_df.groupby(["skip_count", "method"], sort=True):
        for metric in ["continuation_loss", "delta_to_global_static", "delta_to_bank_upper_bound", "delta_to_oracle_sequence"]:
            ci = summarize_metric(subset[metric].to_numpy(), seed=args.seed)
            summary_rows.append(
                {
                    "tag": args.tag,
                    "skip_count": skip_count,
                    "method": method,
                    "metric": metric,
                    **ci,
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / f"{args.tag}_selectors_summary.csv", index=False)

    for metric in ["continuation_loss", "delta_to_global_static", "delta_to_bank_upper_bound"]:
        plt.figure(figsize=(7, 4))
        subset = summary_df[summary_df["metric"] == metric].copy()
        for method in [
            "global_static",
            "prompt_fixed",
            "bank_upper_bound",
            "bank_score_combined",
            "bank_score_attn",
            "local_edit_attn",
            "prototype_sublayer",
            "knn_sublayer",
            "oracle_sequence",
        ]:
            method_df = subset[subset["method"] == method].sort_values("skip_count")
            if method_df.empty:
                continue
            plt.plot(method_df["skip_count"], method_df["mean"], marker="o", label=method)
        plt.xlabel("skip count")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.legend(fontsize=8)
        plt.savefig(plot_dir / f"mask_bank_{args.tag}_{metric}.png", dpi=160)
        plt.close()


if __name__ == "__main__":
    main()
