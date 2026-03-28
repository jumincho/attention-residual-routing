#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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


def build_feature_table(feature_df: pd.DataFrame, oracle_df: pd.DataFrame) -> pd.DataFrame:
    merged = feature_df.merge(
        oracle_df[
            [
                "sequence_idx",
                "split",
                "stability_spearman",
                "stability_top3_jaccard",
                "prompt_margin",
            ]
        ].drop_duplicates(),
        on=["sequence_idx", "split"],
        how="left",
    )
    return merged.drop_duplicates(subset=["sequence_idx", "split"]).reset_index(drop=True)


def parse_prompt_scores(row: pd.Series) -> dict[str, np.ndarray]:
    return {
        "combined": np.asarray(json.loads(row["prompt_scores_json"]), dtype=np.float64)[1:-1],
        "attn": np.asarray(json.loads(row["prompt_scores_attn_json"]), dtype=np.float64)[1:-1],
        "mlp": np.asarray(json.loads(row["prompt_scores_mlp_json"]), dtype=np.float64)[1:-1],
    }


def make_sequence_features(row: pd.Series, feature_mode: str) -> np.ndarray:
    scores = parse_prompt_scores(row)
    scalars = np.asarray(
        [
            float(row["stability_spearman"]),
            float(row["stability_top3_jaccard"]),
            float(row["prompt_margin"]),
            float(row["prompt_depth_entropy"]),
        ],
        dtype=np.float64,
    )
    if feature_mode == "combined":
        return np.concatenate([scores["combined"], scalars], axis=0)
    if feature_mode == "sublayer":
        return np.concatenate([scores["combined"], scores["attn"], scores["mlp"], scalars], axis=0)
    raise ValueError(f"Unknown feature mode: {feature_mode}")


def make_pair_features(row: pd.Series, mask: np.ndarray, global_static_mask: np.ndarray) -> np.ndarray:
    scores = parse_prompt_scores(row)
    mid_mask = mask[:-1].astype(np.float64)
    global_mid = global_static_mask[:-1].astype(np.float64)
    seq_features = make_sequence_features(row, "sublayer")
    keep_combined = scores["combined"] * mid_mask
    keep_attn = scores["attn"] * mid_mask
    keep_mlp = scores["mlp"] * mid_mask
    delta_mid = mid_mask - global_mid
    swap_distance = float(np.abs(delta_mid).sum() / 2.0)
    aggregates = np.asarray(
        [
            float(keep_combined.sum()),
            float(keep_attn.sum()),
            float(keep_mlp.sum()),
            swap_distance,
        ],
        dtype=np.float64,
    )
    return np.concatenate(
        [
            seq_features,
            mid_mask,
            delta_mid,
            keep_combined,
            keep_attn,
            keep_mlp,
            aggregates,
        ],
        axis=0,
    )


def fit_models(seed: int) -> dict[str, Any]:
    return {
        "logreg_cls": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                multi_class="multinomial",
                solver="lbfgs",
                random_state=seed,
            ),
        ),
        "rf_cls": RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=-1,
        ),
        "rf_pair_reg": RandomForestRegressor(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=-1,
        ),
        "hgb_pair_reg": HistGradientBoostingRegressor(
            max_depth=6,
            max_iter=300,
            learning_rate=0.05,
            random_state=seed,
        ),
    }


def bootstrap_summary(values: np.ndarray, seed: int) -> dict[str, float]:
    return bootstrap_mean_ci(values, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-tag", type=str, required=True)
    parser.add_argument("--train-tags", type=str, nargs="+", required=True)
    parser.add_argument("--eval-tag", type=str, required=True)
    parser.add_argument("--output-tag", type=str, required=True)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    oracles_dir = ROOT / "results" / "oracles"
    mask_bank_dir = ROOT / "results" / "mask_bank"
    selector_dir = ROOT / "results" / "selector"
    selector_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = ROOT / "results" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    bank_df = pd.read_csv(mask_bank_dir / f"{args.bank_tag}_candidate_bank.csv")
    baseline_eval_df = pd.read_csv(mask_bank_dir / f"{args.eval_tag}_selectors_per_sequence.csv")

    train_feature_frames = []
    train_oracle_frames = []
    train_mask_frames = []
    for tag in args.train_tags:
        train_feature_frames.append(pd.read_csv(oracles_dir / f"{tag}_sequence_features.csv"))
        train_oracle_frames.append(pd.read_csv(oracles_dir / f"{tag}_oracle_mask_alignment.csv"))
        train_mask_frames.append(pd.read_csv(oracles_dir / f"{tag}_exhaustive_mask_losses.csv"))
    train_feature_df = pd.concat(train_feature_frames, ignore_index=True)
    train_oracle_df = pd.concat(train_oracle_frames, ignore_index=True)
    train_mask_df = pd.concat(train_mask_frames, ignore_index=True)

    eval_feature_df = pd.read_csv(oracles_dir / f"{args.eval_tag}_sequence_features.csv")
    eval_oracle_df = pd.read_csv(oracles_dir / f"{args.eval_tag}_oracle_mask_alignment.csv")
    eval_mask_df = pd.read_csv(oracles_dir / f"{args.eval_tag}_exhaustive_mask_losses.csv")

    num_blocks = len(json.loads(eval_feature_df.iloc[0]["prompt_scores_json"])) - 1
    train_features = build_feature_table(train_feature_df, train_oracle_df).set_index("sequence_idx")
    eval_features = build_feature_table(eval_feature_df, eval_oracle_df).set_index("sequence_idx")

    selection_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    train_seq_ids = np.array(sorted(train_features.index.unique().tolist()), dtype=int)
    rng.shuffle(train_seq_ids)
    split_idx = max(1, int(round(len(train_seq_ids) * args.train_frac)))
    dev_seq_ids = np.sort(train_seq_ids[:split_idx])
    holdout_seq_ids = np.sort(train_seq_ids[split_idx:])
    if holdout_seq_ids.size == 0:
        holdout_seq_ids = dev_seq_ids.copy()

    for skip_count in sorted(bank_df["skip_count"].unique().tolist()):
        bank_subset = bank_df[bank_df["skip_count"] == skip_count].copy()
        bank_ids = bank_subset.sort_values("bank_rank")["mask_id"].tolist()
        global_static = bank_subset[bank_subset["reasons"].str.contains("global_static")].iloc[0]["mask_id"]
        global_static_mask = parse_mask_id(global_static, num_blocks)
        label_to_idx = {mask_id: idx for idx, mask_id in enumerate(bank_ids)}
        idx_to_label = {idx: mask_id for mask_id, idx in label_to_idx.items()}

        train_subset = train_mask_df[(train_mask_df["skip_count"] == skip_count) & (train_mask_df["method"] == "exhaustive_mask")]
        train_pivot = train_subset.pivot(index="sequence_idx", columns="mask_id", values="continuation_loss")
        train_pivot = train_pivot[bank_ids]
        train_best = train_pivot.idxmin(axis=1)
        train_global = train_pivot[global_static]

        eval_subset = eval_mask_df[(eval_mask_df["skip_count"] == skip_count) & (eval_mask_df["method"] == "exhaustive_mask")]
        eval_pivot = eval_subset.pivot(index="sequence_idx", columns="mask_id", values="continuation_loss")
        eval_pivot = eval_pivot[bank_ids]
        eval_best = eval_pivot.idxmin(axis=1)
        eval_global = eval_pivot[global_static]

        cls_x = np.stack([make_sequence_features(train_features.loc[seq_id], "sublayer") for seq_id in train_pivot.index], axis=0)
        cls_y = np.asarray([label_to_idx[str(train_best.loc[seq_id])] for seq_id in train_pivot.index], dtype=np.int64)
        cls_seq_ids = np.asarray(train_pivot.index.tolist(), dtype=int)

        pair_rows = []
        pair_targets = []
        pair_seq_ids = []
        pair_mask_ids = []
        for seq_id in train_pivot.index.tolist():
            row = train_features.loc[int(seq_id)]
            global_loss = float(train_global.loc[int(seq_id)])
            for mask_id in bank_ids:
                mask = parse_mask_id(mask_id, num_blocks)
                pair_rows.append(make_pair_features(row, mask, global_static_mask))
                pair_targets.append(float(train_pivot.loc[int(seq_id), mask_id] - global_loss))
                pair_seq_ids.append(int(seq_id))
                pair_mask_ids.append(mask_id)
        pair_x = np.stack(pair_rows, axis=0)
        pair_y = np.asarray(pair_targets, dtype=np.float64)
        pair_seq_ids = np.asarray(pair_seq_ids, dtype=int)
        pair_mask_ids = np.asarray(pair_mask_ids, dtype=object)

        models = fit_models(args.seed)
        holdout_scores: dict[str, float] = {}
        fitted_models: dict[str, Any] = {}

        cls_train_mask = np.isin(cls_seq_ids, dev_seq_ids)
        cls_holdout_mask = np.isin(cls_seq_ids, holdout_seq_ids)
        pair_train_mask = np.isin(pair_seq_ids, dev_seq_ids)
        pair_holdout_mask = np.isin(pair_seq_ids, holdout_seq_ids)

        for model_name, model in models.items():
            if model_name.endswith("_cls"):
                model.fit(cls_x[cls_train_mask], cls_y[cls_train_mask])
                preds = model.predict(cls_x[cls_holdout_mask])
                holdout_losses = []
                for seq_id, pred in zip(cls_seq_ids[cls_holdout_mask].tolist(), preds.tolist()):
                    chosen = idx_to_label[int(pred)]
                    holdout_losses.append(float(train_pivot.loc[int(seq_id), chosen] - train_global.loc[int(seq_id)]))
                holdout_scores[model_name] = float(np.mean(holdout_losses))
            else:
                model.fit(pair_x[pair_train_mask], pair_y[pair_train_mask])
                holdout_losses = []
                for seq_id in holdout_seq_ids.tolist():
                    pair_idx = np.where(pair_seq_ids == int(seq_id))[0]
                    pred = model.predict(pair_x[pair_idx])
                    chosen = str(pair_mask_ids[pair_idx][int(np.argmin(pred))])
                    holdout_losses.append(float(train_pivot.loc[int(seq_id), chosen] - train_global.loc[int(seq_id)]))
                holdout_scores[model_name] = float(np.mean(holdout_losses))
            fitted_models[model_name] = model

        for model_name, score in holdout_scores.items():
            selection_rows.append(
                {
                    "output_tag": args.output_tag,
                    "bank_tag": args.bank_tag,
                    "eval_tag": args.eval_tag,
                    "skip_count": skip_count,
                    "model_name": model_name,
                    "holdout_delta_to_global_static": score,
                    "num_train_sequences": int(dev_seq_ids.size),
                    "num_holdout_sequences": int(holdout_seq_ids.size),
                }
            )

        best_model_name = min(holdout_scores.items(), key=lambda item: item[1])[0]
        selection_rows.append(
            {
                "output_tag": args.output_tag,
                "bank_tag": args.bank_tag,
                "eval_tag": args.eval_tag,
                "skip_count": skip_count,
                "model_name": "best_model",
                "holdout_delta_to_global_static": float(holdout_scores[best_model_name]),
                "selected_model_name": best_model_name,
                "num_train_sequences": int(dev_seq_ids.size),
                "num_holdout_sequences": int(holdout_seq_ids.size),
            }
        )

        refit_models = fit_models(args.seed)
        for model_name, model in refit_models.items():
            if model_name.endswith("_cls"):
                model.fit(cls_x, cls_y)
                preds = model.predict(np.stack([make_sequence_features(eval_features.loc[seq_id], "sublayer") for seq_id in eval_pivot.index], axis=0))
                for seq_id, pred in zip(eval_pivot.index.tolist(), preds.tolist()):
                    chosen = idx_to_label[int(pred)]
                    loss = float(eval_pivot.loc[int(seq_id), chosen])
                    eval_rows.append(
                        {
                            "output_tag": args.output_tag,
                            "skip_count": skip_count,
                            "sequence_idx": int(seq_id),
                            "method": model_name,
                            "selected_mask_id": chosen,
                            "continuation_loss": loss,
                            "delta_to_global_static": float(loss - eval_global.loc[int(seq_id)]),
                            "delta_to_bank_upper_bound": float(loss - eval_pivot.loc[int(seq_id), str(eval_best.loc[int(seq_id)])]),
                        }
                    )
            else:
                model.fit(pair_x, pair_y)
                for seq_id in eval_pivot.index.tolist():
                    row = eval_features.loc[int(seq_id)]
                    candidate_features = np.stack(
                        [
                            make_pair_features(row, parse_mask_id(mask_id, num_blocks), global_static_mask)
                            for mask_id in bank_ids
                        ],
                        axis=0,
                    )
                    pred = model.predict(candidate_features)
                    chosen = bank_ids[int(np.argmin(pred))]
                    loss = float(eval_pivot.loc[int(seq_id), chosen])
                    eval_rows.append(
                        {
                            "output_tag": args.output_tag,
                            "skip_count": skip_count,
                            "sequence_idx": int(seq_id),
                            "method": model_name,
                            "selected_mask_id": chosen,
                            "continuation_loss": loss,
                            "delta_to_global_static": float(loss - eval_global.loc[int(seq_id)]),
                            "delta_to_bank_upper_bound": float(loss - eval_pivot.loc[int(seq_id), str(eval_best.loc[int(seq_id)])]),
                        }
                    )

        best_model = refit_models[best_model_name]
        for seq_id in eval_pivot.index.tolist():
            if best_model_name.endswith("_cls"):
                chosen = idx_to_label[
                    int(best_model.predict(make_sequence_features(eval_features.loc[int(seq_id)], "sublayer")[None, :])[0])
                ]
            else:
                row = eval_features.loc[int(seq_id)]
                candidate_features = np.stack(
                    [
                        make_pair_features(row, parse_mask_id(mask_id, num_blocks), global_static_mask)
                        for mask_id in bank_ids
                    ],
                    axis=0,
                )
                chosen = bank_ids[int(np.argmin(best_model.predict(candidate_features)))]
            loss = float(eval_pivot.loc[int(seq_id), chosen])
            eval_rows.append(
                {
                    "output_tag": args.output_tag,
                    "skip_count": skip_count,
                    "sequence_idx": int(seq_id),
                    "method": "best_model",
                    "selected_mask_id": chosen,
                    "continuation_loss": loss,
                    "delta_to_global_static": float(loss - eval_global.loc[int(seq_id)]),
                    "delta_to_bank_upper_bound": float(loss - eval_pivot.loc[int(seq_id), str(eval_best.loc[int(seq_id)])]),
                }
            )

    selection_df = pd.DataFrame(selection_rows)
    eval_df = pd.DataFrame(eval_rows)

    baseline_keep = baseline_eval_df[baseline_eval_df["method"].isin(
        [
            "global_static",
            "prompt_fixed",
            "bank_upper_bound",
            "knn_combined",
            "knn_sublayer",
            "prototype_sublayer",
            "prototype_combined",
            "local_edit_attn",
            "bank_score_attn",
            "oracle_sequence",
        ]
    )][["sequence_idx", "skip_count", "method", "continuation_loss", "delta_to_global_static", "delta_to_bank_upper_bound"]].copy()
    baseline_keep["output_tag"] = args.output_tag
    eval_all = pd.concat([baseline_keep, eval_df], ignore_index=True)

    selection_df.to_csv(selector_dir / f"{args.output_tag}_model_selection.csv", index=False)
    eval_all.to_csv(selector_dir / f"{args.output_tag}_eval_per_sequence.csv", index=False)

    for (skip_count, method), subset in eval_all.groupby(["skip_count", "method"], sort=True):
        for metric in ["continuation_loss", "delta_to_global_static", "delta_to_bank_upper_bound"]:
            ci = bootstrap_summary(subset[metric].to_numpy(), seed=args.seed)
            summary_rows.append(
                {
                    "output_tag": args.output_tag,
                    "skip_count": skip_count,
                    "method": method,
                    "metric": metric,
                    **ci,
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(selector_dir / f"{args.output_tag}_eval_summary.csv", index=False)

    for metric in ["delta_to_global_static", "delta_to_bank_upper_bound"]:
        plt.figure(figsize=(7, 4))
        subset = summary_df[summary_df["metric"] == metric]
        for method in [
            "global_static",
            "prompt_fixed",
            "knn_sublayer",
            "prototype_sublayer",
            "rf_cls",
            "logreg_cls",
            "rf_pair_reg",
            "hgb_pair_reg",
            "best_model",
            "bank_upper_bound",
        ]:
            method_df = subset[subset["method"] == method].sort_values("skip_count")
            if method_df.empty:
                continue
            plt.plot(method_df["skip_count"], method_df["mean"], marker="o", label=method)
        plt.xlabel("skip count")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.legend(fontsize=8)
        plt.savefig(plot_dir / f"selector_{args.output_tag}_{metric}.png", dpi=160)
        plt.close()


if __name__ == "__main__":
    main()
