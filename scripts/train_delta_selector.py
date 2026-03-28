#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
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


def bootstrap_summary(values: np.ndarray, seed: int) -> dict[str, float]:
    return bootstrap_mean_ci(values.astype(np.float64), seed=seed)


def concat_frames(base_dir: Path, tags: list[str], suffix: str) -> pd.DataFrame:
    frames = [pd.read_csv(base_dir / f"{tag}_{suffix}.csv") for tag in tags]
    return pd.concat(frames, ignore_index=True)


def build_feature_table(
    feature_df: pd.DataFrame,
    oracle_df: pd.DataFrame,
    hidden_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    keep_cols = [
        "sequence_idx",
        "split",
        "skip_count",
        "stability_spearman",
        "stability_top3_jaccard",
        "prompt_margin",
    ]
    merged = feature_df.merge(
        oracle_df[keep_cols].drop_duplicates(),
        on=["sequence_idx", "split"],
        how="left",
    )
    if hidden_df is not None:
        merged = merged.merge(
            hidden_df.drop_duplicates(subset=["sequence_idx", "split"]),
            on=["sequence_idx", "split"],
            how="left",
        )
    return merged.drop_duplicates(subset=["sequence_idx", "split", "skip_count"]).reset_index(drop=True)


def parse_prompt_scores(row: pd.Series) -> dict[str, np.ndarray]:
    return {
        "combined": np.asarray(json.loads(row["prompt_scores_json"]), dtype=np.float64)[1:-1],
        "attn": np.asarray(json.loads(row["prompt_scores_attn_json"]), dtype=np.float64)[1:-1],
        "mlp": np.asarray(json.loads(row["prompt_scores_mlp_json"]), dtype=np.float64)[1:-1],
        "chunk": np.asarray(json.loads(row["prompt_chunk_utilities_json"]), dtype=np.float64)[:, 1:-1],
    }


def parse_hidden_features(row: pd.Series) -> np.ndarray:
    parts = []
    for column in sorted(row.index.tolist()):
        if column.endswith("_mean_json") or column.endswith("_final_json"):
            value = row[column]
            if isinstance(value, str) and value:
                parts.append(np.asarray(json.loads(value), dtype=np.float64))
        elif column.endswith("_norm_mean") or column.endswith("_norm_std"):
            value = row[column]
            if not pd.isna(value):
                parts.append(np.asarray([float(value)], dtype=np.float64))
    if not parts:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(parts, axis=0)


def make_sequence_features(row: pd.Series, feature_mode: str) -> np.ndarray:
    scores = parse_prompt_scores(row)
    chunk_mean = scores["chunk"].mean(axis=0)
    chunk_std = scores["chunk"].std(axis=0)
    scalars = np.asarray(
        [
            float(row["stability_spearman"]),
            float(row["stability_top3_jaccard"]),
            float(row["prompt_margin"]),
            float(row["prompt_depth_entropy"]),
            float(row["prompt_support_size"]),
        ],
        dtype=np.float64,
    )
    attnres = np.concatenate(
        [
            scores["combined"],
            scores["attn"],
            scores["mlp"],
            chunk_mean,
            chunk_std,
            scalars,
        ],
        axis=0,
    )
    hidden = parse_hidden_features(row)
    if feature_mode == "attnres":
        return attnres
    if feature_mode == "hidden":
        return hidden
    if feature_mode == "combined":
        return np.concatenate([attnres, hidden], axis=0)
    raise ValueError(f"Unsupported feature_mode: {feature_mode}")


def make_pair_features(
    row: pd.Series,
    mask: np.ndarray,
    global_static_mask: np.ndarray,
    feature_mode: str,
) -> np.ndarray:
    scores = parse_prompt_scores(row)
    mid_mask = mask[:-1].astype(np.float64)
    global_mid = global_static_mask[:-1].astype(np.float64)
    delta_mid = mid_mask - global_mid
    edit_count = float(np.abs(delta_mid).sum() / 2.0)
    is_static = float(np.allclose(delta_mid, 0.0))
    cost_aggregates = np.asarray(
        [
            edit_count,
            is_static,
            float(mid_mask.sum()),
        ],
        dtype=np.float64,
    )
    if feature_mode == "hidden":
        prompt_interactions = np.zeros(0, dtype=np.float64)
        prompt_aggregates = np.zeros(0, dtype=np.float64)
    else:
        keep_combined = scores["combined"] * mid_mask
        keep_attn = scores["attn"] * mid_mask
        keep_mlp = scores["mlp"] * mid_mask
        drop_combined = scores["combined"] * (1.0 - mid_mask)
        drop_attn = scores["attn"] * (1.0 - mid_mask)
        drop_mlp = scores["mlp"] * (1.0 - mid_mask)
        prompt_interactions = np.concatenate(
            [
                keep_combined,
                keep_attn,
                keep_mlp,
                drop_combined,
                drop_attn,
                drop_mlp,
            ],
            axis=0,
        )
        prompt_aggregates = np.asarray(
            [
                float(keep_combined.sum()),
                float(keep_attn.sum()),
                float(keep_mlp.sum()),
                float(drop_combined.sum()),
                float(drop_attn.sum()),
                float(drop_mlp.sum()),
            ],
            dtype=np.float64,
        )
    return np.concatenate(
        [
            make_sequence_features(row, feature_mode),
            mid_mask,
            delta_mid,
            prompt_interactions,
            cost_aggregates,
            prompt_aggregates,
        ],
        axis=0,
    )


def fit_models(seed: int) -> dict[str, Any]:
    return {
        "rf_delta_reg": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=-1,
        ),
        "hgb_delta_reg": HistGradientBoostingRegressor(
            max_depth=6,
            max_iter=250,
            learning_rate=0.05,
            random_state=seed,
        ),
        "rf_improve_cls": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=-1,
        ),
        "logreg_improve_cls": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                random_state=seed,
            ),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-tag", type=str, required=True)
    parser.add_argument("--bank-size", type=int, required=True)
    parser.add_argument("--calib-tags", type=str, nargs="+", required=True)
    parser.add_argument("--eval-tags", type=str, nargs="+", required=True)
    parser.add_argument("--output-tag", type=str, required=True)
    parser.add_argument("--feature-mode", type=str, choices=["attnres", "hidden", "combined"], default="attnres")
    parser.add_argument("--hidden-calib-tags", type=str, nargs="*", default=None)
    parser.add_argument("--hidden-eval-tags", type=str, nargs="*", default=None)
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        help="Subset of model names from fit_models(); defaults to all.",
    )
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    delta_dir = ROOT / "results" / "delta_selector"
    delta_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = ROOT / "results" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    bank_dir = ROOT / "results" / "bank_hygiene"
    oracles_dir = ROOT / "results" / "oracles"
    rich_dir = ROOT / "results" / "rich_features"

    bank_df = pd.read_csv(bank_dir / f"{args.bank_tag}_candidate_bank.csv")
    bank_eval_df = pd.read_csv(bank_dir / f"{args.bank_tag}_per_sequence.csv")
    bank_df = bank_df[bank_df["bank_size"] == args.bank_size].copy()
    bank_eval_df = bank_eval_df[bank_eval_df["bank_size"] == args.bank_size].copy()

    calib_features_raw = concat_frames(oracles_dir, args.calib_tags, "sequence_features")
    calib_oracle_raw = concat_frames(oracles_dir, args.calib_tags, "oracle_mask_alignment")
    calib_masks_raw = concat_frames(oracles_dir, args.calib_tags, "exhaustive_mask_losses")
    eval_features_raw = concat_frames(oracles_dir, args.eval_tags, "sequence_features")
    eval_oracle_raw = concat_frames(oracles_dir, args.eval_tags, "oracle_mask_alignment")
    eval_masks_raw = concat_frames(oracles_dir, args.eval_tags, "exhaustive_mask_losses")
    calib_hidden_raw = concat_frames(rich_dir, args.hidden_calib_tags, "hidden_prompt_features") if args.hidden_calib_tags else None
    eval_hidden_raw = concat_frames(rich_dir, args.hidden_eval_tags, "hidden_prompt_features") if args.hidden_eval_tags else None

    calib_features = build_feature_table(calib_features_raw, calib_oracle_raw, calib_hidden_raw)
    eval_features = build_feature_table(eval_features_raw, eval_oracle_raw, eval_hidden_raw)
    num_blocks = len(json.loads(calib_features.iloc[0]["prompt_scores_json"])) - 1

    rng = np.random.default_rng(args.seed)
    selection_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for skip_count in sorted(bank_df["skip_count"].unique().tolist()):
        bank_subset = bank_df[bank_df["skip_count"] == skip_count].sort_values("bank_rank")
        bank_ids = bank_subset["mask_id"].tolist()
        global_static = str(
            bank_subset[bank_subset["reasons"].str.contains("calib_global_static")].iloc[0]["mask_id"]
        )
        global_static_mask = parse_mask_id(global_static, num_blocks)

        calib_feature_subset = calib_features[calib_features["skip_count"] == skip_count].set_index("sequence_idx")
        eval_feature_subset = eval_features[eval_features["skip_count"] == skip_count].set_index("sequence_idx")

        calib_mask_subset = calib_masks_raw[
            (calib_masks_raw["skip_count"] == skip_count) & (calib_masks_raw["method"] == "exhaustive_mask")
        ]
        eval_mask_subset = eval_masks_raw[
            (eval_masks_raw["skip_count"] == skip_count) & (eval_masks_raw["method"] == "exhaustive_mask")
        ]
        calib_pivot = calib_mask_subset.pivot(index="sequence_idx", columns="mask_id", values="continuation_loss")[bank_ids]
        eval_pivot = eval_mask_subset.pivot(index="sequence_idx", columns="mask_id", values="continuation_loss")[bank_ids]

        calib_global = calib_pivot[global_static]
        eval_global = eval_pivot[global_static]
        eval_oracle = eval_pivot.min(axis=1)
        eval_bank_upper = bank_eval_df[bank_eval_df["skip_count"] == skip_count].set_index("sequence_idx")

        seq_ids = np.asarray(sorted(calib_pivot.index.tolist()), dtype=int)
        rng.shuffle(seq_ids)
        split_idx = max(1, int(round(len(seq_ids) * args.train_frac)))
        train_ids = np.sort(seq_ids[:split_idx])
        holdout_ids = np.sort(seq_ids[split_idx:])
        if holdout_ids.size == 0:
            holdout_ids = train_ids.copy()

        pair_rows = []
        delta_targets = []
        improve_targets = []
        pair_seq_ids = []
        pair_mask_ids = []
        for seq_id in calib_pivot.index.tolist():
            row = calib_feature_subset.loc[int(seq_id)]
            global_loss = float(calib_global.loc[int(seq_id)])
            for mask_id in bank_ids:
                mask = parse_mask_id(mask_id, num_blocks)
                pair_rows.append(make_pair_features(row, mask, global_static_mask, args.feature_mode))
                delta = float(calib_pivot.loc[int(seq_id), mask_id] - global_loss)
                delta_targets.append(delta)
                improve_targets.append(int(delta < 0.0))
                pair_seq_ids.append(int(seq_id))
                pair_mask_ids.append(mask_id)
        pair_x = np.stack(pair_rows, axis=0)
        delta_y = np.asarray(delta_targets, dtype=np.float64)
        improve_y = np.asarray(improve_targets, dtype=np.int64)
        pair_seq_ids = np.asarray(pair_seq_ids, dtype=np.int64)
        pair_mask_ids = np.asarray(pair_mask_ids, dtype=object)
        print(
            f"[selector] skip={skip_count} bank={len(bank_ids)} feature_mode={args.feature_mode} "
            f"train_pairs={pair_x.shape[0]} eval_sequences={eval_pivot.shape[0]}",
            flush=True,
        )

        train_mask = np.isin(pair_seq_ids, train_ids)
        holdout_mask = np.isin(pair_seq_ids, holdout_ids)
        models = fit_models(args.seed)
        if args.models:
            models = {name: model for name, model in models.items() if name in set(args.models)}
            if not models:
                raise ValueError(f"No valid models selected from {args.models}")
        fitted_models: dict[str, Any] = {}
        holdout_scores: dict[str, float] = {}

        for model_name, model in models.items():
            print(f"[selector] fitting {model_name} for skip={skip_count}", flush=True)
            if model_name.endswith("_reg"):
                model.fit(pair_x[train_mask], delta_y[train_mask])
                holdout_losses = []
                for seq_id in holdout_ids.tolist():
                    seq_mask = pair_seq_ids[holdout_mask] == int(seq_id)
                    seq_x = pair_x[holdout_mask][seq_mask]
                    seq_mask_ids = pair_mask_ids[holdout_mask][seq_mask]
                    pred = model.predict(seq_x)
                    chosen = str(seq_mask_ids[int(np.argmin(pred))])
                    holdout_losses.append(float(calib_pivot.loc[int(seq_id), chosen] - calib_global.loc[int(seq_id)]))
                holdout_scores[model_name] = float(np.mean(holdout_losses))
            else:
                model.fit(pair_x[train_mask], improve_y[train_mask])
                holdout_losses = []
                for seq_id in holdout_ids.tolist():
                    seq_mask = pair_seq_ids[holdout_mask] == int(seq_id)
                    seq_x = pair_x[holdout_mask][seq_mask]
                    seq_mask_ids = pair_mask_ids[holdout_mask][seq_mask]
                    probs = model.predict_proba(seq_x)[:, 1]
                    best_idx = int(np.argmax(probs))
                    chosen = str(seq_mask_ids[best_idx]) if float(probs[best_idx]) > 0.5 else global_static
                    holdout_losses.append(float(calib_pivot.loc[int(seq_id), chosen] - calib_global.loc[int(seq_id)]))
                holdout_scores[model_name] = float(np.mean(holdout_losses))
            fitted_models[model_name] = model
            print(
                f"[selector] holdout {model_name} skip={skip_count} delta={holdout_scores[model_name]:.6f}",
                flush=True,
            )

        best_model_name = min(holdout_scores.items(), key=lambda item: item[1])[0]
        for model_name, score in holdout_scores.items():
            selection_rows.append(
                {
                    "output_tag": args.output_tag,
                    "feature_mode": args.feature_mode,
                    "skip_count": skip_count,
                    "bank_size": args.bank_size,
                    "model_name": model_name,
                    "holdout_delta_to_calib_global_static": score,
                    "num_train_sequences": int(train_ids.size),
                    "num_holdout_sequences": int(holdout_ids.size),
                }
            )
        selection_rows.append(
            {
                "output_tag": args.output_tag,
                "feature_mode": args.feature_mode,
                "skip_count": skip_count,
                "bank_size": args.bank_size,
                "model_name": "best_model",
                "selected_model_name": best_model_name,
                "holdout_delta_to_calib_global_static": float(holdout_scores[best_model_name]),
                "num_train_sequences": int(train_ids.size),
                "num_holdout_sequences": int(holdout_ids.size),
            }
        )

        refit_models = fit_models(args.seed)
        if args.models:
            refit_models = {name: model for name, model in refit_models.items() if name in set(args.models)}
        for model_name, model in refit_models.items():
            print(f"[selector] refit {model_name} on full calibration for skip={skip_count}", flush=True)
            if model_name.endswith("_reg"):
                model.fit(pair_x, delta_y)
            else:
                model.fit(pair_x, improve_y)
            fitted_models[model_name] = model

        for model_name, model in fitted_models.items():
            for seq_id in eval_pivot.index.tolist():
                row = eval_feature_subset.loc[int(seq_id)]
                candidate_features = np.stack(
                    [
                        make_pair_features(row, parse_mask_id(mask_id, num_blocks), global_static_mask, args.feature_mode)
                        for mask_id in bank_ids
                    ],
                    axis=0,
                )
                if model_name.endswith("_reg"):
                    pred = model.predict(candidate_features)
                    best_idx = int(np.argmin(pred))
                    predicted_best_delta = float(pred[best_idx])
                    chosen = bank_ids[best_idx]
                else:
                    probs = model.predict_proba(candidate_features)[:, 1]
                    best_idx = int(np.argmax(probs))
                    predicted_best_delta = float(-(probs[best_idx]))
                    chosen = bank_ids[best_idx] if float(probs[best_idx]) > 0.5 else global_static
                loss = float(eval_pivot.loc[int(seq_id), chosen])
                eval_rows.append(
                    {
                        "output_tag": args.output_tag,
                        "feature_mode": args.feature_mode,
                        "skip_count": skip_count,
                        "bank_size": args.bank_size,
                        "sequence_idx": int(seq_id),
                        "method": model_name,
                        "selected_mask_id": chosen,
                        "predicted_best_delta": predicted_best_delta,
                        "continuation_loss": loss,
                        "delta_to_calib_global_static": float(loss - eval_global.loc[int(seq_id)]),
                        "delta_to_oracle": float(loss - eval_oracle.loc[int(seq_id)]),
                        "delta_to_bank_upper_bound": float(loss - eval_bank_upper.loc[int(seq_id), "bank_upper_bound_loss"]),
                        "selected_nonstatic": float(chosen != global_static),
                    }
                )

        best_model = fitted_models[best_model_name]
        for seq_id in eval_pivot.index.tolist():
            row = eval_feature_subset.loc[int(seq_id)]
            candidate_features = np.stack(
                [
                    make_pair_features(row, parse_mask_id(mask_id, num_blocks), global_static_mask, args.feature_mode)
                    for mask_id in bank_ids
                ],
                axis=0,
            )
            if best_model_name.endswith("_reg"):
                pred = best_model.predict(candidate_features)
                best_idx = int(np.argmin(pred))
                predicted_best_delta = float(pred[best_idx])
                chosen = bank_ids[best_idx]
            else:
                probs = best_model.predict_proba(candidate_features)[:, 1]
                best_idx = int(np.argmax(probs))
                predicted_best_delta = float(-(probs[best_idx]))
                chosen = bank_ids[best_idx] if float(probs[best_idx]) > 0.5 else global_static
            loss = float(eval_pivot.loc[int(seq_id), chosen])
            eval_rows.append(
                {
                    "output_tag": args.output_tag,
                    "feature_mode": args.feature_mode,
                    "skip_count": skip_count,
                    "bank_size": args.bank_size,
                    "sequence_idx": int(seq_id),
                    "method": "best_model",
                    "selected_mask_id": chosen,
                    "predicted_best_delta": predicted_best_delta,
                    "continuation_loss": loss,
                    "delta_to_calib_global_static": float(loss - eval_global.loc[int(seq_id)]),
                    "delta_to_oracle": float(loss - eval_oracle.loc[int(seq_id)]),
                    "delta_to_bank_upper_bound": float(loss - eval_bank_upper.loc[int(seq_id), "bank_upper_bound_loss"]),
                    "selected_nonstatic": float(chosen != global_static),
                }
            )

        baseline = bank_eval_df[bank_eval_df["skip_count"] == skip_count][
            [
                "sequence_idx",
                "bank_upper_bound_loss",
                "calib_global_static_loss",
                "oracle_loss",
            ]
        ].copy()
        baseline["output_tag"] = args.output_tag
        baseline["skip_count"] = skip_count
        baseline["bank_size"] = args.bank_size
        baseline_rows = []
        for _, row in baseline.iterrows():
            baseline_rows.extend(
                [
                    {
                        "output_tag": args.output_tag,
                        "feature_mode": args.feature_mode,
                        "skip_count": skip_count,
                        "bank_size": args.bank_size,
                        "sequence_idx": int(row["sequence_idx"]),
                        "method": "calib_global_static",
                        "selected_mask_id": global_static,
                        "predicted_best_delta": 0.0,
                        "continuation_loss": float(row["calib_global_static_loss"]),
                        "delta_to_calib_global_static": 0.0,
                        "delta_to_oracle": float(row["calib_global_static_loss"] - row["oracle_loss"]),
                        "delta_to_bank_upper_bound": float(row["calib_global_static_loss"] - row["bank_upper_bound_loss"]),
                        "selected_nonstatic": 0.0,
                    },
                    {
                        "output_tag": args.output_tag,
                        "feature_mode": args.feature_mode,
                        "skip_count": skip_count,
                        "bank_size": args.bank_size,
                        "sequence_idx": int(row["sequence_idx"]),
                        "method": "bank_upper_bound",
                        "selected_mask_id": "",
                        "predicted_best_delta": float(row["bank_upper_bound_loss"] - row["calib_global_static_loss"]),
                        "continuation_loss": float(row["bank_upper_bound_loss"]),
                        "delta_to_calib_global_static": float(row["bank_upper_bound_loss"] - row["calib_global_static_loss"]),
                        "delta_to_oracle": float(row["bank_upper_bound_loss"] - row["oracle_loss"]),
                        "delta_to_bank_upper_bound": 0.0,
                        "selected_nonstatic": 1.0,
                    },
                ]
            )
        eval_rows.extend(baseline_rows)

    selection_df = pd.DataFrame(selection_rows)
    eval_df = pd.DataFrame(eval_rows)
    selection_df.to_csv(delta_dir / f"{args.output_tag}_model_selection.csv", index=False)
    eval_df.to_csv(delta_dir / f"{args.output_tag}_eval_per_sequence.csv", index=False)

    for (skip_count, method), subset in eval_df.groupby(["skip_count", "method"], sort=True):
        for metric in [
            "continuation_loss",
            "delta_to_calib_global_static",
            "delta_to_oracle",
            "delta_to_bank_upper_bound",
            "selected_nonstatic",
        ]:
            ci = bootstrap_summary(subset[metric].to_numpy(dtype=np.float64), seed=args.seed)
            summary_rows.append(
                {
                    "output_tag": args.output_tag,
                    "feature_mode": args.feature_mode,
                    "skip_count": skip_count,
                    "bank_size": args.bank_size,
                    "method": method,
                    "metric": metric,
                    **ci,
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(delta_dir / f"{args.output_tag}_eval_summary.csv", index=False)

    for metric in ["delta_to_calib_global_static", "delta_to_oracle", "selected_nonstatic"]:
        plt.figure(figsize=(7, 4))
        subset = summary_df[summary_df["metric"] == metric]
        for method in [
            "calib_global_static",
            "rf_delta_reg",
            "hgb_delta_reg",
            "rf_improve_cls",
            "logreg_improve_cls",
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
        plt.savefig(plot_dir / f"delta_selector_{args.output_tag}_{metric}.png", dpi=160)
        plt.close()


if __name__ == "__main__":
    main()
