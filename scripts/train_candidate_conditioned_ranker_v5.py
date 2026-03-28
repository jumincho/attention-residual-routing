#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attnres_routing.analysis import bootstrap_mean_ci  # noqa: E402


def column_fill_values(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        finite = arr[np.isfinite(arr)]
        fill = float(finite.mean()) if finite.size else 0.0
        return np.asarray([fill], dtype=np.float64)
    masked = np.where(np.isfinite(arr), arr, np.nan)
    fill = np.nanmean(masked, axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.0)
    return fill.astype(np.float64, copy=False)


def sanitize_array(x: np.ndarray, fill: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        fill_value = float(fill[0]) if fill is not None else float(column_fill_values(arr)[0])
        return np.nan_to_num(arr, nan=fill_value, posinf=fill_value, neginf=fill_value)
    if fill is None:
        fill = column_fill_values(arr)
    bad = ~np.isfinite(arr)
    if not bad.any():
        return arr
    arr = arr.copy()
    rows, cols = np.where(bad)
    arr[rows, cols] = fill[cols]
    return arr


def concat_frames(base_dir: Path, tags: list[str], suffix: str) -> pd.DataFrame:
    frames = [pd.read_csv(base_dir / f"{tag}_{suffix}.csv") for tag in tags]
    return pd.concat(frames, ignore_index=True)


def parse_mask_id(mask_id: str, num_blocks: int) -> np.ndarray:
    mask = np.zeros(num_blocks, dtype=np.float32)
    _, kept = mask_id.split(":", 1)
    if kept.strip():
        for token in kept.split(","):
            mask[int(token) - 1] = 1.0
    return mask


def parse_json_vec(value: str | float | int | None) -> np.ndarray:
    if not isinstance(value, str) or not value:
        return np.zeros(0, dtype=np.float32)
    return np.asarray(json.loads(value), dtype=np.float32)


def parse_hidden_row(row: pd.Series) -> np.ndarray:
    parts = []
    for column in sorted(row.index.tolist()):
        if column.endswith("_mean_json") or column.endswith("_final_json"):
            parts.append(parse_json_vec(row[column]))
        elif column.endswith("_norm_mean") or column.endswith("_norm_std"):
            if not pd.isna(row[column]):
                parts.append(np.asarray([float(row[column])], dtype=np.float32))
    return np.concatenate(parts, axis=0) if parts else np.zeros(0, dtype=np.float32)


def parse_difficulty_row(row: pd.Series) -> np.ndarray:
    columns = [
        "prompt_surprisal_mean",
        "prompt_surprisal_std",
        "prompt_surprisal_max",
        "prompt_ppl",
        "unique_token_ratio",
        "adjacent_repeat_fraction",
        "max_repeat_run_fraction",
    ]
    values = []
    for column in columns:
        value = row[column] if column in row.index and not pd.isna(row[column]) else 0.0
        values.append(float(value))
    return np.asarray(values, dtype=np.float32)


def prompt_feature_vector(row: pd.Series, feature_mode: str) -> np.ndarray:
    combined = parse_json_vec(row["prompt_scores_json"])[1:-1]
    attn = parse_json_vec(row["prompt_scores_attn_json"])[1:-1]
    mlp = parse_json_vec(row["prompt_scores_mlp_json"])[1:-1]
    chunks = parse_json_vec(row["prompt_chunk_utilities_json"]).reshape(-1, combined.size + 2)[:, 1:-1]
    chunk_mean = chunks.mean(axis=0) if chunks.size else np.zeros_like(combined)
    chunk_std = chunks.std(axis=0) if chunks.size else np.zeros_like(combined)
    attnres = np.concatenate(
        [
            combined,
            attn,
            mlp,
            chunk_mean,
            chunk_std,
            np.asarray(
                [
                    float(row.get("stability_spearman", 0.0)),
                    float(row.get("stability_top3_jaccard", 0.0)),
                    float(row.get("prompt_margin", 0.0)),
                    float(row.get("prompt_depth_entropy", 0.0)),
                    float(row.get("prompt_support_size", 0.0)),
                ],
                dtype=np.float32,
            ),
        ],
        axis=0,
    )
    hidden = parse_hidden_row(row)
    difficulty = parse_difficulty_row(row)
    if feature_mode == "attnres":
        return attnres
    if feature_mode == "hidden":
        return hidden
    if feature_mode == "difficulty":
        return difficulty
    if feature_mode in {"combined", "attnres_hidden"}:
        return np.concatenate([attnres, hidden], axis=0)
    if feature_mode in {"full", "attnres_hidden_difficulty"}:
        return np.concatenate([attnres, hidden, difficulty], axis=0)
    if feature_mode == "hidden_difficulty":
        return np.concatenate([hidden, difficulty], axis=0)
    raise ValueError(f"Unsupported feature_mode: {feature_mode}")


def candidate_feature_vector(
    row: pd.Series,
    mask: np.ndarray,
    global_static_mask: np.ndarray,
) -> np.ndarray:
    mid_mask = mask[:-1]
    global_mid = global_static_mask[:-1]
    delta_mid = mid_mask - global_mid
    combined = parse_json_vec(row["prompt_scores_json"])[1:-1]
    attn = parse_json_vec(row["prompt_scores_attn_json"])[1:-1]
    mlp = parse_json_vec(row["prompt_scores_mlp_json"])[1:-1]
    keep_combined = combined * mid_mask
    keep_attn = attn * mid_mask
    keep_mlp = mlp * mid_mask
    drop_combined = combined * (1.0 - mid_mask)
    drop_attn = attn * (1.0 - mid_mask)
    drop_mlp = mlp * (1.0 - mid_mask)
    edit_count = float(np.abs(delta_mid).sum() / 2.0)
    return np.concatenate(
        [
            mid_mask.astype(np.float32),
            delta_mid.astype(np.float32),
            np.asarray(
                [
                    float(mid_mask.sum()),
                    edit_count,
                    float(np.allclose(delta_mid, 0.0)),
                    float(np.dot(combined, mid_mask)),
                    float(np.dot(attn, mid_mask)),
                    float(np.dot(mlp, mid_mask)),
                    float(keep_combined.sum()),
                    float(keep_attn.sum()),
                    float(keep_mlp.sum()),
                    float(drop_combined.sum()),
                    float(drop_attn.sum()),
                    float(drop_mlp.sum()),
                ],
                dtype=np.float32,
            ),
        ],
        axis=0,
    )


def pair_feature_vector(prompt_vec: np.ndarray, cand_vec: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            prompt_vec,
            cand_vec,
            np.asarray(
                [
                    float(np.linalg.norm(prompt_vec)),
                    float(np.linalg.norm(cand_vec)),
                ],
                dtype=np.float32,
            ),
        ],
        axis=0,
    )


def build_feature_table(
    feature_df: pd.DataFrame,
    oracle_df: pd.DataFrame,
    hidden_df: pd.DataFrame | None,
) -> pd.DataFrame:
    if "document_idx" not in feature_df.columns:
        feature_df = feature_df.copy()
        feature_df["document_idx"] = feature_df["sequence_idx"]
    if "document_idx" not in oracle_df.columns:
        oracle_df = oracle_df.copy()
        oracle_df["document_idx"] = oracle_df["sequence_idx"]
    merge_keys = ["sequence_idx", "split", "document_idx"]
    keep_cols = merge_keys + [
        "skip_count",
        "stability_spearman",
        "stability_top3_jaccard",
        "prompt_margin",
    ]
    merged = feature_df.merge(
        oracle_df[keep_cols].drop_duplicates(),
        on=merge_keys,
        how="left",
    )
    if hidden_df is not None:
        if "document_idx" not in hidden_df.columns:
            hidden_df = hidden_df.copy()
            hidden_df["document_idx"] = hidden_df["sequence_idx"]
        merged = merged.merge(
            hidden_df.drop_duplicates(subset=merge_keys),
            on=merge_keys,
            how="left",
        )
    return merged.drop_duplicates(subset=merge_keys + ["skip_count"]).reset_index(drop=True)


@dataclass
class RankerDataset:
    prompt_vectors: dict[int, np.ndarray]
    rows: pd.DataFrame
    bank_ids: list[str]
    global_static: str
    global_static_mask: np.ndarray
    num_blocks: int


def build_ranker_dataset(
    feature_df: pd.DataFrame,
    mask_df: pd.DataFrame,
    bank_df: pd.DataFrame,
    bank_size: int,
    skip_count: int,
    feature_mode: str,
) -> RankerDataset:
    bank_subset = bank_df[(bank_df["bank_size"] == bank_size) & (bank_df["skip_count"] == skip_count)].sort_values("bank_rank")
    bank_ids = bank_subset["mask_id"].tolist()
    num_blocks = len(parse_json_vec(feature_df.iloc[0]["prompt_scores_json"])) - 1
    global_static = str(bank_subset[bank_subset["reasons"].str.contains("calib_global_static")].iloc[0]["mask_id"])
    global_static_mask = parse_mask_id(global_static, num_blocks)

    feature_subset = feature_df[feature_df["skip_count"] == skip_count].copy()
    mask_subset = mask_df[(mask_df["skip_count"] == skip_count) & (mask_df["method"] == "exhaustive_mask")].copy()
    pivot = mask_subset.pivot(index="sequence_idx", columns="mask_id", values="continuation_loss")
    rows = []
    prompt_vectors = {}
    for _, row in feature_subset.iterrows():
        seq_id = int(row["sequence_idx"])
        prompt_vec = prompt_feature_vector(row, feature_mode)
        prompt_vectors[seq_id] = prompt_vec
        global_loss = float(pivot.loc[seq_id, global_static])
        for mask_id in bank_ids:
            mask = parse_mask_id(mask_id, num_blocks)
            cand_vec = candidate_feature_vector(row, mask, global_static_mask)
            rows.append(
                {
                    "sequence_idx": seq_id,
                    "document_idx": int(row["document_idx"]),
                    "mask_id": mask_id,
                    "global_static_mask_id": global_static,
                    "actual_loss": float(pivot.loc[seq_id, mask_id]),
                    "delta_to_global_static": float(pivot.loc[seq_id, mask_id] - global_loss),
                    "pair_features": pair_feature_vector(prompt_vec, cand_vec),
                    "candidate_features": cand_vec,
                }
            )
    return RankerDataset(
        prompt_vectors=prompt_vectors,
        rows=pd.DataFrame(rows),
        bank_ids=bank_ids,
        global_static=global_static,
        global_static_mask=global_static_mask,
        num_blocks=num_blocks,
    )


class LateFusionMLP(nn.Module):
    def __init__(self, prompt_dim: int, cand_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.prompt_net = nn.Sequential(
            nn.Linear(prompt_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.cand_net = nn.Sequential(
            nn.Linear(cand_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, prompt_x: torch.Tensor, cand_x: torch.Tensor) -> torch.Tensor:
        prompt_h = self.prompt_net(prompt_x)
        cand_h = self.cand_net(cand_x)
        joint = torch.cat([prompt_h, cand_h, prompt_h * cand_h], dim=-1)
        return self.head(joint).squeeze(-1)


def train_mlp(
    train_prompt: np.ndarray,
    train_cand: np.ndarray,
    train_y: np.ndarray,
    seed: int,
    epochs: int = 20,
    batch_size: int = 2048,
) -> tuple[LateFusionMLP, StandardScaler, StandardScaler]:
    torch.manual_seed(seed)
    prompt_scaler = StandardScaler().fit(train_prompt)
    cand_scaler = StandardScaler().fit(train_cand)
    x_prompt = torch.tensor(prompt_scaler.transform(train_prompt), dtype=torch.float32)
    x_cand = torch.tensor(cand_scaler.transform(train_cand), dtype=torch.float32)
    y = torch.tensor(train_y, dtype=torch.float32)

    model = LateFusionMLP(prompt_dim=x_prompt.size(1), cand_dim=x_cand.size(1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for _epoch in range(epochs):
        perm = torch.randperm(x_prompt.size(0))
        for start in range(0, x_prompt.size(0), batch_size):
            idx = perm[start : start + batch_size]
            pred = model(x_prompt[idx], x_cand[idx])
            loss = torch.nn.functional.mse_loss(pred, y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.eval(), prompt_scaler, cand_scaler


def predict_mlp(
    model: LateFusionMLP,
    prompt_scaler: StandardScaler,
    cand_scaler: StandardScaler,
    prompt_x: np.ndarray,
    cand_x: np.ndarray,
) -> np.ndarray:
    with torch.no_grad():
        prompt_t = torch.tensor(prompt_scaler.transform(prompt_x), dtype=torch.float32)
        cand_t = torch.tensor(cand_scaler.transform(cand_x), dtype=torch.float32)
        pred = model(prompt_t, cand_t).cpu().numpy()
    return pred.astype(np.float64)


def knn_predict(
    train_prompt: np.ndarray,
    train_mask_ids: np.ndarray,
    train_y: np.ndarray,
    eval_prompt: np.ndarray,
    eval_mask_ids: np.ndarray,
    k: int = 16,
) -> np.ndarray:
    train_fill = column_fill_values(train_prompt)
    train_prompt = sanitize_array(train_prompt, fill=train_fill)
    eval_prompt = sanitize_array(eval_prompt, fill=train_fill)
    scaler = StandardScaler().fit(train_prompt)
    train_z = sanitize_array(scaler.transform(train_prompt))
    eval_z = sanitize_array(scaler.transform(eval_prompt), fill=column_fill_values(train_z))
    nbrs = NearestNeighbors(n_neighbors=min(k, train_z.shape[0]), metric="euclidean").fit(train_z)
    distances, indices = nbrs.kneighbors(eval_z)
    preds = np.zeros(eval_prompt.shape[0], dtype=np.float64)
    for row_idx in range(eval_prompt.shape[0]):
        neighbor_ids = indices[row_idx]
        neighbor_mask = train_mask_ids[neighbor_ids] == eval_mask_ids[row_idx]
        if neighbor_mask.any():
            preds[row_idx] = float(train_y[neighbor_ids[neighbor_mask]].mean())
        else:
            same_mask = train_mask_ids == eval_mask_ids[row_idx]
            preds[row_idx] = float(train_y[same_mask].mean()) if same_mask.any() else float(train_y.mean())
    return preds


def bootstrap_summary(values: np.ndarray) -> dict[str, float]:
    return bootstrap_mean_ci(values.astype(np.float64), seed=42)


def evaluate_selected_masks(
    rows_df: pd.DataFrame,
    score_col: str,
    global_static: str,
    bank_upper_df: pd.DataFrame | None = None,
    output_tag: str = "",
    feature_mode: str = "",
    skip_count: int = 0,
    bank_size: int = 0,
    model_name: str = "",
) -> pd.DataFrame:
    per_sequence = []
    global_lookup = (
        rows_df[rows_df["mask_id"] == global_static][["sequence_idx", "actual_loss"]]
        .rename(columns={"actual_loss": "global_static_loss"})
        .set_index("sequence_idx")
    )
    for seq_id, group in rows_df.groupby("sequence_idx", sort=False):
        best_pos = int(np.argmin(group[score_col].to_numpy(dtype=np.float64)))
        chosen = group.iloc[best_pos]
        actual_delta = float(chosen["actual_loss"] - global_lookup.loc[int(seq_id), "global_static_loss"])
        row = {
            "output_tag": output_tag,
            "feature_mode": feature_mode,
            "skip_count": skip_count,
            "bank_size": bank_size,
            "model_name": model_name,
            "sequence_idx": int(seq_id),
            "selected_mask_id": str(chosen["mask_id"]),
            "predicted_delta": float(chosen[score_col]),
            "actual_delta_to_static": actual_delta,
            "improved_over_static": float(actual_delta < 0.0),
        }
        if bank_upper_df is not None:
            row["delta_to_bank_upper_bound"] = float(
                chosen["actual_loss"] - bank_upper_df.loc[int(seq_id), "bank_upper_bound_loss"]
            )
            row["oracle_in_bank_match"] = float(
                str(chosen["mask_id"]) == str(bank_upper_df.loc[int(seq_id), "bank_best_mask_id"])
            )
        per_sequence.append(row)
    return pd.DataFrame(per_sequence)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-tag", type=str, required=True)
    parser.add_argument("--bank-size", type=int, required=True)
    parser.add_argument("--train-tags", type=str, nargs="+", required=True)
    parser.add_argument("--eval-tags", type=str, nargs="+", required=True)
    parser.add_argument("--output-tag", type=str, required=True)
    parser.add_argument(
        "--feature-mode",
        choices=[
            "attnres",
            "hidden",
            "difficulty",
            "combined",
            "attnres_hidden",
            "hidden_difficulty",
            "full",
            "attnres_hidden_difficulty",
        ],
        default="combined",
    )
    parser.add_argument("--hidden-train-tags", type=str, nargs="*", default=None)
    parser.add_argument("--hidden-eval-tags", type=str, nargs="*", default=None)
    parser.add_argument("--train-doc-frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    bank_dir = ROOT / "results" / "bank_hygiene"
    oracle_dir = ROOT / "results" / "oracles"
    rich_dir = ROOT / "results" / "rich_features"
    out_dir = ROOT / "results" / "ranker_v5"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = ROOT / "results" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    bank_df = pd.read_csv(bank_dir / f"{args.bank_tag}_candidate_bank.csv")
    bank_eval_df = pd.read_csv(bank_dir / f"{args.bank_tag}_per_sequence.csv")
    train_features = concat_frames(oracle_dir, args.train_tags, "sequence_features")
    train_oracle = concat_frames(oracle_dir, args.train_tags, "oracle_mask_alignment")
    train_masks = concat_frames(oracle_dir, args.train_tags, "exhaustive_mask_losses")
    eval_features = concat_frames(oracle_dir, args.eval_tags, "sequence_features")
    eval_oracle = concat_frames(oracle_dir, args.eval_tags, "oracle_mask_alignment")
    eval_masks = concat_frames(oracle_dir, args.eval_tags, "exhaustive_mask_losses")
    train_hidden = concat_frames(rich_dir, args.hidden_train_tags, "hidden_prompt_features") if args.hidden_train_tags else None
    eval_hidden = concat_frames(rich_dir, args.hidden_eval_tags, "hidden_prompt_features") if args.hidden_eval_tags else None

    train_feature_table = build_feature_table(train_features, train_oracle, train_hidden)
    eval_feature_table = build_feature_table(eval_features, eval_oracle, eval_hidden)

    rng = np.random.default_rng(args.seed)
    eval_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []

    for skip_count in sorted(bank_df["skip_count"].unique().tolist()):
        train_ds = build_ranker_dataset(train_feature_table, train_masks, bank_df, args.bank_size, skip_count, args.feature_mode)
        eval_ds = build_ranker_dataset(eval_feature_table, eval_masks, bank_df, args.bank_size, skip_count, args.feature_mode)

        train_docs = np.asarray(sorted(train_ds.rows["document_idx"].unique().tolist()), dtype=int)
        rng.shuffle(train_docs)
        split_at = max(1, int(round(len(train_docs) * args.train_doc_frac)))
        fit_docs = set(train_docs[:split_at].tolist())
        dev_docs = set(train_docs[split_at:].tolist()) or fit_docs

        train_rows = train_ds.rows[train_ds.rows["document_idx"].isin(fit_docs)].copy()
        dev_rows = train_ds.rows[train_ds.rows["document_idx"].isin(dev_docs)].copy()
        eval_rows_df = eval_ds.rows.copy()

        x_train_pair = np.stack(train_rows["pair_features"].to_list(), axis=0)
        x_dev_pair = np.stack(dev_rows["pair_features"].to_list(), axis=0)
        x_eval_pair = np.stack(eval_rows_df["pair_features"].to_list(), axis=0)
        x_train_prompt = np.stack([train_ds.prompt_vectors[int(seq)] for seq in train_rows["sequence_idx"].tolist()], axis=0)
        x_dev_prompt = np.stack([train_ds.prompt_vectors[int(seq)] for seq in dev_rows["sequence_idx"].tolist()], axis=0)
        x_eval_prompt = np.stack([eval_ds.prompt_vectors[int(seq)] for seq in eval_rows_df["sequence_idx"].tolist()], axis=0)
        x_train_cand = np.stack(train_rows["candidate_features"].to_list(), axis=0)
        x_dev_cand = np.stack(dev_rows["candidate_features"].to_list(), axis=0)
        x_eval_cand = np.stack(eval_rows_df["candidate_features"].to_list(), axis=0)
        y_train = train_rows["delta_to_global_static"].to_numpy(dtype=np.float64)

        pair_fill = column_fill_values(x_train_pair)
        prompt_fill = column_fill_values(x_train_prompt)
        cand_fill = column_fill_values(x_train_cand)
        y_fill = column_fill_values(y_train)

        x_train_pair = sanitize_array(x_train_pair, fill=pair_fill)
        x_dev_pair = sanitize_array(x_dev_pair, fill=pair_fill)
        x_eval_pair = sanitize_array(x_eval_pair, fill=pair_fill)
        x_train_prompt = sanitize_array(x_train_prompt, fill=prompt_fill)
        x_dev_prompt = sanitize_array(x_dev_prompt, fill=prompt_fill)
        x_eval_prompt = sanitize_array(x_eval_prompt, fill=prompt_fill)
        x_train_cand = sanitize_array(x_train_cand, fill=cand_fill)
        x_dev_cand = sanitize_array(x_dev_cand, fill=cand_fill)
        x_eval_cand = sanitize_array(x_eval_cand, fill=cand_fill)
        y_train = sanitize_array(y_train, fill=y_fill)

        models_eval: dict[str, np.ndarray] = {}
        models_dev: dict[str, np.ndarray] = {}
        hgb = HistGradientBoostingRegressor(max_depth=6, max_iter=300, learning_rate=0.05, random_state=args.seed)
        hgb.fit(x_train_pair, y_train)
        models_eval["hgb_pair"] = hgb.predict(x_eval_pair)
        models_dev["hgb_pair"] = hgb.predict(x_dev_pair)

        rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=14,
            min_samples_leaf=2,
            random_state=args.seed,
            n_jobs=-1,
        )
        rf.fit(x_train_pair, y_train)
        models_eval["rf_pair"] = rf.predict(x_eval_pair)
        models_dev["rf_pair"] = rf.predict(x_dev_pair)

        models_eval["knn_prompt"] = knn_predict(
            x_train_prompt,
            train_rows["mask_id"].to_numpy(dtype=object),
            y_train,
            x_eval_prompt,
            eval_rows_df["mask_id"].to_numpy(dtype=object),
        )
        models_dev["knn_prompt"] = knn_predict(
            x_train_prompt,
            train_rows["mask_id"].to_numpy(dtype=object),
            y_train,
            x_dev_prompt,
            dev_rows["mask_id"].to_numpy(dtype=object),
        )

        mlp_model, prompt_scaler, cand_scaler = train_mlp(x_train_prompt, x_train_cand, y_train, seed=args.seed)
        models_eval["mlp_late_fusion"] = predict_mlp(mlp_model, prompt_scaler, cand_scaler, x_eval_prompt, x_eval_cand)
        models_dev["mlp_late_fusion"] = predict_mlp(mlp_model, prompt_scaler, cand_scaler, x_dev_prompt, x_dev_cand)

        bank_upper = (
            bank_eval_df[(bank_eval_df["bank_size"] == args.bank_size) & (bank_eval_df["skip_count"] == skip_count)][
                ["sequence_idx", "bank_upper_bound_loss", "bank_best_mask_id"]
            ]
            .drop_duplicates(subset=["sequence_idx"])
            .set_index("sequence_idx")
        )

        for model_name, preds in models_dev.items():
            dev_rows_scored = dev_rows.copy()
            dev_rows_scored[model_name] = preds
            dev_selected = evaluate_selected_masks(
                rows_df=dev_rows_scored,
                score_col=model_name,
                global_static=train_ds.global_static,
                bank_upper_df=None,
                output_tag=args.output_tag,
                feature_mode=args.feature_mode,
                skip_count=skip_count,
                bank_size=args.bank_size,
                model_name=model_name,
            )
            for metric_name, values in [
                ("dev_delta_to_static", dev_selected["actual_delta_to_static"].to_numpy(dtype=np.float64)),
                ("dev_fraction_improved", dev_selected["improved_over_static"].to_numpy(dtype=np.float64)),
            ]:
                selection_rows.append(
                    {
                        "output_tag": args.output_tag,
                        "feature_mode": args.feature_mode,
                        "skip_count": skip_count,
                        "bank_size": args.bank_size,
                        "model_name": model_name,
                        "metric": metric_name,
                        **bootstrap_summary(values),
                    }
                )

        for model_name, preds in models_eval.items():
            eval_rows_df[model_name] = preds
            per_sequence_df = evaluate_selected_masks(
                rows_df=eval_rows_df,
                score_col=model_name,
                global_static=eval_ds.global_static,
                bank_upper_df=bank_upper,
                output_tag=args.output_tag,
                feature_mode=args.feature_mode,
                skip_count=skip_count,
                bank_size=args.bank_size,
                model_name=model_name,
            )
            eval_rows.extend(per_sequence_df.to_dict(orient="records"))
            for metric_name, values in [
                ("delta_to_static", per_sequence_df["actual_delta_to_static"].to_numpy(dtype=np.float64)),
                ("fraction_improved", per_sequence_df["improved_over_static"].to_numpy(dtype=np.float64)),
                ("delta_to_bank_upper_bound", per_sequence_df["delta_to_bank_upper_bound"].to_numpy(dtype=np.float64)),
                ("oracle_in_bank_match", per_sequence_df["oracle_in_bank_match"].to_numpy(dtype=np.float64)),
            ]:
                summary_rows.append(
                    {
                        "output_tag": args.output_tag,
                        "feature_mode": args.feature_mode,
                        "skip_count": skip_count,
                        "bank_size": args.bank_size,
                        "model_name": model_name,
                        "metric": metric_name,
                        **bootstrap_summary(values),
                    }
                )

    eval_df = pd.DataFrame(eval_rows)
    summary_df = pd.DataFrame(summary_rows)
    selection_df = pd.DataFrame(selection_rows)
    eval_df.to_csv(out_dir / f"{args.output_tag}_{args.feature_mode}_per_sequence.csv", index=False)
    summary_df.to_csv(out_dir / f"{args.output_tag}_{args.feature_mode}_summary.csv", index=False)
    if not selection_df.empty:
        selection_df.to_csv(out_dir / f"{args.output_tag}_{args.feature_mode}_model_selection.csv", index=False)

    if not summary_df.empty:
        plt.figure(figsize=(7, 4))
        for skip_count in sorted(summary_df["skip_count"].unique().tolist()):
            subset = summary_df[
                (summary_df["skip_count"] == skip_count) & (summary_df["metric"] == "delta_to_static")
            ]
            plt.plot(subset["model_name"], subset["mean"], marker="o", label=f"skip={skip_count}")
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("delta to global static")
        plt.tight_layout()
        plt.legend()
        plt.savefig(plot_dir / f"ranker_v5_{args.output_tag}_{args.feature_mode}_delta_to_static.png", dpi=160)
        plt.close()


if __name__ == "__main__":
    main()
