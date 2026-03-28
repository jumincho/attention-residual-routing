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
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attnres_routing.analysis import bootstrap_mean_ci  # noqa: E402


ATTNRES_MODES = {
    "attnres",
    "attnres_hidden",
    "full",
}
HIDDEN_MODES = {
    "hidden",
    "attnres_hidden",
    "full",
}
DIFFICULTY_MODES = {
    "difficulty",
    "full",
}


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


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    try:
        return float(value)
    except Exception:
        return default


def parse_hidden_row(row: pd.Series) -> np.ndarray:
    parts = []
    for column in sorted(row.index.tolist()):
        if column.endswith("_mean_json") or column.endswith("_final_json"):
            parts.append(parse_json_vec(row[column]))
        elif column.endswith("_norm_mean") or column.endswith("_norm_std"):
            parts.append(np.asarray([safe_float(row[column])], dtype=np.float32))
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
    return np.asarray([safe_float(row.get(column, 0.0)) for column in columns], dtype=np.float32)


def topk_mass(values: np.ndarray, k: int) -> float:
    if values.size == 0 or k <= 0:
        return 0.0
    k = min(k, values.size)
    return float(np.partition(values, -k)[-k:].sum())


@dataclass
class PromptInfo:
    sequence_idx: int
    document_idx: int
    combined_scores: np.ndarray
    attn_scores: np.ndarray
    mlp_scores: np.ndarray
    attnres_raw: np.ndarray
    hidden_raw: np.ndarray
    difficulty: np.ndarray
    split: str


def build_prompt_info_table(feature_df: pd.DataFrame, hidden_df: pd.DataFrame | None) -> dict[int, PromptInfo]:
    merge_keys = ["sequence_idx", "split", "document_idx"]
    base_df = feature_df.copy()
    if "document_idx" not in base_df.columns:
        base_df["document_idx"] = base_df["sequence_idx"]
    if hidden_df is not None:
        hidden_base = hidden_df.copy()
        if "document_idx" not in hidden_base.columns:
            hidden_base["document_idx"] = hidden_base["sequence_idx"]
        base_df = base_df.merge(
            hidden_base.drop_duplicates(subset=merge_keys),
            on=merge_keys,
            how="left",
        )
    prompt_infos: dict[int, PromptInfo] = {}
    dedup_df = base_df.drop_duplicates(subset=merge_keys).reset_index(drop=True)
    for _, row in dedup_df.iterrows():
        seq_id = int(row["sequence_idx"])
        combined = parse_json_vec(row["prompt_scores_json"])[1:-1]
        attn = parse_json_vec(row["prompt_scores_attn_json"])[1:-1]
        mlp = parse_json_vec(row["prompt_scores_mlp_json"])[1:-1]
        chunk_blob = parse_json_vec(row.get("prompt_chunk_utilities_json"))
        if chunk_blob.size and combined.size:
            chunk_utils = chunk_blob.reshape(-1, combined.size + 2)[:, 1:-1]
            chunk_mean = chunk_utils.mean(axis=0)
            chunk_std = chunk_utils.std(axis=0)
        else:
            chunk_mean = np.zeros_like(combined)
            chunk_std = np.zeros_like(combined)
        attnres_raw = np.concatenate(
            [
                combined,
                attn,
                mlp,
                chunk_mean,
                chunk_std,
                combined - attn,
                combined - mlp,
                attn - mlp,
                np.asarray(
                    [
                        safe_float(row.get("stability_spearman", 0.0)),
                        safe_float(row.get("stability_top3_jaccard", 0.0)),
                        safe_float(row.get("prompt_margin", 0.0)),
                        safe_float(row.get("prompt_depth_entropy", 0.0)),
                        safe_float(row.get("prompt_support_size", 0.0)),
                        topk_mass(combined, 1),
                        topk_mass(combined, 2),
                        topk_mass(combined, 3),
                        topk_mass(attn, 2),
                        topk_mass(mlp, 2),
                    ],
                    dtype=np.float32,
                ),
            ],
            axis=0,
        )
        prompt_infos[seq_id] = PromptInfo(
            sequence_idx=seq_id,
            document_idx=int(row["document_idx"]),
            combined_scores=combined.astype(np.float32),
            attn_scores=attn.astype(np.float32),
            mlp_scores=mlp.astype(np.float32),
            attnres_raw=attnres_raw.astype(np.float32),
            hidden_raw=parse_hidden_row(row),
            difficulty=parse_difficulty_row(row),
            split=str(row["split"]),
        )
    return prompt_infos


def reason_flags(reason_text: str) -> np.ndarray:
    tokens = {token.strip() for token in str(reason_text).split(",") if token.strip()}
    ordered = [
        "calib_global_static",
        "calib_top_global",
        "calib_oracle_freq",
        "calib_oracle_diverse",
        "calib_swap_1",
        "calib_swap_2",
    ]
    return np.asarray([1.0 if token in tokens else 0.0 for token in ordered], dtype=np.float32)


@dataclass
class PromptProjection:
    attnres: dict[int, np.ndarray]
    hidden: dict[int, np.ndarray]
    difficulty: dict[int, np.ndarray]


def fit_prompt_projection(
    train_infos: dict[int, PromptInfo],
    eval_infos: dict[int, PromptInfo],
    hidden_components: int,
) -> PromptProjection:
    train_hidden = [info.hidden_raw for info in train_infos.values() if info.hidden_raw.size > 0]
    eval_hidden = [info.hidden_raw for info in eval_infos.values() if info.hidden_raw.size > 0]
    hidden_dim = train_hidden[0].shape[0] if train_hidden else 0
    use_pca = hidden_dim > 0 and hidden_components > 0 and len(train_hidden) > 4
    pca_model = None
    hidden_scaler = None
    if use_pca:
        n_components = min(hidden_components, hidden_dim, len(train_hidden) - 1)
        if n_components > 0:
            hidden_scaler = StandardScaler().fit(np.stack(train_hidden, axis=0))
            hidden_train_scaled = hidden_scaler.transform(np.stack(train_hidden, axis=0))
            pca_model = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
            pca_model.fit(hidden_train_scaled)

    def project_hidden(info: PromptInfo) -> np.ndarray:
        raw = info.hidden_raw
        if raw.size == 0:
            return np.zeros(0, dtype=np.float32)
        if pca_model is None or hidden_scaler is None:
            return raw.astype(np.float32)
        return pca_model.transform(hidden_scaler.transform(raw.reshape(1, -1))).reshape(-1).astype(np.float32)

    return PromptProjection(
        attnres={seq_id: info.attnres_raw for seq_id, info in {**train_infos, **eval_infos}.items()},
        hidden={seq_id: project_hidden(info) for seq_id, info in {**train_infos, **eval_infos}.items()},
        difficulty={seq_id: info.difficulty for seq_id, info in {**train_infos, **eval_infos}.items()},
    )


def prompt_vector(seq_id: int, projection: PromptProjection, feature_mode: str) -> np.ndarray:
    parts: list[np.ndarray] = []
    if feature_mode in ATTNRES_MODES:
        parts.append(projection.attnres[seq_id])
    if feature_mode in HIDDEN_MODES:
        parts.append(projection.hidden[seq_id])
    if feature_mode in DIFFICULTY_MODES:
        parts.append(projection.difficulty[seq_id])
    if not parts:
        raise ValueError(f"Empty prompt vector for feature_mode={feature_mode}")
    return np.concatenate(parts, axis=0).astype(np.float32)


def candidate_feature_vector(
    info: PromptInfo,
    mask: np.ndarray,
    global_static_mask: np.ndarray,
    bank_meta: dict[str, Any],
) -> np.ndarray:
    mid_mask = mask[:-1]
    global_mid = global_static_mask[:-1]
    delta_mid = mid_mask - global_mid
    combined = info.combined_scores
    attn = info.attn_scores
    mlp = info.mlp_scores
    keep_positions = np.flatnonzero(mid_mask > 0.5).astype(np.float32) + 1.0
    drop_positions = np.flatnonzero(mid_mask < 0.5).astype(np.float32) + 1.0
    edit_positions = np.flatnonzero(np.abs(delta_mid) > 0.5).astype(np.float32) + 1.0

    def pos_stats(values: np.ndarray) -> tuple[float, float]:
        if values.size == 0:
            return 0.0, 0.0
        return float(values.mean()), float(values.std())

    keep_mean, keep_std = pos_stats(keep_positions)
    drop_mean, drop_std = pos_stats(drop_positions)
    edit_mean, edit_std = pos_stats(edit_positions)
    edit_count = float(np.abs(delta_mid).sum() / 2.0)
    reason_vec = reason_flags(str(bank_meta.get("reasons", "")))
    return np.concatenate(
        [
            mid_mask.astype(np.float32),
            delta_mid.astype(np.float32),
            np.asarray(
                [
                    float(mid_mask.sum()),
                    float((mid_mask < 0.5).sum()),
                    edit_count,
                    float(np.allclose(delta_mid, 0.0)),
                    float(np.dot(combined, mid_mask)),
                    float(np.dot(attn, mid_mask)),
                    float(np.dot(mlp, mid_mask)),
                    float(np.dot(combined, 1.0 - mid_mask)),
                    float(np.dot(attn, 1.0 - mid_mask)),
                    float(np.dot(mlp, 1.0 - mid_mask)),
                    keep_mean,
                    keep_std,
                    drop_mean,
                    drop_std,
                    edit_mean,
                    edit_std,
                    safe_float(bank_meta.get("bank_rank", 0.0)),
                    safe_float(bank_meta.get("calib_global_mean_loss", 0.0)),
                    math.log1p(safe_float(bank_meta.get("calib_oracle_frequency", 0.0))),
                    safe_float(bank_meta.get("swap_distance_to_calib_global_static", 0.0)),
                ],
                dtype=np.float32,
            ),
            reason_vec,
        ],
        axis=0,
    )


def pair_feature_vector(
    prompt_vec: np.ndarray,
    cand_vec: np.ndarray,
    info: PromptInfo,
    mask: np.ndarray,
    global_static_mask: np.ndarray,
) -> np.ndarray:
    mid_mask = mask[:-1]
    delta_mid = np.abs(mid_mask - global_static_mask[:-1])
    cross = np.asarray(
        [
            float(np.dot(info.combined_scores, mid_mask)),
            float(np.dot(info.attn_scores, mid_mask)),
            float(np.dot(info.mlp_scores, mid_mask)),
            float(np.dot(info.combined_scores, delta_mid)),
            float(np.dot(info.attn_scores, delta_mid)),
            float(np.dot(info.mlp_scores, delta_mid)),
            float(np.linalg.norm(prompt_vec)),
            float(np.linalg.norm(cand_vec)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([prompt_vec, cand_vec, cross], axis=0)


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
        "prompt_depth_entropy",
        "prompt_support_size",
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
    prompt_infos: dict[int, PromptInfo],
    projection: PromptProjection,
) -> RankerDataset:
    bank_subset = bank_df[(bank_df["bank_size"] == bank_size) & (bank_df["skip_count"] == skip_count)].sort_values("bank_rank")
    bank_ids = bank_subset["mask_id"].tolist()
    bank_meta_lookup = {str(row["mask_id"]): row for _, row in bank_subset.iterrows()}
    num_blocks = len(next(iter(prompt_infos.values())).combined_scores) + 1
    global_static = str(bank_subset[bank_subset["reasons"].str.contains("calib_global_static")].iloc[0]["mask_id"])
    global_static_mask = parse_mask_id(global_static, num_blocks)

    feature_subset = feature_df[feature_df["skip_count"] == skip_count].copy()
    mask_subset = mask_df[(mask_df["skip_count"] == skip_count) & (mask_df["method"] == "exhaustive_mask")].copy()
    pivot = mask_subset.pivot(index="sequence_idx", columns="mask_id", values="continuation_loss")
    rows = []
    prompt_vectors = {}
    for _, row in feature_subset.iterrows():
        seq_id = int(row["sequence_idx"])
        info = prompt_infos[seq_id]
        prompt_vec = prompt_vector(seq_id, projection, feature_mode)
        prompt_vectors[seq_id] = prompt_vec
        global_loss = float(pivot.loc[seq_id, global_static])
        for mask_id in bank_ids:
            mask = parse_mask_id(mask_id, num_blocks)
            cand_vec = candidate_feature_vector(info, mask, global_static_mask, bank_meta_lookup[str(mask_id)])
            rows.append(
                {
                    "sequence_idx": seq_id,
                    "document_idx": int(row["document_idx"]),
                    "mask_id": mask_id,
                    "global_static_mask_id": global_static,
                    "actual_loss": float(pivot.loc[seq_id, mask_id]),
                    "delta_to_global_static": float(pivot.loc[seq_id, mask_id] - global_loss),
                    "pair_features": pair_feature_vector(prompt_vec, cand_vec, info, mask, global_static_mask),
                    "prompt_features": prompt_vec,
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


class DualTowerScorer(nn.Module):
    def __init__(self, prompt_dim: int, cand_dim: int, hidden_dim: int = 192) -> None:
        super().__init__()
        self.prompt_net = nn.Sequential(
            nn.Linear(prompt_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.cand_net = nn.Sequential(
            nn.Linear(cand_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, prompt_x: torch.Tensor, cand_x: torch.Tensor) -> torch.Tensor:
        prompt_h = self.prompt_net(prompt_x)
        cand_h = self.cand_net(cand_x)
        joint = torch.cat([prompt_h, cand_h, prompt_h * cand_h, torch.abs(prompt_h - cand_h)], dim=-1)
        return self.head(joint).squeeze(-1)


def build_pairwise_samples(y_train: np.ndarray, seq_ids: np.ndarray) -> np.ndarray:
    sampled_pairs: list[tuple[int, int]] = []
    order = np.argsort(seq_ids, kind="stable")
    sorted_seq = seq_ids[order]
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_seq[end] == sorted_seq[start]:
            end += 1
        local_idx = order[start:end]
        if len(local_idx) >= 2:
            seq_order = local_idx[np.argsort(y_train[local_idx])]
            best = int(seq_order[0])
            worst = [int(idx) for idx in seq_order[-3:] if int(idx) != best]
            static_candidates = [int(idx) for idx in local_idx if abs(y_train[int(idx)]) < 1e-9]
            for neg in worst:
                sampled_pairs.append((best, neg))
            for static_idx in static_candidates[:1]:
                if static_idx != best:
                    sampled_pairs.append((best, static_idx))
        start = end
    return np.asarray(sampled_pairs, dtype=np.int64)


def train_dual_tower(
    train_prompt: np.ndarray,
    train_cand: np.ndarray,
    train_y: np.ndarray,
    train_seq_ids: np.ndarray,
    seed: int,
    hidden_dim: int = 192,
    epochs: int = 20,
    batch_size: int = 4096,
    pair_batch_size: int = 8192,
    pair_weight: float = 0.2,
) -> tuple[DualTowerScorer, StandardScaler, StandardScaler]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    prompt_scaler = StandardScaler().fit(train_prompt)
    cand_scaler = StandardScaler().fit(train_cand)
    x_prompt = torch.tensor(prompt_scaler.transform(train_prompt), dtype=torch.float32)
    x_cand = torch.tensor(cand_scaler.transform(train_cand), dtype=torch.float32)
    y = torch.tensor(train_y, dtype=torch.float32)
    pair_idx = build_pairwise_samples(train_y, train_seq_ids)
    pair_tensor = torch.tensor(pair_idx, dtype=torch.long) if pair_idx.size > 0 else None

    device = torch.device("cpu")
    model = DualTowerScorer(prompt_dim=x_prompt.size(1), cand_dim=x_cand.size(1), hidden_dim=hidden_dim).to(device)
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
        if pair_tensor is not None and pair_tensor.numel() > 0:
            pair_perm = pair_tensor[torch.randperm(pair_tensor.size(0))]
            for start in range(0, pair_perm.size(0), pair_batch_size):
                idx = pair_perm[start : start + pair_batch_size]
                left = idx[:, 0]
                right = idx[:, 1]
                left_score = model(x_prompt[left], x_cand[left])
                right_score = model(x_prompt[right], x_cand[right])
                pair_loss = torch.nn.functional.softplus(left_score - right_score).mean()
                optimizer.zero_grad()
                (pair_weight * pair_loss).backward()
                optimizer.step()
    return model.eval(), prompt_scaler, cand_scaler


def predict_dual_tower(
    model: DualTowerScorer,
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


def knn_prompt_predict(
    train_prompt: np.ndarray,
    train_mask_ids: np.ndarray,
    train_y: np.ndarray,
    eval_prompt: np.ndarray,
    eval_mask_ids: np.ndarray,
    k: int = 32,
) -> np.ndarray:
    scaler = StandardScaler().fit(train_prompt)
    train_z = scaler.transform(train_prompt)
    eval_z = scaler.transform(eval_prompt)
    nbrs = NearestNeighbors(n_neighbors=min(k, train_z.shape[0]), metric="euclidean").fit(train_z)
    _distances, indices = nbrs.kneighbors(eval_z)
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


def knn_pair_predict(train_pair: np.ndarray, train_y: np.ndarray, eval_pair: np.ndarray, k: int = 32) -> np.ndarray:
    scaler = StandardScaler().fit(train_pair)
    train_z = scaler.transform(train_pair)
    eval_z = scaler.transform(eval_pair)
    nbrs = NearestNeighbors(n_neighbors=min(k, train_z.shape[0]), metric="euclidean").fit(train_z)
    _distances, indices = nbrs.kneighbors(eval_z)
    return np.asarray([float(train_y[idx].mean()) for idx in indices], dtype=np.float64)


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


def calibration_rows(selected_df: pd.DataFrame, output_tag: str, feature_mode: str, skip_count: int, bank_size: int, model_name: str) -> list[dict[str, Any]]:
    predicted = selected_df["predicted_delta"].to_numpy(dtype=np.float64)
    actual = selected_df["actual_delta_to_static"].to_numpy(dtype=np.float64)
    rank_corr = float(pd.Series(predicted).corr(pd.Series(actual), method="spearman")) if len(selected_df) > 1 else float("nan")
    negative_mask = predicted < 0.0
    negative_precision = float((actual[negative_mask] < 0.0).mean()) if negative_mask.any() else float("nan")
    bins = pd.qcut(pd.Series(predicted), q=min(5, len(selected_df)), duplicates="drop")
    rows: list[dict[str, Any]] = [
        {
            "output_tag": output_tag,
            "feature_mode": feature_mode,
            "skip_count": skip_count,
            "bank_size": bank_size,
            "model_name": model_name,
            "metric": "predicted_vs_actual_spearman",
            "value": rank_corr,
        },
        {
            "output_tag": output_tag,
            "feature_mode": feature_mode,
            "skip_count": skip_count,
            "bank_size": bank_size,
            "model_name": model_name,
            "metric": "predicted_negative_precision",
            "value": negative_precision,
        },
    ]
    for bin_id, (_, group) in enumerate(selected_df.groupby(bins, sort=False, observed=False), start=1):
        rows.append(
            {
                "output_tag": output_tag,
                "feature_mode": feature_mode,
                "skip_count": skip_count,
                "bank_size": bank_size,
                "model_name": model_name,
                "metric": f"bin{bin_id}_mean_actual_delta",
                "value": float(group["actual_delta_to_static"].mean()),
            }
        )
        rows.append(
            {
                "output_tag": output_tag,
                "feature_mode": feature_mode,
                "skip_count": skip_count,
                "bank_size": bank_size,
                "model_name": model_name,
                "metric": f"bin{bin_id}_mean_predicted_delta",
                "value": float(group["predicted_delta"].mean()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-tag", type=str, required=True)
    parser.add_argument("--bank-size", type=int, required=True)
    parser.add_argument("--train-tags", type=str, nargs="+", required=True)
    parser.add_argument("--eval-tags", type=str, nargs="+", required=True)
    parser.add_argument("--output-tag", type=str, required=True)
    parser.add_argument(
        "--feature-mode",
        choices=["attnres", "hidden", "difficulty", "attnres_hidden", "full"],
        default="attnres",
    )
    parser.add_argument("--hidden-train-tags", type=str, nargs="*", default=None)
    parser.add_argument("--hidden-eval-tags", type=str, nargs="*", default=None)
    parser.add_argument("--train-doc-frac", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-pca-dim", type=int, default=64)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    bank_dir = ROOT / "results" / "bank_hygiene"
    oracle_dir = ROOT / "results" / "oracles"
    rich_dir = ROOT / "results" / "rich_features"
    out_dir = ROOT / "results" / "hetero_scorer_v6"
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
    train_prompt_infos = build_prompt_info_table(train_feature_table, None)
    eval_prompt_infos = build_prompt_info_table(eval_feature_table, None)
    projection = fit_prompt_projection(train_prompt_infos, eval_prompt_infos, hidden_components=args.hidden_pca_dim)

    eval_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    calibration_summary_rows: list[dict[str, Any]] = []

    for skip_count in sorted(bank_df["skip_count"].unique().tolist()):
        train_ds = build_ranker_dataset(
            train_feature_table,
            train_masks,
            bank_df,
            args.bank_size,
            skip_count,
            args.feature_mode,
            train_prompt_infos,
            projection,
        )
        eval_ds = build_ranker_dataset(
            eval_feature_table,
            eval_masks,
            bank_df,
            args.bank_size,
            skip_count,
            args.feature_mode,
            eval_prompt_infos,
            projection,
        )

        train_docs = np.asarray(sorted(train_ds.rows["document_idx"].unique().tolist()), dtype=int)
        rng.shuffle(train_docs)
        split_at = max(1, int(round(len(train_docs) * args.train_doc_frac)))
        fit_docs = set(train_docs[:split_at].tolist())
        dev_docs = set(train_docs[split_at:].tolist()) or fit_docs

        train_rows = train_ds.rows[train_ds.rows["document_idx"].isin(fit_docs)].copy().reset_index(drop=True)
        dev_rows = train_ds.rows[train_ds.rows["document_idx"].isin(dev_docs)].copy().reset_index(drop=True)
        eval_rows_df = eval_ds.rows.copy().reset_index(drop=True)

        x_train_pair = np.stack(train_rows["pair_features"].to_list(), axis=0)
        x_dev_pair = np.stack(dev_rows["pair_features"].to_list(), axis=0)
        x_eval_pair = np.stack(eval_rows_df["pair_features"].to_list(), axis=0)
        x_train_prompt = np.stack(train_rows["prompt_features"].to_list(), axis=0)
        x_dev_prompt = np.stack(dev_rows["prompt_features"].to_list(), axis=0)
        x_eval_prompt = np.stack(eval_rows_df["prompt_features"].to_list(), axis=0)
        x_train_cand = np.stack(train_rows["candidate_features"].to_list(), axis=0)
        x_dev_cand = np.stack(dev_rows["candidate_features"].to_list(), axis=0)
        x_eval_cand = np.stack(eval_rows_df["candidate_features"].to_list(), axis=0)
        y_train = train_rows["delta_to_global_static"].to_numpy(dtype=np.float64)
        improve_y_train = (y_train < 0.0).astype(np.int64)
        train_seq_ids = train_rows["sequence_idx"].to_numpy(dtype=np.int64)

        models_eval: dict[str, np.ndarray] = {}
        models_dev: dict[str, np.ndarray] = {}

        hgb = HistGradientBoostingRegressor(max_depth=8, max_iter=400, learning_rate=0.05, random_state=args.seed)
        hgb.fit(x_train_pair, y_train)
        models_eval["hgb_pair"] = hgb.predict(x_eval_pair)
        models_dev["hgb_pair"] = hgb.predict(x_dev_pair)

        rf = RandomForestRegressor(
            n_estimators=400,
            max_depth=18,
            min_samples_leaf=2,
            random_state=args.seed,
            n_jobs=-1,
        )
        rf.fit(x_train_pair, y_train)
        models_eval["rf_pair"] = rf.predict(x_eval_pair)
        models_dev["rf_pair"] = rf.predict(x_dev_pair)

        et = ExtraTreesRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            random_state=args.seed,
            n_jobs=-1,
        )
        et.fit(x_train_pair, y_train)
        models_eval["extra_pair"] = et.predict(x_eval_pair)
        models_dev["extra_pair"] = et.predict(x_dev_pair)

        models_eval["knn_prompt_v2"] = knn_prompt_predict(
            x_train_prompt,
            train_rows["mask_id"].to_numpy(dtype=object),
            y_train,
            x_eval_prompt,
            eval_rows_df["mask_id"].to_numpy(dtype=object),
        )
        models_dev["knn_prompt_v2"] = knn_prompt_predict(
            x_train_prompt,
            train_rows["mask_id"].to_numpy(dtype=object),
            y_train,
            x_dev_prompt,
            dev_rows["mask_id"].to_numpy(dtype=object),
        )

        models_eval["knn_pair_v2"] = knn_pair_predict(x_train_pair, y_train, x_eval_pair)
        models_dev["knn_pair_v2"] = knn_pair_predict(x_train_pair, y_train, x_dev_pair)

        hgb_clf = HistGradientBoostingClassifier(max_depth=6, max_iter=300, learning_rate=0.05, random_state=args.seed)
        hgb_clf.fit(x_train_pair, improve_y_train)
        eval_prob = hgb_clf.predict_proba(x_eval_pair)[:, 1]
        dev_prob = hgb_clf.predict_proba(x_dev_pair)[:, 1]
        fallback = np.maximum(models_eval["hgb_pair"], 0.0)
        fallback_dev = np.maximum(models_dev["hgb_pair"], 0.0)
        models_eval["hgb_delta_cls"] = fallback - eval_prob
        models_dev["hgb_delta_cls"] = fallback_dev - dev_prob

        dual_model, prompt_scaler, cand_scaler = train_dual_tower(
            x_train_prompt,
            x_train_cand,
            y_train,
            train_seq_ids,
            seed=args.seed,
        )
        models_eval["dual_tower_rank"] = predict_dual_tower(
            dual_model, prompt_scaler, cand_scaler, x_eval_prompt, x_eval_cand
        )
        models_dev["dual_tower_rank"] = predict_dual_tower(
            dual_model, prompt_scaler, cand_scaler, x_dev_prompt, x_dev_cand
        )

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
            eval_rows_scored = eval_rows_df.copy()
            eval_rows_scored[model_name] = preds
            per_sequence_df = evaluate_selected_masks(
                rows_df=eval_rows_scored,
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
            calibration_summary_rows.extend(
                calibration_rows(
                    per_sequence_df,
                    output_tag=args.output_tag,
                    feature_mode=args.feature_mode,
                    skip_count=skip_count,
                    bank_size=args.bank_size,
                    model_name=model_name,
                )
            )
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
    calibration_df = pd.DataFrame(calibration_summary_rows)

    eval_df.to_csv(out_dir / f"{args.output_tag}_{args.feature_mode}_per_sequence.csv", index=False)
    summary_df.to_csv(out_dir / f"{args.output_tag}_{args.feature_mode}_summary.csv", index=False)
    selection_df.to_csv(out_dir / f"{args.output_tag}_{args.feature_mode}_model_selection.csv", index=False)
    calibration_df.to_csv(out_dir / f"{args.output_tag}_{args.feature_mode}_calibration.csv", index=False)

    if not summary_df.empty:
        plt.figure(figsize=(8, 4))
        for skip_count in sorted(summary_df["skip_count"].unique().tolist()):
            subset = summary_df[
                (summary_df["skip_count"] == skip_count) & (summary_df["metric"] == "delta_to_static")
            ].sort_values("mean")
            plt.plot(subset["model_name"], subset["mean"], marker="o", label=f"skip={skip_count}")
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("delta to global static")
        plt.tight_layout()
        plt.legend()
        plt.savefig(plot_dir / f"hetero_scorer_v6_{args.output_tag}_{args.feature_mode}_delta_to_static.png", dpi=160)
        plt.close()


if __name__ == "__main__":
    main()
