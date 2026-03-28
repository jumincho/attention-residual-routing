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
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attnres_routing.analysis import bootstrap_mean_ci  # noqa: E402


ATTNRES_MODES = {
    "attnres",
    "attnres_hidden",
    "full",
    "attnres_stp_scalar",
    "attnres_stp_diff",
}
HIDDEN_MODES = {
    "hidden",
    "attnres_hidden",
    "full",
    "hidden_stp_diff",
}
DIFFICULTY_MODES = {
    "difficulty",
    "full",
}
STP_SCALAR_MODES = {
    "stp_scalar",
    "attnres_stp_scalar",
}
STP_DIFF_MODES = {
    "attnres_stp_diff",
    "hidden_stp_diff",
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


def safe_ratio(numerator: float, denominator: float, eps: float = 1e-8) -> float:
    return float(numerator / max(abs(denominator), eps))


def safe_cosine(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(left, right) / denom)


def parse_hidden_row(row: pd.Series) -> np.ndarray:
    parts = []
    for column in sorted(row.index.tolist()):
        if column.endswith("_mean_json") or column.endswith("_final_json"):
            parts.append(parse_json_vec(row[column]))
        elif column.endswith("_norm_mean") or column.endswith("_norm_std"):
            parts.append(np.asarray([safe_float(row[column])], dtype=np.float32))
    return np.concatenate(parts, axis=0) if parts else np.zeros(0, dtype=np.float32)


def ordered_hidden_prefixes(row: pd.Series) -> list[str]:
    prefixes = {column[: -len("_mean_json")] for column in row.index.tolist() if column.endswith("_mean_json")}
    ordered = []
    block_items: list[tuple[int, str]] = []
    for prefix in prefixes:
        if prefix.startswith("block_"):
            try:
                block_items.append((int(prefix.split("_", 1)[1]), prefix))
            except Exception:
                continue
    ordered.extend(prefix for _idx, prefix in sorted(block_items))
    if "final" in prefixes:
        ordered.append("final")
    return ordered


def hidden_summary_triplet(row: pd.Series) -> tuple[str, str, str] | None:
    prefixes = ordered_hidden_prefixes(row)
    if len(prefixes) < 3:
        return None
    return prefixes[0], prefixes[len(prefixes) // 2], prefixes[-1]


def hidden_vector(row: pd.Series, prefix: str, suffix: str) -> np.ndarray:
    return parse_json_vec(row.get(f"{prefix}_{suffix}"))


def build_stp_scalar_row(row: pd.Series) -> np.ndarray:
    triplet = hidden_summary_triplet(row)
    if triplet is None:
        return np.zeros(0, dtype=np.float32)
    start_prefix, mid_prefix, end_prefix = triplet

    start_mean = hidden_vector(row, start_prefix, "mean_json")
    mid_mean = hidden_vector(row, mid_prefix, "mean_json")
    end_mean = hidden_vector(row, end_prefix, "mean_json")
    start_final = hidden_vector(row, start_prefix, "final_json")
    mid_final = hidden_vector(row, mid_prefix, "final_json")
    end_final = hidden_vector(row, end_prefix, "final_json")

    d_sm_mean = mid_mean - start_mean
    d_me_mean = end_mean - mid_mean
    d_se_mean = end_mean - start_mean
    d_sm_final = mid_final - start_final
    d_me_final = end_final - mid_final
    d_se_final = end_final - start_final

    norm_sm_mean = float(np.linalg.norm(d_sm_mean))
    norm_me_mean = float(np.linalg.norm(d_me_mean))
    norm_se_mean = float(np.linalg.norm(d_se_mean))
    norm_sm_final = float(np.linalg.norm(d_sm_final))
    norm_me_final = float(np.linalg.norm(d_me_final))
    norm_se_final = float(np.linalg.norm(d_se_final))

    return np.asarray(
        [
            1.0 - safe_cosine(d_me_mean, d_sm_mean),
            1.0 - safe_cosine(d_me_final, d_sm_final),
            safe_cosine(d_me_mean, d_sm_mean),
            safe_cosine(d_me_final, d_sm_final),
            norm_sm_mean,
            norm_me_mean,
            norm_se_mean,
            safe_ratio(norm_me_mean, norm_sm_mean),
            norm_sm_final,
            norm_me_final,
            norm_se_final,
            safe_ratio(norm_me_final, norm_sm_final),
            safe_cosine(d_se_mean, d_sm_mean),
            safe_cosine(d_se_mean, d_me_mean),
            safe_cosine(d_se_final, d_sm_final),
            safe_cosine(d_se_final, d_me_final),
            safe_float(row.get(f"{start_prefix}_norm_mean", 0.0)),
            safe_float(row.get(f"{mid_prefix}_norm_mean", 0.0)),
            safe_float(row.get(f"{end_prefix}_norm_mean", 0.0)),
            safe_float(row.get(f"{start_prefix}_norm_std", 0.0)),
            safe_float(row.get(f"{mid_prefix}_norm_std", 0.0)),
            safe_float(row.get(f"{end_prefix}_norm_std", 0.0)),
        ],
        dtype=np.float32,
    )


def build_stp_diff_row(row: pd.Series) -> np.ndarray:
    triplet = hidden_summary_triplet(row)
    if triplet is None:
        return np.zeros(0, dtype=np.float32)
    start_prefix, mid_prefix, end_prefix = triplet

    start_mean = hidden_vector(row, start_prefix, "mean_json")
    mid_mean = hidden_vector(row, mid_prefix, "mean_json")
    end_mean = hidden_vector(row, end_prefix, "mean_json")
    start_final = hidden_vector(row, start_prefix, "final_json")
    mid_final = hidden_vector(row, mid_prefix, "final_json")
    end_final = hidden_vector(row, end_prefix, "final_json")

    return np.concatenate(
        [
            mid_mean - start_mean,
            end_mean - mid_mean,
            end_mean - start_mean,
            mid_final - start_final,
            end_final - mid_final,
            end_final - start_final,
        ],
        axis=0,
    ).astype(np.float32)


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
    stp_scalar_raw: np.ndarray
    stp_diff_raw: np.ndarray
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
            stp_scalar_raw=build_stp_scalar_row(row),
            stp_diff_raw=build_stp_diff_row(row),
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
    stp_scalar: dict[int, np.ndarray]
    stp_diff: dict[int, np.ndarray]
    difficulty: dict[int, np.ndarray]


def fit_prompt_projection(
    train_infos: dict[int, PromptInfo],
    eval_infos: dict[int, PromptInfo],
    hidden_components: int,
    stp_diff_components: int,
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

    train_stp_diff = [info.stp_diff_raw for info in train_infos.values() if info.stp_diff_raw.size > 0]
    stp_diff_dim = train_stp_diff[0].shape[0] if train_stp_diff else 0
    use_stp_pca = stp_diff_dim > 0 and stp_diff_components > 0 and len(train_stp_diff) > 4
    stp_scaler = None
    stp_pca = None
    if use_stp_pca:
        n_components = min(stp_diff_components, stp_diff_dim, len(train_stp_diff) - 1)
        if n_components > 0:
            stp_scaler = StandardScaler().fit(np.stack(train_stp_diff, axis=0))
            stp_train_scaled = stp_scaler.transform(np.stack(train_stp_diff, axis=0))
            stp_pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
            stp_pca.fit(stp_train_scaled)

    def project_hidden(info: PromptInfo) -> np.ndarray:
        raw = info.hidden_raw
        if raw.size == 0:
            return np.zeros(0, dtype=np.float32)
        if pca_model is None or hidden_scaler is None:
            return raw.astype(np.float32)
        return pca_model.transform(hidden_scaler.transform(raw.reshape(1, -1))).reshape(-1).astype(np.float32)

    def project_stp_diff(info: PromptInfo) -> np.ndarray:
        raw = info.stp_diff_raw
        if raw.size == 0:
            return np.zeros(0, dtype=np.float32)
        if stp_pca is None or stp_scaler is None:
            return raw.astype(np.float32)
        return stp_pca.transform(stp_scaler.transform(raw.reshape(1, -1))).reshape(-1).astype(np.float32)

    return PromptProjection(
        attnres={seq_id: info.attnres_raw for seq_id, info in {**train_infos, **eval_infos}.items()},
        hidden={seq_id: project_hidden(info) for seq_id, info in {**train_infos, **eval_infos}.items()},
        stp_scalar={seq_id: info.stp_scalar_raw for seq_id, info in {**train_infos, **eval_infos}.items()},
        stp_diff={seq_id: project_stp_diff(info) for seq_id, info in {**train_infos, **eval_infos}.items()},
        difficulty={seq_id: info.difficulty for seq_id, info in {**train_infos, **eval_infos}.items()},
    )


def prompt_vector(seq_id: int, projection: PromptProjection, feature_mode: str) -> np.ndarray:
    parts: list[np.ndarray] = []
    if feature_mode in ATTNRES_MODES:
        parts.append(projection.attnres[seq_id])
    if feature_mode in HIDDEN_MODES:
        parts.append(projection.hidden[seq_id])
    if feature_mode in STP_SCALAR_MODES:
        parts.append(projection.stp_scalar[seq_id])
    if feature_mode in STP_DIFF_MODES:
        parts.append(projection.stp_diff[seq_id])
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


def compute_sample_weights(y_train: np.ndarray, seq_ids: np.ndarray) -> np.ndarray:
    weights = np.ones_like(y_train, dtype=np.float64)
    unique_seq = np.unique(seq_ids)
    for seq_id in unique_seq:
        idx = np.flatnonzero(seq_ids == seq_id)
        seq_y = y_train[idx]
        best = float(seq_y.min())
        regret = seq_y - best
        seq_weights = 1.0 + 4.0 * np.clip(-seq_y, 0.0, None) + np.exp(-regret / 0.05)
        weights[idx] = seq_weights
    weights = weights / max(weights.mean(), 1e-8)
    return weights


def train_dual_tower(
    train_prompt: np.ndarray,
    train_cand: np.ndarray,
    train_y: np.ndarray,
    train_seq_ids: np.ndarray,
    seed: int,
    hidden_dim: int = 160,
    epochs: int = 10,
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


def train_dual_tower_listwise(
    train_prompt: np.ndarray,
    train_cand: np.ndarray,
    train_y: np.ndarray,
    train_seq_ids: np.ndarray,
    seed: int,
    hidden_dim: int = 160,
    epochs: int = 8,
    batch_groups: int = 256,
    target_temp: float = 0.05,
) -> tuple[DualTowerScorer, StandardScaler, StandardScaler]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    prompt_scaler = StandardScaler().fit(train_prompt)
    cand_scaler = StandardScaler().fit(train_cand)
    x_prompt = torch.tensor(prompt_scaler.transform(train_prompt), dtype=torch.float32)
    x_cand = torch.tensor(cand_scaler.transform(train_cand), dtype=torch.float32)
    y = torch.tensor(train_y, dtype=torch.float32)

    seq_to_indices: list[np.ndarray] = []
    for seq_id in np.unique(train_seq_ids):
        idx = np.flatnonzero(train_seq_ids == seq_id)
        if idx.size >= 2:
            seq_to_indices.append(idx)

    device = torch.device("cpu")
    model = DualTowerScorer(prompt_dim=x_prompt.size(1), cand_dim=x_cand.size(1), hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for _epoch in range(epochs):
        np.random.shuffle(seq_to_indices)
        for start in range(0, len(seq_to_indices), batch_groups):
            groups = seq_to_indices[start : start + batch_groups]
            if not groups:
                continue
            optimizer.zero_grad()
            total_loss = torch.zeros((), dtype=torch.float32)
            for idx in groups:
                idx_t = torch.tensor(idx, dtype=torch.long)
                pred = model(x_prompt[idx_t], x_cand[idx_t])
                target_logits = (-y[idx_t] / target_temp).detach()
                target_probs = torch.softmax(target_logits, dim=0)
                list_loss = -(target_probs * torch.log_softmax(pred, dim=0)).sum()
                best_idx = int(torch.argmin(y[idx_t]).item())
                cls_loss = torch.nn.functional.cross_entropy(pred.unsqueeze(0), torch.tensor([best_idx]))
                total_loss = total_loss + list_loss + (0.5 * cls_loss)
            total_loss = total_loss / max(len(groups), 1)
            total_loss.backward()
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


def top_alternative_masks(rows_df: pd.DataFrame, global_static: str, top_k: int) -> list[str]:
    best_mask_df = (
        rows_df.sort_values(["sequence_idx", "actual_loss"])
        .drop_duplicates(subset=["sequence_idx"])
        .copy()
    )
    alt_counts = (
        best_mask_df[best_mask_df["mask_id"] != global_static]["mask_id"]
        .value_counts()
        .sort_values(ascending=False)
    )
    masks = [str(mask_id) for mask_id in alt_counts.index.tolist()[:top_k]]
    if not masks:
        fallback = (
            rows_df[rows_df["mask_id"] != global_static]["mask_id"]
            .value_counts()
            .sort_values(ascending=False)
            .index.tolist()
        )
        masks = [str(mask_id) for mask_id in fallback[:top_k]]
    return masks


def fit_prompt_template_gate(
    rows_df: pd.DataFrame,
    global_static: str,
    alt_masks: list[str],
    seed: int,
) -> tuple[HistGradientBoostingClassifier, list[str], np.ndarray]:
    classes = [global_static] + [mask for mask in alt_masks if mask != global_static]
    subset = rows_df[rows_df["mask_id"].isin(classes)].copy()
    if subset.empty:
        raise ValueError("Template-gate subset is empty.")
    prompt_lookup = (
        subset.groupby("sequence_idx", sort=False)["prompt_features"]
        .first()
        .to_dict()
    )
    pivot = subset.pivot(index="sequence_idx", columns="mask_id", values="actual_loss")
    usable_classes = [mask for mask in classes if mask in pivot.columns]
    if global_static not in usable_classes:
        raise ValueError("Global static missing from template-gate pivot.")
    best_class_idx = np.argmin(pivot[usable_classes].to_numpy(dtype=np.float64), axis=1)
    x_train = np.stack([prompt_lookup[int(seq_id)] for seq_id in pivot.index.tolist()], axis=0)
    y_train = best_class_idx.astype(np.int64)
    clf = HistGradientBoostingClassifier(
        max_depth=6,
        max_iter=300,
        learning_rate=0.05,
        random_state=seed,
    )
    clf.fit(x_train, y_train)
    return clf, usable_classes, x_train


def gate_scores_from_classifier(
    rows_df: pd.DataFrame,
    classifier: HistGradientBoostingClassifier,
    classes: list[str],
) -> np.ndarray:
    class_to_index = {mask_id: idx for idx, mask_id in enumerate(classes)}
    prompt_lookup = (
        rows_df.groupby("sequence_idx", sort=False)["prompt_features"]
        .first()
        .to_dict()
    )
    seq_ids = list(prompt_lookup.keys())
    prompt_matrix = np.stack([prompt_lookup[seq_id] for seq_id in seq_ids], axis=0)
    probs = classifier.predict_proba(prompt_matrix)
    seq_prob = {int(seq_id): probs[idx] for idx, seq_id in enumerate(seq_ids)}
    scores = np.full(len(rows_df), 1e6, dtype=np.float64)
    for row_idx, row in enumerate(rows_df.itertuples(index=False)):
        seq_id = int(row.sequence_idx)
        mask_id = str(row.mask_id)
        if mask_id in class_to_index:
            scores[row_idx] = float(-seq_prob[seq_id][class_to_index[mask_id]])
    return scores


def retrieval_rerank_scores(
    rows_df: pd.DataFrame,
    retrieval_scores: np.ndarray,
    rerank_scores: np.ndarray,
    top_k: int,
) -> np.ndarray:
    output = np.full(len(rows_df), 1e6, dtype=np.float64)
    for _, group in rows_df.assign(_retrieval=retrieval_scores, _rerank=rerank_scores).groupby("sequence_idx", sort=False):
        chosen_idx = group.nsmallest(top_k, columns="_retrieval").index.to_numpy(dtype=np.int64)
        output[chosen_idx] = rerank_scores[chosen_idx]
    return output


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
        choices=[
            "attnres",
            "hidden",
            "difficulty",
            "attnres_hidden",
            "full",
            "stp_scalar",
            "attnres_stp_scalar",
            "attnres_stp_diff",
            "hidden_stp_diff",
        ],
        default="attnres",
    )
    parser.add_argument("--hidden-train-tags", type=str, nargs="*", default=None)
    parser.add_argument("--hidden-eval-tags", type=str, nargs="*", default=None)
    parser.add_argument("--train-doc-frac", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-pca-dim", type=int, default=64)
    parser.add_argument("--stp-pca-dim", type=int, default=64)
    parser.add_argument("--skip-counts", type=int, nargs="*", default=None)
    parser.add_argument("--fast-mode", action="store_true")
    parser.add_argument("--output-subdir", type=str, default="regret_reduction_v7")
    parser.add_argument("--plot-prefix", type=str, default="regret_reduction_v7")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    bank_dir = ROOT / "results" / "bank_hygiene"
    oracle_dir = ROOT / "results" / "oracles"
    rich_dir = ROOT / "results" / "rich_features"
    out_dir = ROOT / "results" / args.output_subdir
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
    projection = fit_prompt_projection(
        train_prompt_infos,
        eval_prompt_infos,
        hidden_components=args.hidden_pca_dim,
        stp_diff_components=args.stp_pca_dim,
    )

    eval_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    calibration_summary_rows: list[dict[str, Any]] = []

    requested_skip_counts = sorted(bank_df["skip_count"].unique().tolist())
    if args.skip_counts:
        requested = set(int(v) for v in args.skip_counts)
        requested_skip_counts = [v for v in requested_skip_counts if v in requested]
        if not requested_skip_counts:
            raise ValueError(f"No requested skip counts {sorted(requested)} present in bank data")

    for skip_count in requested_skip_counts:
        print(
            f"[ranker-v7] output={args.output_tag} feature_mode={args.feature_mode} "
            f"skip={skip_count} bank={args.bank_size} fast_mode={int(args.fast_mode)}",
            flush=True,
        )
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

        sample_weights = compute_sample_weights(y_train, train_seq_ids)
        models_eval: dict[str, np.ndarray] = {}
        models_dev: dict[str, np.ndarray] = {}

        print("[ranker-v7] fit hgb_pair", flush=True)
        hgb = HistGradientBoostingRegressor(max_depth=8, max_iter=400, learning_rate=0.05, random_state=args.seed)
        hgb.fit(x_train_pair, y_train)
        models_eval["hgb_pair"] = hgb.predict(x_eval_pair)
        models_dev["hgb_pair"] = hgb.predict(x_dev_pair)

        print("[ranker-v7] fit hgb_pair_weighted", flush=True)
        hgb_weighted = HistGradientBoostingRegressor(max_depth=8, max_iter=500, learning_rate=0.04, random_state=args.seed)
        hgb_weighted.fit(x_train_pair, y_train, sample_weight=sample_weights)
        models_eval["hgb_pair_weighted"] = hgb_weighted.predict(x_eval_pair)
        models_dev["hgb_pair_weighted"] = hgb_weighted.predict(x_dev_pair)

        if not args.fast_mode:
            print("[ranker-v7] fit rf_pair / rf_pair_weighted", flush=True)
            rf = RandomForestRegressor(
                n_estimators=200,
                max_depth=18,
                min_samples_leaf=2,
                random_state=args.seed,
                n_jobs=-1,
            )
            rf.fit(x_train_pair, y_train)
            models_eval["rf_pair"] = rf.predict(x_eval_pair)
            models_dev["rf_pair"] = rf.predict(x_dev_pair)

            rf_weighted = RandomForestRegressor(
                n_estimators=240,
                max_depth=20,
                min_samples_leaf=2,
                random_state=args.seed,
                n_jobs=-1,
            )
            rf_weighted.fit(x_train_pair, y_train, sample_weight=sample_weights)
            models_eval["rf_pair_weighted"] = rf_weighted.predict(x_eval_pair)
            models_dev["rf_pair_weighted"] = rf_weighted.predict(x_dev_pair)

        if not args.fast_mode:
            print("[ranker-v7] fit extra_pair / extra_pair_weighted", flush=True)
            et = ExtraTreesRegressor(
                n_estimators=200,
                max_depth=None,
                min_samples_leaf=2,
                random_state=args.seed,
                n_jobs=-1,
            )
            et.fit(x_train_pair, y_train)
            models_eval["extra_pair"] = et.predict(x_eval_pair)
            models_dev["extra_pair"] = et.predict(x_dev_pair)

            et_weighted = ExtraTreesRegressor(
                n_estimators=240,
                max_depth=None,
                min_samples_leaf=2,
                random_state=args.seed,
                n_jobs=-1,
            )
            et_weighted.fit(x_train_pair, y_train, sample_weight=sample_weights)
            models_eval["extra_pair_weighted"] = et_weighted.predict(x_eval_pair)
            models_dev["extra_pair_weighted"] = et_weighted.predict(x_dev_pair)

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

        models_eval["ensemble_hgb_knn"] = 0.65 * models_eval["hgb_pair_weighted"] + 0.35 * models_eval["knn_pair_v2"]
        models_dev["ensemble_hgb_knn"] = 0.65 * models_dev["hgb_pair_weighted"] + 0.35 * models_dev["knn_pair_v2"]

        retrieval_top2_eval = retrieval_rerank_scores(
            eval_rows_df,
            models_eval["knn_prompt_v2"],
            models_eval["hgb_pair_weighted"],
            top_k=2,
        )
        retrieval_top2_dev = retrieval_rerank_scores(
            dev_rows,
            models_dev["knn_prompt_v2"],
            models_dev["hgb_pair_weighted"],
            top_k=2,
        )
        models_eval["retrieval_rerank_top2"] = retrieval_top2_eval
        models_dev["retrieval_rerank_top2"] = retrieval_top2_dev

        retrieval_top4_eval = retrieval_rerank_scores(
            eval_rows_df,
            models_eval["knn_prompt_v2"],
            models_eval["hgb_pair_weighted"],
            top_k=4,
        )
        retrieval_top4_dev = retrieval_rerank_scores(
            dev_rows,
            models_dev["knn_prompt_v2"],
            models_dev["hgb_pair_weighted"],
            top_k=4,
        )
        models_eval["retrieval_rerank_top4"] = retrieval_top4_eval
        models_dev["retrieval_rerank_top4"] = retrieval_top4_dev

        top1_alt = top_alternative_masks(train_rows, train_ds.global_static, top_k=1)
        if top1_alt:
            gate_binary, gate_binary_classes, _ = fit_prompt_template_gate(
                train_rows,
                global_static=train_ds.global_static,
                alt_masks=top1_alt,
                seed=args.seed,
            )
            models_eval["binary_gate_top1"] = gate_scores_from_classifier(
                eval_rows_df,
                classifier=gate_binary,
                classes=gate_binary_classes,
            )
            models_dev["binary_gate_top1"] = gate_scores_from_classifier(
                dev_rows,
                classifier=gate_binary,
                classes=gate_binary_classes,
            )

        top2_alt = top_alternative_masks(train_rows, train_ds.global_static, top_k=2)
        if top2_alt:
            gate_ternary, gate_ternary_classes, _ = fit_prompt_template_gate(
                train_rows,
                global_static=train_ds.global_static,
                alt_masks=top2_alt,
                seed=args.seed,
            )
            models_eval["ternary_gate_top2"] = gate_scores_from_classifier(
                eval_rows_df,
                classifier=gate_ternary,
                classes=gate_ternary_classes,
            )
            models_dev["ternary_gate_top2"] = gate_scores_from_classifier(
                dev_rows,
                classifier=gate_ternary,
                classes=gate_ternary_classes,
            )

        print("[ranker-v7] fit hgb_delta_cls / knn ensemble", flush=True)
        hgb_clf = HistGradientBoostingClassifier(max_depth=6, max_iter=300, learning_rate=0.05, random_state=args.seed)
        hgb_clf.fit(x_train_pair, improve_y_train, sample_weight=sample_weights)
        eval_prob = hgb_clf.predict_proba(x_eval_pair)[:, 1]
        dev_prob = hgb_clf.predict_proba(x_dev_pair)[:, 1]
        fallback = np.maximum(models_eval["hgb_pair_weighted"], 0.0)
        fallback_dev = np.maximum(models_dev["hgb_pair_weighted"], 0.0)
        models_eval["hgb_delta_cls"] = fallback - eval_prob
        models_dev["hgb_delta_cls"] = fallback_dev - dev_prob

        if not args.fast_mode:
            print("[ranker-v7] fit dual_tower_rank / dual_tower_listwise", flush=True)
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

            listwise_model, listwise_prompt_scaler, listwise_cand_scaler = train_dual_tower_listwise(
                x_train_prompt,
                x_train_cand,
                y_train,
                train_seq_ids,
                seed=args.seed,
            )
            models_eval["dual_tower_listwise"] = predict_dual_tower(
                listwise_model,
                listwise_prompt_scaler,
                listwise_cand_scaler,
                x_eval_prompt,
                x_eval_cand,
            )
            models_dev["dual_tower_listwise"] = predict_dual_tower(
                listwise_model,
                listwise_prompt_scaler,
                listwise_cand_scaler,
                x_dev_prompt,
                x_dev_cand,
            )

            models_eval["ensemble_tree_tower"] = (
                0.65 * models_eval["hgb_pair_weighted"]
                + 0.15 * models_eval["rf_pair_weighted"]
                + 0.20 * models_eval["dual_tower_listwise"]
            )
            models_dev["ensemble_tree_tower"] = (
                0.65 * models_dev["hgb_pair_weighted"]
                + 0.15 * models_dev["rf_pair_weighted"]
                + 0.20 * models_dev["dual_tower_listwise"]
            )

        bank_upper = (
            eval_rows_df.groupby("sequence_idx", as_index=False)["actual_loss"]
            .min()
            .rename(columns={"actual_loss": "bank_upper_bound_loss"})
            .set_index("sequence_idx")
        )
        eval_bank_best_mask = (
            eval_rows_df.sort_values(["sequence_idx", "actual_loss"])
            .drop_duplicates(subset=["sequence_idx"])
            .set_index("sequence_idx")["mask_id"]
        )
        bank_upper["bank_best_mask_id"] = eval_bank_best_mask

        dev_bank_upper = (
            dev_rows.groupby("sequence_idx", as_index=False)["actual_loss"]
            .min()
            .rename(columns={"actual_loss": "bank_upper_bound_loss"})
            .set_index("sequence_idx")
        )
        dev_bank_best_mask = (
            dev_rows.sort_values(["sequence_idx", "actual_loss"])
            .drop_duplicates(subset=["sequence_idx"])
            .set_index("sequence_idx")["mask_id"]
        )
        dev_bank_upper["bank_best_mask_id"] = dev_bank_best_mask

        for model_name, preds in models_dev.items():
            dev_rows_scored = dev_rows.copy()
            dev_rows_scored[model_name] = preds
            dev_selected = evaluate_selected_masks(
                rows_df=dev_rows_scored,
                score_col=model_name,
                global_static=train_ds.global_static,
                bank_upper_df=dev_bank_upper,
                output_tag=args.output_tag,
                feature_mode=args.feature_mode,
                skip_count=skip_count,
                bank_size=args.bank_size,
                model_name=model_name,
            )
            for metric_name, values in [
                ("dev_delta_to_static", dev_selected["actual_delta_to_static"].to_numpy(dtype=np.float64)),
                ("dev_fraction_improved", dev_selected["improved_over_static"].to_numpy(dtype=np.float64)),
                ("dev_delta_to_bank_upper_bound", dev_selected["delta_to_bank_upper_bound"].to_numpy(dtype=np.float64)),
                ("dev_oracle_in_bank_match", dev_selected["oracle_in_bank_match"].to_numpy(dtype=np.float64)),
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
        print(
            f"[ranker-v7] completed skip={skip_count} models={len(models_eval)} "
            f"train_rows={len(train_rows)} eval_rows={len(eval_rows_df)}",
            flush=True,
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
        plt.savefig(plot_dir / f"{args.plot_prefix}_{args.output_tag}_{args.feature_mode}_delta_to_static.png", dpi=160)
        plt.close()


if __name__ == "__main__":
    main()
