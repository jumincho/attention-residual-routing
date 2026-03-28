#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


def concat_frames(base_dir: Path, tags: list[str], suffix: str) -> pd.DataFrame:
    frames = [pd.read_csv(base_dir / f"{tag}_{suffix}.csv") for tag in tags]
    return pd.concat(frames, ignore_index=True)


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
    # Sublayer candidates operate on concrete model blocks, so we drop the
    # always-available embedding source `b0` but keep the final block signal.
    combined = parse_json_vec(row["prompt_scores_json"])[1:]
    attn = parse_json_vec(row["prompt_scores_attn_json"])[1:]
    mlp = parse_json_vec(row["prompt_scores_mlp_json"])[1:]
    attnres = np.concatenate(
        [
            combined,
            attn,
            mlp,
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
    merge_keys = [
        column
        for column in ["sequence_idx", "split", "document_idx", "window_idx"]
        if column in feature_df.columns and column in oracle_df.columns
    ]
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


def parse_mask_json(mask_json: str) -> np.ndarray:
    return np.asarray(json.loads(mask_json), dtype=np.float32)


def candidate_feature_vector(row: pd.Series, candidate_row: pd.Series, global_attn: np.ndarray, global_mlp: np.ndarray) -> np.ndarray:
    combined = parse_json_vec(row["prompt_scores_json"])[1:]
    prompt_attn = parse_json_vec(row["prompt_scores_attn_json"])[1:]
    prompt_mlp = parse_json_vec(row["prompt_scores_mlp_json"])[1:]

    attn_mask = parse_mask_json(candidate_row["attn_mask_json"])
    mlp_mask = parse_mask_json(candidate_row["mlp_mask_json"])
    delta_attn = attn_mask - global_attn
    delta_mlp = mlp_mask - global_mlp

    keep_attn_score = float(np.dot(prompt_attn, attn_mask))
    keep_mlp_score = float(np.dot(prompt_mlp, mlp_mask))
    drop_attn_score = float(np.dot(prompt_attn, 1.0 - attn_mask))
    drop_mlp_score = float(np.dot(prompt_mlp, 1.0 - mlp_mask))
    keep_combined_score = float(np.dot(combined, np.minimum(attn_mask, mlp_mask)))

    return np.concatenate(
        [
            attn_mask.astype(np.float32),
            mlp_mask.astype(np.float32),
            delta_attn.astype(np.float32),
            delta_mlp.astype(np.float32),
            np.asarray(
                [
                    float(candidate_row["estimated_decode_seconds"]),
                    float(candidate_row["estimated_reduction_ratio"]),
                    float(candidate_row["min_anchor_edit_distance"]),
                    float(candidate_row["attn_skip_count"]),
                    float(candidate_row["mlp_skip_count"]),
                    float(candidate_row["whole_block_skip_count"]),
                    keep_attn_score,
                    keep_mlp_score,
                    drop_attn_score,
                    drop_mlp_score,
                    keep_combined_score,
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
            np.asarray([float(np.linalg.norm(prompt_vec)), float(np.linalg.norm(cand_vec))], dtype=np.float32),
        ],
        axis=0,
    )


@dataclass
class SelectorDataset:
    rows: pd.DataFrame
    prompt_vectors: dict[int, np.ndarray]
    candidate_bank: pd.DataFrame
    global_static_id: str
    global_attn: np.ndarray
    global_mlp: np.ndarray


def prepare_budget_dataset(
    candidate_losses: pd.DataFrame,
    candidate_defs: pd.DataFrame,
    prompt_features: pd.DataFrame,
    feature_mode: str,
    budget_low: float,
    budget_high: float,
    feature_skip_count: int,
    candidate_bank: pd.DataFrame | None = None,
    global_static_id: str | None = None,
) -> SelectorDataset:
    if candidate_bank is None:
        candidate_bank = candidate_defs[
            (candidate_defs["estimated_reduction_ratio"] >= budget_low)
            & (candidate_defs["estimated_reduction_ratio"] < budget_high)
        ].copy()
    else:
        candidate_bank = candidate_bank.copy()
    if candidate_bank.empty:
        raise ValueError(f"No candidates for budget band [{budget_low}, {budget_high})")
    candidate_bank = candidate_bank.sort_values(["estimated_decode_seconds", "candidate_id"]).drop_duplicates("candidate_id")
    loss_subset = candidate_losses[candidate_losses["candidate_id"].isin(candidate_bank["candidate_id"])].copy()
    if global_static_id is None:
        global_means = (
            loss_subset.groupby("candidate_id", as_index=False)["continuation_loss"]
            .mean()
            .sort_values("continuation_loss")
            .reset_index(drop=True)
        )
        global_static_id = str(global_means.iloc[0]["candidate_id"])
    global_row = candidate_bank[candidate_bank["candidate_id"] == global_static_id].iloc[0]
    global_attn = parse_mask_json(global_row["attn_mask_json"])
    global_mlp = parse_mask_json(global_row["mlp_mask_json"])

    feature_rows = (
        prompt_features[prompt_features["skip_count"] == feature_skip_count]
        .drop_duplicates(subset=["split", "document_idx", "window_idx"])
        .set_index(["split", "document_idx", "window_idx"])
    )
    sequence_meta = (
        loss_subset[["sequence_idx", "split", "document_idx", "window_idx"]]
        .drop_duplicates(subset=["sequence_idx"])
        .set_index("sequence_idx")
    )
    pivot = loss_subset.pivot(index="sequence_idx", columns="candidate_id", values="continuation_loss")

    rows = []
    prompt_vectors = {}
    for seq_id in pivot.index.tolist():
        meta_row = sequence_meta.loc[int(seq_id)]
        prompt_row = feature_rows.loc[(str(meta_row["split"]), int(meta_row["document_idx"]), int(meta_row["window_idx"]))]
        prompt_vec = prompt_feature_vector(prompt_row, feature_mode)
        prompt_vectors[int(seq_id)] = prompt_vec
        global_loss = float(pivot.loc[int(seq_id), global_static_id])
        for _, cand_row in candidate_bank.iterrows():
            cand_id = str(cand_row["candidate_id"])
            cand_vec = candidate_feature_vector(prompt_row, cand_row, global_attn=global_attn, global_mlp=global_mlp)
            rows.append(
                {
                    "sequence_idx": int(seq_id),
                    "document_idx": int(meta_row["document_idx"]),
                    "candidate_id": cand_id,
                    "actual_loss": float(pivot.loc[int(seq_id), cand_id]),
                    "delta_to_global_static": float(pivot.loc[int(seq_id), cand_id] - global_loss),
                    "pair_features": pair_feature_vector(prompt_vec, cand_vec),
                    "candidate_features": cand_vec,
                }
            )
    return SelectorDataset(
        rows=pd.DataFrame(rows),
        prompt_vectors=prompt_vectors,
        candidate_bank=candidate_bank,
        global_static_id=global_static_id,
        global_attn=global_attn,
        global_mlp=global_mlp,
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
        return self.head(torch.cat([prompt_h, cand_h, prompt_h * cand_h], dim=-1)).squeeze(-1)


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
    train_candidate_ids: np.ndarray,
    train_y: np.ndarray,
    eval_prompt: np.ndarray,
    eval_candidate_ids: np.ndarray,
    k: int = 16,
) -> np.ndarray:
    scaler = StandardScaler().fit(train_prompt)
    train_z = scaler.transform(train_prompt)
    eval_z = scaler.transform(eval_prompt)
    nbrs = NearestNeighbors(n_neighbors=min(k, train_z.shape[0]), metric="euclidean").fit(train_z)
    _, indices = nbrs.kneighbors(eval_z)
    preds = np.zeros(eval_prompt.shape[0], dtype=np.float64)
    for row_idx in range(eval_prompt.shape[0]):
        neighbor_ids = indices[row_idx]
        neighbor_mask = train_candidate_ids[neighbor_ids] == eval_candidate_ids[row_idx]
        if neighbor_mask.any():
            preds[row_idx] = float(train_y[neighbor_ids[neighbor_mask]].mean())
        else:
            same_candidate = train_candidate_ids == eval_candidate_ids[row_idx]
            preds[row_idx] = float(train_y[same_candidate].mean()) if same_candidate.any() else float(train_y.mean())
    return preds


def bootstrap_summary(values: np.ndarray) -> dict[str, float]:
    return bootstrap_mean_ci(values.astype(np.float64), seed=42)


def evaluate_selected(
    rows_df: pd.DataFrame,
    score_col: str,
    global_static_id: str,
    candidate_meta: pd.DataFrame,
    output_tag: str,
    feature_mode: str,
    budget_label: str,
    model_name: str,
) -> pd.DataFrame:
    per_sequence = []
    global_lookup = (
        rows_df[rows_df["candidate_id"] == global_static_id][["sequence_idx", "actual_loss"]]
        .rename(columns={"actual_loss": "global_static_loss"})
        .set_index("sequence_idx")
    )
    candidate_meta = candidate_meta.copy()
    for metric in ["decode_seconds_per_sequence", "decode_tokens_per_sec"]:
        if metric not in candidate_meta.columns:
            left = f"{metric}_x"
            right = f"{metric}_y"
            if left in candidate_meta.columns or right in candidate_meta.columns:
                candidate_meta[metric] = candidate_meta.get(left).combine_first(candidate_meta.get(right))
    for metric in ["estimated_reduction_ratio", "estimated_decode_seconds", "decode_seconds_per_sequence", "decode_tokens_per_sec"]:
        if metric not in candidate_meta.columns:
            raise KeyError(f"Missing candidate metadata column: {metric}")
    candidate_meta = candidate_meta.set_index("candidate_id")
    bank_upper = rows_df.groupby("sequence_idx")["actual_loss"].min()
    bank_best = rows_df.sort_values(["sequence_idx", "actual_loss"]).groupby("sequence_idx").first()["candidate_id"]
    for seq_id, group in rows_df.groupby("sequence_idx", sort=False):
        best_pos = int(np.argmin(group[score_col].to_numpy(dtype=np.float64)))
        chosen = group.iloc[best_pos]
        candidate_id = str(chosen["candidate_id"])
        row = {
            "output_tag": output_tag,
            "feature_mode": feature_mode,
            "budget_label": budget_label,
            "model_name": model_name,
            "sequence_idx": int(seq_id),
            "selected_candidate_id": candidate_id,
            "actual_delta_to_static": float(chosen["actual_loss"] - global_lookup.loc[int(seq_id), "global_static_loss"]),
            "delta_to_bank_upper_bound": float(chosen["actual_loss"] - bank_upper.loc[int(seq_id)]),
            "improved_over_static": float(chosen["actual_loss"] < global_lookup.loc[int(seq_id), "global_static_loss"]),
            "oracle_in_bank_match": float(candidate_id == str(bank_best.loc[int(seq_id)])),
            "estimated_reduction_ratio": float(candidate_meta.loc[candidate_id, "estimated_reduction_ratio"]),
            "estimated_decode_seconds": float(candidate_meta.loc[candidate_id, "estimated_decode_seconds"]),
            "decode_seconds_per_sequence": float(candidate_meta.loc[candidate_id, "decode_seconds_per_sequence"]),
            "decode_tokens_per_sec": float(candidate_meta.loc[candidate_id, "decode_tokens_per_sec"]),
        }
        per_sequence.append(row)
    return pd.DataFrame(per_sequence)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-candidate-tags", type=str, nargs="+", required=True)
    parser.add_argument("--eval-candidate-tags", type=str, nargs="+", required=True)
    parser.add_argument("--train-feature-tags", type=str, nargs="+", required=True)
    parser.add_argument("--eval-feature-tags", type=str, nargs="+", required=True)
    parser.add_argument("--feature-mode", type=str, default="attnres_hidden_difficulty")
    parser.add_argument("--hidden-train-tags", type=str, nargs="*", default=None)
    parser.add_argument("--hidden-eval-tags", type=str, nargs="*", default=None)
    parser.add_argument("--output-tag", type=str, required=True)
    parser.add_argument("--budget-bands", type=str, nargs="+", default=["0.05:0.10", "0.10:0.15", "0.15:0.20"])
    parser.add_argument("--train-doc-frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    latency_dir = ROOT / "results" / "latency_budgeted_sublayer_v5"
    oracle_dir = ROOT / "results" / "oracles"
    rich_dir = ROOT / "results" / "rich_features"
    out_dir = ROOT / "results" / "latency_budgeted_sublayer_v5"
    plot_dir = ROOT / "results" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    train_losses = concat_frames(latency_dir, args.train_candidate_tags, "candidate_losses")
    eval_losses = concat_frames(latency_dir, args.eval_candidate_tags, "candidate_losses")
    train_summary = concat_frames(latency_dir, args.train_candidate_tags, "candidate_summary")
    eval_summary = concat_frames(latency_dir, args.eval_candidate_tags, "candidate_summary")
    train_defs = concat_frames(latency_dir, args.train_candidate_tags, "candidate_defs").drop_duplicates("candidate_id")
    eval_defs = concat_frames(latency_dir, args.eval_candidate_tags, "candidate_defs").drop_duplicates("candidate_id")
    candidate_defs = train_defs.copy()
    train_features = concat_frames(oracle_dir, args.train_feature_tags, "sequence_features")
    eval_features = concat_frames(oracle_dir, args.eval_feature_tags, "sequence_features")
    train_oracle = concat_frames(oracle_dir, args.train_feature_tags, "oracle_mask_alignment")
    eval_oracle = concat_frames(oracle_dir, args.eval_feature_tags, "oracle_mask_alignment")
    train_hidden = concat_frames(rich_dir, args.hidden_train_tags, "hidden_prompt_features") if args.hidden_train_tags else None
    eval_hidden = concat_frames(rich_dir, args.hidden_eval_tags, "hidden_prompt_features") if args.hidden_eval_tags else None

    train_feature_table = build_feature_table(train_features, train_oracle, train_hidden)
    eval_feature_table = build_feature_table(eval_features, eval_oracle, eval_hidden)

    rng = np.random.default_rng(args.seed)
    per_sequence_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    model_selection_rows: list[dict[str, Any]] = []

    for band in args.budget_bands:
        low_s, high_s = band.split(":", 1)
        budget_low = float(low_s)
        budget_high = float(high_s)
        budget_label = f"{int(round(budget_low * 100)):02d}-{int(round(budget_high * 100)):02d}"
        if budget_low < 0.10:
            feature_skip_count = 1
        elif budget_low < 0.15:
            feature_skip_count = 2
        else:
            feature_skip_count = 3

        train_ds = prepare_budget_dataset(
            train_losses,
            candidate_defs,
            train_feature_table,
            args.feature_mode,
            budget_low,
            budget_high,
            feature_skip_count,
        )
        eval_ds = prepare_budget_dataset(
            eval_losses,
            candidate_defs,
            eval_feature_table,
            args.feature_mode,
            budget_low,
            budget_high,
            feature_skip_count,
            candidate_bank=train_ds.candidate_bank,
            global_static_id=train_ds.global_static_id,
        )

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

        models_dev: dict[str, np.ndarray] = {}
        models_eval: dict[str, np.ndarray] = {}

        hgb = HistGradientBoostingRegressor(max_depth=6, max_iter=300, learning_rate=0.05, random_state=args.seed)
        hgb.fit(x_train_pair, y_train)
        models_dev["hgb_pair"] = hgb.predict(x_dev_pair)
        models_eval["hgb_pair"] = hgb.predict(x_eval_pair)

        rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=14,
            min_samples_leaf=2,
            random_state=args.seed,
            n_jobs=-1,
        )
        rf.fit(x_train_pair, y_train)
        models_dev["rf_pair"] = rf.predict(x_dev_pair)
        models_eval["rf_pair"] = rf.predict(x_eval_pair)

        models_dev["knn_prompt"] = knn_predict(
            x_train_prompt,
            train_rows["candidate_id"].to_numpy(dtype=object),
            y_train,
            x_dev_prompt,
            dev_rows["candidate_id"].to_numpy(dtype=object),
        )
        models_eval["knn_prompt"] = knn_predict(
            x_train_prompt,
            train_rows["candidate_id"].to_numpy(dtype=object),
            y_train,
            x_eval_prompt,
            eval_rows_df["candidate_id"].to_numpy(dtype=object),
        )

        mlp_model, prompt_scaler, cand_scaler = train_mlp(x_train_prompt, x_train_cand, y_train, seed=args.seed)
        models_dev["mlp_late_fusion"] = predict_mlp(mlp_model, prompt_scaler, cand_scaler, x_dev_prompt, x_dev_cand)
        models_eval["mlp_late_fusion"] = predict_mlp(mlp_model, prompt_scaler, cand_scaler, x_eval_prompt, x_eval_cand)

        dev_rows_scored_cache: dict[str, pd.DataFrame] = {}
        eval_rows_scored_cache: dict[str, pd.DataFrame] = {}
        for model_name, preds in models_dev.items():
            dev_rows_scored = dev_rows.copy()
            dev_rows_scored[model_name] = preds
            dev_selected = evaluate_selected(
                rows_df=dev_rows_scored,
                score_col=model_name,
                global_static_id=train_ds.global_static_id,
                candidate_meta=train_ds.candidate_bank.merge(
                    train_summary[["candidate_id", "decode_seconds_per_sequence", "decode_tokens_per_sec"]].drop_duplicates("candidate_id"),
                    on="candidate_id",
                    how="left",
                ),
                output_tag=args.output_tag,
                feature_mode=args.feature_mode,
                budget_label=budget_label,
                model_name=model_name,
            )
            dev_rows_scored_cache[model_name] = dev_selected
            for metric_name, values in [
                ("dev_delta_to_static", dev_selected["actual_delta_to_static"].to_numpy(dtype=np.float64)),
                ("dev_fraction_improved", dev_selected["improved_over_static"].to_numpy(dtype=np.float64)),
            ]:
                model_selection_rows.append(
                    {
                        "output_tag": args.output_tag,
                        "feature_mode": args.feature_mode,
                        "budget_label": budget_label,
                        "model_name": model_name,
                        "metric": metric_name,
                        **bootstrap_summary(values),
                    }
                )

        best_model_by_metric: dict[str, str] = {}
        selection_df = pd.DataFrame(model_selection_rows)
        metric_df = selection_df[
            (selection_df["output_tag"] == args.output_tag)
            & (selection_df["feature_mode"] == args.feature_mode)
            & (selection_df["budget_label"] == budget_label)
        ]
        best_model_by_metric["delta"] = str(metric_df[metric_df["metric"] == "dev_delta_to_static"].sort_values("mean").iloc[0]["model_name"])
        best_model_by_metric["improve"] = str(metric_df[metric_df["metric"] == "dev_fraction_improved"].sort_values("mean", ascending=False).iloc[0]["model_name"])

        for target_name, model_name in best_model_by_metric.items():
            eval_rows_scored = eval_rows_df.copy()
            eval_rows_scored[model_name] = models_eval[model_name]
            selected = evaluate_selected(
                rows_df=eval_rows_scored,
                score_col=model_name,
                global_static_id=eval_ds.global_static_id,
                candidate_meta=eval_ds.candidate_bank.merge(
                    eval_summary[["candidate_id", "decode_seconds_per_sequence", "decode_tokens_per_sec"]].drop_duplicates("candidate_id"),
                    on="candidate_id",
                    how="left",
                ),
                output_tag=args.output_tag,
                feature_mode=args.feature_mode,
                budget_label=budget_label,
                model_name=f"{target_name}:{model_name}",
            )
            eval_rows_scored_cache[f"{target_name}:{model_name}"] = selected
            per_sequence_rows.extend(selected.to_dict(orient="records"))
            for metric_name, values in [
                ("delta_to_static", selected["actual_delta_to_static"].to_numpy(dtype=np.float64)),
                ("delta_to_bank_upper_bound", selected["delta_to_bank_upper_bound"].to_numpy(dtype=np.float64)),
                ("fraction_improved", selected["improved_over_static"].to_numpy(dtype=np.float64)),
                ("oracle_in_bank_match", selected["oracle_in_bank_match"].to_numpy(dtype=np.float64)),
                ("decode_seconds_per_sequence", selected["decode_seconds_per_sequence"].to_numpy(dtype=np.float64)),
                ("decode_tokens_per_sec", selected["decode_tokens_per_sec"].to_numpy(dtype=np.float64)),
            ]:
                summary_rows.append(
                    {
                        "output_tag": args.output_tag,
                        "feature_mode": args.feature_mode,
                        "budget_label": budget_label,
                        "model_name": f"{target_name}:{model_name}",
                        "metric": metric_name,
                        **bootstrap_summary(values),
                    }
                )

        summary_df = pd.DataFrame(summary_rows)
        if not summary_df.empty:
            plot_df = summary_df[
                (summary_df["output_tag"] == args.output_tag)
                & (summary_df["feature_mode"] == args.feature_mode)
                & (summary_df["budget_label"] == budget_label)
                & (summary_df["metric"] == "delta_to_static")
            ]
            if not plot_df.empty:
                plt.figure(figsize=(6, 4))
                plt.bar(plot_df["model_name"], plot_df["mean"])
                plt.ylabel("delta to static")
                plt.xticks(rotation=30, ha="right")
                plt.tight_layout()
                plt.savefig(
                    plot_dir / f"latency_budgeted_sublayer_v5_{args.output_tag}_{args.feature_mode}_{budget_label}_delta_to_static.png",
                    dpi=160,
                )
                plt.close()

    pd.DataFrame(per_sequence_rows).to_csv(out_dir / f"{args.output_tag}_{args.feature_mode}_selector_per_sequence.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out_dir / f"{args.output_tag}_{args.feature_mode}_selector_summary.csv", index=False)
    pd.DataFrame(model_selection_rows).to_csv(out_dir / f"{args.output_tag}_{args.feature_mode}_selector_model_selection.csv", index=False)


if __name__ == "__main__":
    main()
