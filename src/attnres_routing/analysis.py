"""Summary metrics, rank statistics, and figure helpers.

Once a model has run with ``record_mode != "none"``, every layer's
:class:`~attnres_routing.model.DepthMix` head produces depth-axis weight
matrices. This module turns those raw weights into the per-source ``utility``
and the rank-overlap statistics that show up in the closure reports.

What lives here:

- :func:`compute_utility_from_records` / :func:`compute_chunk_utility` —
  collapse the recorded ``(source_id, weights)`` traces into a per-source
  "how often / how much was this depth read from" scalar, optionally
  centered against the uniform baseline so positive values mean "above
  uniform usage." :class:`UtilitySummary` packages the full-prompt utility
  plus per-chunk utility / variance / top-k frequency — those three are the
  inputs to the routing-score modes in :mod:`attnres_routing.routing`.
- :func:`topk_overlap`, :func:`recall_at_k`, :func:`ndcg_at_k`,
  :func:`_safe_rank_corr` — the rank-quality metrics used to compare a
  predicted score vector against an oracle score vector. Used by the
  prompt-vs-decode transfer analyses.
- :func:`summarize_prompt_decode_transfer` — bundles spearman / kendall /
  jaccard / recall / nDCG into the single dict that ends up as one row in
  ``depth_support_raw.csv``.
- :func:`bootstrap_mean_ci` — the bootstrap CI helper every "mean ± CI"
  cell in the reports goes through.
- :func:`write_depth_support_outputs` and :func:`save_json` — the IO side
  that drops the CSV / PNGs that the reports embed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


@dataclass
class UtilitySummary:
    full_utility: np.ndarray
    chunk_utilities: np.ndarray
    chunk_variance: np.ndarray
    topk_frequency: np.ndarray


def compute_utility_from_records(
    records: list[dict[str, Any]],
    num_sources: int,
    token_start: int = 0,
    token_end: Optional[int] = None,
    normalize: bool = True,
    center_uniform: bool = True,
) -> np.ndarray:
    utility = np.zeros(num_sources, dtype=np.float64)
    exposure = np.zeros(num_sources, dtype=np.float64)
    for record in records:
        source_ids = np.asarray(record["source_ids"])
        weights = np.asarray(record["weights"])[:, 0, :]
        selected = weights[:, token_start:token_end]
        if selected.size == 0:
            continue
        baseline = 1.0 / max(selected.shape[0], 1)
        for src_idx, source_id in enumerate(source_ids.tolist()):
            values = selected[src_idx]
            if center_uniform:
                values = values - baseline
            utility[source_id] += float(values.sum())
            exposure[source_id] += selected.shape[1]
    if normalize:
        return utility / np.maximum(exposure, 1.0)
    return utility


def compute_chunk_utility(
    records: list[dict[str, Any]],
    num_sources: int,
    prompt_len: int,
    num_chunks: int,
    normalize: bool = True,
    center_uniform: bool = True,
) -> UtilitySummary:
    chunk_boundaries = np.linspace(0, prompt_len, num_chunks + 1, dtype=int)
    chunk_utilities = []
    for idx in range(num_chunks):
        chunk_utilities.append(
            compute_utility_from_records(
                records,
                num_sources=num_sources,
                token_start=int(chunk_boundaries[idx]),
                token_end=int(chunk_boundaries[idx + 1]),
                normalize=normalize,
                center_uniform=center_uniform,
            )
        )
    chunk_utilities = np.stack(chunk_utilities, axis=0)
    full_utility = compute_utility_from_records(
        records,
        num_sources=num_sources,
        token_start=0,
        token_end=prompt_len,
        normalize=normalize,
        center_uniform=center_uniform,
    )
    chunk_variance = chunk_utilities.var(axis=0)
    topk_frequency = np.zeros(num_sources, dtype=np.float64)
    topk = min(3, max(num_sources - 1, 1))
    for chunk in chunk_utilities:
        top_ids = np.argsort(chunk[1:])[::-1][:topk] + 1
        topk_frequency[top_ids] += 1
    topk_frequency /= max(num_chunks, 1)
    return UtilitySummary(
        full_utility=full_utility,
        chunk_utilities=chunk_utilities,
        chunk_variance=chunk_variance,
        topk_frequency=topk_frequency,
    )


def _safe_rank_corr(a: np.ndarray, b: np.ndarray, fn) -> float:
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    result = fn(a, b)
    return float(result.correlation if hasattr(result, "correlation") else result[0])


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    a_ids = set(np.argsort(a)[::-1][:k].tolist())
    b_ids = set(np.argsort(b)[::-1][:k].tolist())
    if not a_ids and not b_ids:
        return 1.0
    return len(a_ids & b_ids) / max(len(a_ids | b_ids), 1)


def recall_at_k(source_scores: np.ndarray, oracle_scores: np.ndarray, k: int) -> float:
    pred = set(np.argsort(source_scores)[::-1][:k].tolist())
    target = set(np.argsort(oracle_scores)[::-1][:k].tolist())
    return len(pred & target) / max(len(target), 1)


def ndcg_at_k(source_scores: np.ndarray, oracle_scores: np.ndarray, k: int) -> float:
    kk = min(k, len(source_scores))
    if kk <= 0:
        return float("nan")
    pred_order = np.argsort(source_scores)[::-1][:kk]
    ideal_order = np.argsort(oracle_scores)[::-1][:kk]
    gains = np.maximum(oracle_scores[pred_order], 0.0)
    ideal_gains = np.maximum(oracle_scores[ideal_order], 0.0)
    discounts = 1.0 / np.log2(np.arange(kk, dtype=np.float64) + 2.0)
    dcg = float((gains * discounts).sum())
    idcg = float((ideal_gains * discounts).sum())
    if idcg <= 0.0:
        return 1.0 if dcg <= 0.0 else 0.0
    return dcg / idcg


def summarize_prompt_decode_transfer(
    prompt_scores: np.ndarray,
    decode_scores: np.ndarray,
    ks: tuple[int, ...] = (1, 2, 3),
) -> dict[str, float]:
    prompt_mid = prompt_scores[1:]
    decode_mid = decode_scores[1:]
    degenerate = np.allclose(prompt_mid, prompt_mid[0]) or np.allclose(decode_mid, decode_mid[0])
    summary = {
        "spearman": _safe_rank_corr(prompt_mid, decode_mid, spearmanr),
        "kendall": _safe_rank_corr(prompt_mid, decode_mid, kendalltau),
    }
    for k in ks:
        kk = min(k, len(prompt_mid))
        if degenerate:
            summary[f"topk_jaccard_{kk}"] = float("nan")
            summary[f"recall_at_{kk}"] = float("nan")
            summary[f"ndcg_at_{kk}"] = float("nan")
        else:
            summary[f"topk_jaccard_{kk}"] = topk_overlap(prompt_mid, decode_mid, kk)
            summary[f"recall_at_{kk}"] = recall_at_k(prompt_mid, decode_mid, kk)
            summary[f"ndcg_at_{kk}"] = ndcg_at_k(prompt_mid, decode_mid, kk)
    return summary


def bootstrap_mean_ci(
    values: np.ndarray | list[float],
    num_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    if arr.size == 1:
        value = float(arr[0])
        return {"mean": value, "ci_low": value, "ci_high": value, "n": 1}
    samples = rng.choice(arr, size=(num_bootstrap, arr.size), replace=True)
    means = samples.mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(arr.mean()),
        "ci_low": float(np.quantile(means, alpha)),
        "ci_high": float(np.quantile(means, 1.0 - alpha)),
        "n": int(arr.size),
    }


def write_depth_support_outputs(
    rows: list[dict[str, Any]],
    raw_path: Path,
    plot_prefix: Path,
) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(raw_path, index=False)

    if "spearman" in df.columns and "prompt_entropy" in df.columns:
        plt.figure(figsize=(5, 4))
        plt.scatter(df["prompt_entropy"], df["spearman"], alpha=0.7)
        plt.xlabel("prompt entropy")
        plt.ylabel("prompt vs decode spearman")
        plt.tight_layout()
        plt.savefig(f"{plot_prefix}_entropy_vs_spearman.png", dpi=160)
        plt.close()

    if "prompt_decode_corr_proxy" in df.columns:
        plt.figure(figsize=(5, 4))
        plt.hist(df["prompt_decode_corr_proxy"], bins=20)
        plt.xlabel("prompt/decode utility dot-product")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(f"{plot_prefix}_corr_proxy_hist.png", dpi=160)
        plt.close()


def save_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
