#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attnres_routing.analysis import (  # noqa: E402
    bootstrap_mean_ci,
    compute_chunk_utility,
    ndcg_at_k,
    recall_at_k,
    topk_overlap,
)
from attnres_routing.data import DataConfig, prepare_lm_dataset_splits  # noqa: E402
from attnres_routing.model import AttnResConfig, DecoderLM  # noqa: E402
from attnres_routing.routing import (  # noqa: E402
    balanced_skip_route,
    compute_routing_scores,
    continuation_losses_from_decode_logits,
    random_skip_route,
    select_prompt_fixed_route,
    stack_past_key_values,
    teacher_forced_decode_from_past,
)
from attnres_routing.sequence_manifest import load_manifest_jsonl  # noqa: E402
from attnres_routing.utils import ensure_dir, save_yaml  # noqa: E402


def load_checkpoint(path: Path, device: torch.device, precision: str) -> tuple[DecoderLM, dict[str, Any], int]:
    payload = torch.load(path, map_location=device)
    config = payload["config"]
    model = DecoderLM(AttnResConfig.from_dict(config["model"])).to(device)
    model.load_state_dict(payload["model_state"])
    if precision == "fp16" and device.type == "cuda":
        model = model.half()
    model.eval()
    return model, config, int(payload.get("step", 0))


def middle_block_masks(num_blocks: int, skip_count: int) -> list[np.ndarray]:
    middle_ids = list(range(num_blocks - 1))
    masks = []
    for skipped in itertools.combinations(middle_ids, skip_count):
        mask = np.ones(num_blocks, dtype=np.bool_)
        for block_idx in skipped:
            mask[block_idx] = False
        mask[-1] = True
        masks.append(mask)
    return masks


def mask_to_id(mask: np.ndarray) -> str:
    kept = [str(idx + 1) for idx, value in enumerate(mask.tolist()) if value]
    return "keep:" + ",".join(kept)


def safe_rank_corr(a: np.ndarray, b: np.ndarray, fn) -> float:
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    result = fn(a, b)
    return float(result.correlation if hasattr(result, "correlation") else result[0])


def summarize_alignment(scores: np.ndarray, targets: np.ndarray, ks: tuple[int, ...] = (1, 2, 3)) -> dict[str, float]:
    summary = {
        "spearman": safe_rank_corr(scores, targets, spearmanr),
        "kendall": safe_rank_corr(scores, targets, kendalltau),
    }
    for k in ks:
        kk = min(k, len(scores))
        summary[f"topk_jaccard_{kk}"] = topk_overlap(scores, targets, kk)
        summary[f"recall_at_{kk}"] = recall_at_k(scores, targets, kk)
        summary[f"ndcg_at_{kk}"] = ndcg_at_k(scores, targets, kk)
    return summary


def mean_pairwise_chunk_spearman(chunk_utilities: np.ndarray) -> float:
    if chunk_utilities.shape[0] <= 1:
        return float("nan")
    values = []
    for a_idx in range(chunk_utilities.shape[0]):
        for b_idx in range(a_idx + 1, chunk_utilities.shape[0]):
            values.append(safe_rank_corr(chunk_utilities[a_idx], chunk_utilities[b_idx], spearmanr))
    if not values or np.all(np.isnan(values)):
        return 0.0
    return float(np.nanmean(values))


def mean_pairwise_chunk_jaccard(chunk_utilities: np.ndarray, k: int = 3) -> float:
    if chunk_utilities.shape[0] <= 1:
        return float("nan")
    values = []
    for a_idx in range(chunk_utilities.shape[0]):
        for b_idx in range(a_idx + 1, chunk_utilities.shape[0]):
            values.append(topk_overlap(chunk_utilities[a_idx], chunk_utilities[b_idx], min(k, chunk_utilities.shape[1])))
    if not values or np.all(np.isnan(values)):
        return 0.0
    return float(np.nanmean(values))


def prompt_margin(scores: np.ndarray, skip_count: int) -> float:
    middle_scores = scores[1:-1]
    keep_middle = max(len(middle_scores) - skip_count, 0)
    if keep_middle <= 0 or keep_middle >= len(middle_scores):
        return float("nan")
    ordered = np.sort(middle_scores)[::-1]
    return float(ordered[keep_middle - 1] - ordered[keep_middle])


def bootstrap_metric_table(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str], seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = {col: value for col, value in zip(group_cols, keys)}
        for metric in metric_cols:
            ci = bootstrap_mean_ci(group[metric].to_numpy(), seed=seed)
            rows.append({**key_map, "metric": metric, **ci})
    return pd.DataFrame(rows)


def batched(iterable: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [iterable[start : start + batch_size] for start in range(0, len(iterable), batch_size)]


def evaluate_mask_over_cache(
    model: DecoderLM,
    sequence_cache: list[dict[str, Any]],
    mask: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[int, float], float, float]:
    mask_tensor = torch.tensor(mask, device=device, dtype=torch.bool)
    losses_by_sequence: dict[int, float] = {}
    total_elapsed = 0.0
    total_effective_tokens = 0

    for batch_rows in batched(sequence_cache, batch_size):
        continuation_ids = torch.cat([row["continuation_ids"] for row in batch_rows], dim=0)
        prompt_last_logits = torch.cat([row["prompt_last_logits"] for row in batch_rows], dim=0)
        past_key_values = stack_past_key_values([row["past_key_values"] for row in batch_rows])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        decode_logits = teacher_forced_decode_from_past(
            model,
            continuation_ids=continuation_ids,
            past_key_values=past_key_values,
            prompt_last_logits=prompt_last_logits,
            active_block_mask=mask_tensor,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        total_elapsed += time.perf_counter() - t0
        total_effective_tokens += continuation_ids.size(0) * max(continuation_ids.size(1) - 1, 0)
        losses = continuation_losses_from_decode_logits(decode_logits, continuation_ids).detach().cpu().numpy()
        for row, loss in zip(batch_rows, losses.tolist()):
            losses_by_sequence[int(row["sequence_idx"])] = float(loss)
        del decode_logits

    decode_tokens_per_sec = total_effective_tokens / max(total_elapsed, 1e-6)
    decode_seconds_per_sequence = total_elapsed / max(len(sequence_cache), 1)
    return losses_by_sequence, decode_seconds_per_sequence, decode_tokens_per_sec


def write_functional_oracles_doc(
    path: Path,
    step: int,
    loo_summary: pd.DataFrame,
    mask_summary: pd.DataFrame,
) -> None:
    def lookup(summary_df: pd.DataFrame, budget: int, metric: str) -> str:
        row = summary_df[(summary_df["skip_count"] == budget) & (summary_df["metric"] == metric)].iloc[0]
        return f"{row['mean']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}]"

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Functional Oracles\n\n")
        f.write(f"- checkpoint step: {step}\n\n")
        if not loo_summary.empty:
            f.write("## Leave-one-block-out alignment\n\n")
            for metric in ["spearman", "kendall", "topk_jaccard_1", "recall_at_3", "ndcg_at_3"]:
                row = loo_summary[loo_summary["metric"] == metric].iloc[0]
                f.write(f"- {metric}: {row['mean']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}]\n")
            f.write("\n")
        if not mask_summary.empty:
            f.write("## Exhaustive oracle-mask search\n\n")
            for budget in sorted(mask_summary["skip_count"].unique().tolist()):
                f.write(f"### skip {budget} middle block(s)\n\n")
                f.write(f"- oracle match rate: {lookup(mask_summary, budget, 'oracle_exact_match')}\n")
                f.write(f"- prompt-vs-oracle mask overlap: {lookup(mask_summary, budget, 'oracle_mask_jaccard')}\n")
                f.write(f"- prompt delta NLL to oracle: {lookup(mask_summary, budget, 'delta_to_oracle')}\n")
                f.write(f"- prompt delta NLL to best global static: {lookup(mask_summary, budget, 'delta_to_global_static')}\n\n")


def write_routing_doc(path: Path, summary_df: pd.DataFrame) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Routing Baselines\n\n")
        for skip_count in sorted(summary_df["skip_count"].unique().tolist()):
            subset = summary_df[summary_df["skip_count"] == skip_count]
            f.write(f"## Skip {skip_count} middle block(s)\n\n")
            for method in [
                "no_skip",
                "random",
                "balanced",
                "mismatched_prompt",
                "global_static",
                "oracle_sequence",
                "prompt_fixed",
            ]:
                method_df = subset[subset["method"] == method]
                if method_df.empty:
                    continue
                f.write(f"### {method}\n\n")
                for metric in ["continuation_loss", "decode_tokens_per_sec", "end_to_end_seconds"]:
                    row = method_df[method_df["metric"] == metric].iloc[0]
                    f.write(f"- {metric}: {row['mean']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}]\n")
                f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-sequences", type=int, default=32)
    parser.add_argument("--prompt-len", type=int, default=256)
    parser.add_argument("--decode-len", type=int, default=64)
    parser.add_argument("--num-chunks", type=int, default=4)
    parser.add_argument("--skip-counts", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--score-mode", type=str, default="utility_over_variance")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--sequence-offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--tag", type=str, default="main")
    parser.add_argument("--manifest-path", type=str, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-docs", action="store_true")
    parser.add_argument("--prompt-features-only", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    model, config, checkpoint_step = load_checkpoint(Path(args.checkpoint), device, args.precision)
    if args.num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")

    manifest_rows: list[dict[str, Any]] | None = None
    target_ds = None
    if args.manifest_path:
        manifest_rows = load_manifest_jsonl(Path(args.manifest_path))
        if args.num_shards > 1:
            manifest_rows = manifest_rows[args.shard_index :: args.num_shards]
        if args.sequence_offset > 0:
            manifest_rows = manifest_rows[args.sequence_offset :]
        if args.num_sequences > 0:
            manifest_rows = manifest_rows[: args.num_sequences]
    else:
        split_datasets, _ = prepare_lm_dataset_splits(DataConfig(**config["data"]))
        if args.split not in split_datasets:
            raise ValueError(f"Split {args.split!r} not available. Found {sorted(split_datasets.keys())}")
        target_ds = split_datasets[args.split]

    oracles_dir = ensure_dir(ROOT / "results" / "oracles")
    routing_dir = ensure_dir(ROOT / "results" / "routing")
    plot_dir = ensure_dir(ROOT / "results" / "plots")
    tag_prefix = f"{args.tag}_" if args.tag else ""

    num_blocks = model.config.num_blocks
    middle_count = num_blocks - 1
    skip_counts = sorted({count for count in args.skip_counts if 0 < count <= middle_count})
    masks_by_budget = {skip_count: middle_block_masks(num_blocks, skip_count) for skip_count in skip_counts}
    all_on = np.ones(num_blocks, dtype=np.bool_)
    all_masks: dict[str, np.ndarray] = {mask_to_id(all_on): all_on}
    for skip_count in skip_counts:
        for mask in masks_by_budget[skip_count]:
            all_masks.setdefault(mask_to_id(mask), mask)

    sequence_cache: list[dict[str, Any]] = []
    loo_rows: list[dict[str, Any]] = []
    mask_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []

    if manifest_rows is not None:
        target_sequences = len(manifest_rows)
    else:
        max_candidates = max(len(target_ds) - args.sequence_offset, 0)
        target_sequences = min(args.num_sequences, max_candidates)
    for local_idx in range(target_sequences):
        if manifest_rows is not None:
            source_row = manifest_rows[local_idx]
            seq_idx = int(source_row["sequence_idx"])
            split_name = str(source_row.get("source_split", args.split))
            document_idx = int(source_row.get("document_idx", -1))
            document_title = str(source_row.get("document_title", ""))
            window_idx = int(source_row.get("window_idx", 0))
            tokens = torch.tensor(source_row["input_ids"], dtype=torch.long)
        else:
            seq_idx = args.sequence_offset + local_idx
            split_name = args.split
            document_idx = -1
            document_title = ""
            window_idx = 0
            tokens = target_ds[seq_idx]["input_ids"]
        if tokens.numel() < args.prompt_len + args.decode_len:
            continue
        prompt_ids = tokens[: args.prompt_len].unsqueeze(0).to(device)
        continuation_ids = tokens[args.prompt_len : args.prompt_len + args.decode_len].unsqueeze(0).to(device)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_prefill = time.perf_counter()
        with torch.no_grad():
            prompt_outputs = model(
                prompt_ids,
                labels=None,
                use_cache=True,
                active_block_mask=None,
                record_mode="full",
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        prefill_seconds = time.perf_counter() - t_prefill
        past_key_values = prompt_outputs["past_key_values"]
        t_route = time.perf_counter()
        records = prompt_outputs["stats"]["records"]
        if records:
            prompt_summary = compute_chunk_utility(
                records,
                num_sources=num_blocks + 1,
                prompt_len=args.prompt_len,
                num_chunks=args.num_chunks,
                normalize=True,
                center_uniform=True,
            )
            attn_records = [record for record in records if record["sublayer"] == "attn"]
            mlp_records = [record for record in records if record["sublayer"] == "mlp"]
            attn_summary = compute_chunk_utility(
                attn_records,
                num_sources=num_blocks + 1,
                prompt_len=args.prompt_len,
                num_chunks=args.num_chunks,
                normalize=True,
                center_uniform=True,
            )
            mlp_summary = compute_chunk_utility(
                mlp_records,
                num_sources=num_blocks + 1,
                prompt_len=args.prompt_len,
                num_chunks=args.num_chunks,
                normalize=True,
                center_uniform=True,
            )
        else:
            zeros = np.zeros(num_blocks + 1, dtype=np.float64)
            zero_chunks = np.zeros((args.num_chunks, num_blocks + 1), dtype=np.float64)
            prompt_summary = SimpleNamespace(
                full_utility=zeros.copy(),
                chunk_variance=zeros.copy(),
                topk_frequency=zeros.copy(),
                chunk_utilities=zero_chunks.copy(),
            )
            attn_summary = SimpleNamespace(
                full_utility=zeros.copy(),
                chunk_variance=zeros.copy(),
                topk_frequency=zeros.copy(),
                chunk_utilities=zero_chunks.copy(),
            )
            mlp_summary = SimpleNamespace(
                full_utility=zeros.copy(),
                chunk_variance=zeros.copy(),
                topk_frequency=zeros.copy(),
                chunk_utilities=zero_chunks.copy(),
            )
        prompt_last_logits = prompt_outputs["logits"][:, -1:, :]
        prompt_scores = compute_routing_scores(
            prompt_summary.full_utility,
            prompt_summary.chunk_variance,
            prompt_summary.topk_frequency,
            mode=args.score_mode,
        )
        prompt_scores_attn = compute_routing_scores(
            attn_summary.full_utility,
            attn_summary.chunk_variance,
            attn_summary.topk_frequency,
            mode=args.score_mode,
        )
        prompt_scores_mlp = compute_routing_scores(
            mlp_summary.full_utility,
            mlp_summary.chunk_variance,
            mlp_summary.topk_frequency,
            mode=args.score_mode,
        )
        route_seconds = time.perf_counter() - t_route
        prompt_depth_entropy = (
            float(np.mean(prompt_outputs["stats"]["depth_entropies"]))
            if prompt_outputs["stats"]["depth_entropies"]
            else 0.0
        )
        prompt_support_size = (
            float(np.mean(prompt_outputs["stats"]["depth_support_sizes"]))
            if prompt_outputs["stats"]["depth_support_sizes"]
            else 0.0
        )
        feature_rows.append(
            {
                "sequence_idx": seq_idx,
                "split": split_name,
                "document_idx": document_idx,
                "document_title": document_title,
                "window_idx": window_idx,
                "prompt_depth_entropy": prompt_depth_entropy,
                "prompt_support_size": prompt_support_size,
                "prompt_scores_json": json.dumps(prompt_scores.tolist()),
                "prompt_scores_attn_json": json.dumps(prompt_scores_attn.tolist()),
                "prompt_scores_mlp_json": json.dumps(prompt_scores_mlp.tolist()),
                "prompt_chunk_utilities_json": json.dumps(prompt_summary.chunk_utilities.tolist()),
            }
        )

        sequence_cache.append(
            {
                "sequence_idx": seq_idx,
                "split": split_name,
                "document_idx": document_idx,
                "document_title": document_title,
                "window_idx": window_idx,
                "continuation_ids": continuation_ids,
                "past_key_values": past_key_values,
                "prompt_last_logits": prompt_last_logits,
                "prefill_seconds": prefill_seconds,
                "route_seconds": route_seconds,
                "prompt_scores": prompt_scores,
                "prompt_scores_attn": prompt_scores_attn,
                "prompt_scores_mlp": prompt_scores_mlp,
                "prompt_chunk_utilities": prompt_summary.chunk_utilities[:, 1:-1],
                "stability_spearman": mean_pairwise_chunk_spearman(prompt_summary.chunk_utilities[:, 1:-1]),
                "stability_top3_jaccard": mean_pairwise_chunk_jaccard(prompt_summary.chunk_utilities[:, 1:-1], k=3),
                "prompt_depth_entropy": prompt_depth_entropy,
                "prompt_support_size": prompt_support_size,
            }
        )
        if (local_idx + 1) % 8 == 0 or (local_idx + 1) == target_sequences:
            print(
                f"[prefill] cached {local_idx + 1}/{target_sequences} sequences from split={split_name}",
                flush=True,
            )

    feature_df = pd.DataFrame(feature_rows)

    if not sequence_cache:
        raise RuntimeError("No valid sequences found for oracle evaluation.")

    if args.prompt_features_only:
        oracle_feature_rows = []
        for row in sequence_cache:
            prompt_scores = np.asarray(row["prompt_scores"], dtype=np.float64)
            for skip_count in skip_counts:
                oracle_feature_rows.append(
                    {
                        "sequence_idx": row["sequence_idx"],
                        "split": row["split"],
                        "document_idx": row["document_idx"],
                        "document_title": row["document_title"],
                        "window_idx": row["window_idx"],
                        "skip_count": skip_count,
                        "stability_spearman": row["stability_spearman"],
                        "stability_top3_jaccard": row["stability_top3_jaccard"],
                        "prompt_margin": prompt_margin(prompt_scores, skip_count),
                        "prompt_depth_entropy": row["prompt_depth_entropy"],
                        "prompt_support_size": row["prompt_support_size"],
                    }
                )
        oracle_df = pd.DataFrame(oracle_feature_rows)
        oracle_summary = bootstrap_metric_table(
            oracle_df,
            group_cols=["skip_count"],
            metric_cols=[
                "stability_spearman",
                "stability_top3_jaccard",
                "prompt_margin",
                "prompt_depth_entropy",
                "prompt_support_size",
            ],
            seed=args.seed,
        )
        feature_df.to_csv(oracles_dir / f"{tag_prefix}sequence_features.csv", index=False)
        oracle_df.to_csv(oracles_dir / f"{tag_prefix}oracle_mask_alignment.csv", index=False)
        oracle_summary.to_csv(oracles_dir / f"{tag_prefix}oracle_mask_alignment_summary.csv", index=False)
        save_yaml(
            routing_dir / f"{tag_prefix}routing_eval_config.yaml",
            {
                "checkpoint": str(args.checkpoint),
                "checkpoint_step": checkpoint_step,
                "num_sequences": args.num_sequences,
                "prompt_len": args.prompt_len,
                "decode_len": args.decode_len,
                "split": args.split,
                "sequence_offset": args.sequence_offset,
                "manifest_path": args.manifest_path,
                "num_shards": args.num_shards,
                "shard_index": args.shard_index,
                "skip_counts": skip_counts,
                "score_mode": args.score_mode,
                "prompt_features_only": True,
            },
        )
        print(
            f"[prompt-features-only] wrote {len(feature_df)} feature rows and {len(oracle_df)} prompt-alignment rows",
            flush=True,
        )
        return

    mask_eval_table: dict[str, dict[str, Any]] = {}
    mask_items = list(all_masks.items())
    eval_start = time.perf_counter()
    for mask_pos, (mask_id, mask) in enumerate(mask_items, start=1):
        losses_by_sequence, decode_seconds, decode_tok_s = evaluate_mask_over_cache(
            model,
            sequence_cache=sequence_cache,
            mask=mask,
            batch_size=args.batch_size,
            device=device,
        )
        mask_eval_table[mask_id] = {
            "mask": mask,
            "losses_by_sequence": losses_by_sequence,
            "decode_seconds_per_sequence": decode_seconds,
            "decode_tokens_per_sec": decode_tok_s,
        }
        elapsed = time.perf_counter() - eval_start
        avg_per_mask = elapsed / mask_pos
        eta_seconds = avg_per_mask * (len(mask_items) - mask_pos)
        print(
            f"[masks] {mask_pos}/{len(mask_items)} {mask_id} "
            f"elapsed={elapsed/60.0:.1f}m eta={eta_seconds/60.0:.1f}m",
            flush=True,
        )

    for row in sequence_cache:
        seq_idx = int(row["sequence_idx"])
        full_mask_id = mask_to_id(all_on)
        full_loss = float(mask_eval_table[full_mask_id]["losses_by_sequence"][seq_idx])
        row["full_loss"] = full_loss
        row["full_decode_seconds"] = float(mask_eval_table[full_mask_id]["decode_seconds_per_sequence"])
        row["full_decode_tokens_per_sec"] = float(mask_eval_table[full_mask_id]["decode_tokens_per_sec"])

        leave_one_out = np.zeros(middle_count, dtype=np.float64)
        for block_idx in range(middle_count):
            mask = all_on.copy()
            mask[block_idx] = False
            mask_id = mask_to_id(mask)
            loss = float(mask_eval_table[mask_id]["losses_by_sequence"][seq_idx])
            delta = loss - full_loss
            leave_one_out[block_idx] = delta
            mask_rows.append(
                {
                    "sequence_idx": seq_idx,
                    "split": row["split"],
                    "document_idx": row["document_idx"],
                    "window_idx": row["window_idx"],
                    "skip_count": 1,
                    "mask_id": mask_id,
                    "method": "leave_one_out",
                    "continuation_loss": loss,
                    "delta_loss": delta,
                    "block_idx": block_idx + 1,
                }
            )

        alignment = summarize_alignment(row["prompt_scores"][1:-1], leave_one_out)
        loo_rows.append(
            {
                "sequence_idx": seq_idx,
                "split": row["split"],
                "document_idx": row["document_idx"],
                "window_idx": row["window_idx"],
                **alignment,
            }
        )

        budget_losses: dict[int, dict[str, float]] = {}
        for skip_count, masks in masks_by_budget.items():
            losses_for_budget: dict[str, float] = {}
            for mask in masks:
                mask_id = mask_to_id(mask)
                loss = float(mask_eval_table[mask_id]["losses_by_sequence"][seq_idx])
                losses_for_budget[mask_id] = loss
                mask_rows.append(
                    {
                        "sequence_idx": seq_idx,
                        "split": row["split"],
                        "document_idx": row["document_idx"],
                        "window_idx": row["window_idx"],
                        "skip_count": skip_count,
                        "mask_id": mask_id,
                        "method": "exhaustive_mask",
                        "continuation_loss": loss,
                        "delta_loss": loss - full_loss,
                        "block_idx": "",
                    }
                )
            budget_losses[skip_count] = losses_for_budget
        row["budget_losses"] = budget_losses

    mask_df = pd.DataFrame(mask_rows)
    loo_df = pd.DataFrame(loo_rows)
    feature_df = pd.DataFrame(feature_rows)
    loo_summary = bootstrap_metric_table(
        loo_df.assign(split="all"),
        group_cols=["split"],
        metric_cols=["spearman", "kendall", "topk_jaccard_1", "recall_at_3", "ndcg_at_3"],
        seed=args.seed,
    ).drop(columns=["split"])

    exhaustive_df = mask_df[mask_df["method"] == "exhaustive_mask"].copy()
    global_best_rows = (
        exhaustive_df.groupby(["skip_count", "mask_id"], as_index=False)["continuation_loss"].mean().sort_values("continuation_loss")
    )
    global_best_masks: dict[int, str] = {}
    for skip_count in skip_counts:
        subset = global_best_rows[global_best_rows["skip_count"] == skip_count]
        global_best_masks[skip_count] = str(subset.iloc[0]["mask_id"])

    sequence_scores = [np.asarray(row["prompt_scores"]) for row in sequence_cache]
    mismatched_idx = np.roll(np.arange(len(sequence_cache)), -1)
    routing_rows: list[dict[str, Any]] = []
    oracle_summary_rows: list[dict[str, Any]] = []

    for cache_pos, row in enumerate(sequence_cache):
        seq_idx = int(row["sequence_idx"])
        prompt_scores = np.asarray(row["prompt_scores"])
        mismatched_scores = sequence_scores[int(mismatched_idx[cache_pos])]

        for skip_count in skip_counts:
            skip_fraction = skip_count / num_blocks
            prompt_mask = select_prompt_fixed_route(prompt_scores, num_blocks, skip_fraction)
            mismatched_mask = select_prompt_fixed_route(mismatched_scores, num_blocks, skip_fraction)
            balanced_mask = balanced_skip_route(num_blocks, skip_fraction)
            random_mask = random_skip_route(num_blocks, skip_fraction, rng)

            budget_losses = row["budget_losses"][skip_count]
            oracle_mask_id = min(budget_losses.items(), key=lambda item: item[1])[0]
            global_mask_id = global_best_masks[skip_count]
            id_to_mask = {mask_to_id(mask): mask for mask in masks_by_budget[skip_count]}

            method_to_mask = {
                "no_skip": all_on,
                "random": random_mask,
                "balanced": balanced_mask,
                "mismatched_prompt": mismatched_mask,
                "global_static": id_to_mask[global_mask_id],
                "oracle_sequence": id_to_mask[oracle_mask_id],
                "prompt_fixed": prompt_mask,
            }

            prompt_mask_id = mask_to_id(prompt_mask)
            oracle_mask = id_to_mask[oracle_mask_id]
            oracle_summary_rows.append(
                {
                    "sequence_idx": seq_idx,
                    "split": row["split"],
                    "document_idx": row["document_idx"],
                    "window_idx": row["window_idx"],
                    "skip_count": skip_count,
                    "oracle_exact_match": float(prompt_mask_id == oracle_mask_id),
                    "oracle_mask_jaccard": float((prompt_mask & oracle_mask).sum() / max((prompt_mask | oracle_mask).sum(), 1)),
                    "delta_to_oracle": float(budget_losses[prompt_mask_id] - budget_losses[oracle_mask_id]),
                    "delta_to_global_static": float(budget_losses[prompt_mask_id] - budget_losses[global_mask_id]),
                    "stability_spearman": row["stability_spearman"],
                    "stability_top3_jaccard": row["stability_top3_jaccard"],
                    "prompt_margin": prompt_margin(prompt_scores, skip_count),
                    "prompt_depth_entropy": row["prompt_depth_entropy"],
                    "prompt_support_size": row["prompt_support_size"],
                }
            )

            for method, mask in method_to_mask.items():
                mask_id = mask_to_id(mask)
                timing_info = mask_eval_table[mask_id]
                continuation_loss = float(timing_info["losses_by_sequence"][seq_idx])
                decode_seconds = float(timing_info["decode_seconds_per_sequence"])
                decode_tok_s = float(timing_info["decode_tokens_per_sec"])
                routing_seconds = row["route_seconds"] if method in {"prompt_fixed", "mismatched_prompt"} else 0.0

                routing_rows.append(
                    {
                        "sequence_idx": seq_idx,
                        "split": row["split"],
                        "document_idx": row["document_idx"],
                        "window_idx": row["window_idx"],
                        "skip_count": skip_count,
                        "method": method,
                        "continuation_loss": continuation_loss,
                        "prefill_seconds": row["prefill_seconds"],
                        "decode_seconds": decode_seconds,
                        "decode_tokens_per_sec": decode_tok_s,
                        "routing_overhead_seconds": routing_seconds,
                        "end_to_end_seconds": row["prefill_seconds"] + decode_seconds + routing_seconds,
                        "active_blocks": int(mask.sum()),
                    }
                )

    oracle_df = pd.DataFrame(oracle_summary_rows)
    routing_df = pd.DataFrame(routing_rows)
    oracle_summary = bootstrap_metric_table(
        oracle_df,
        group_cols=["skip_count"],
        metric_cols=[
            "oracle_exact_match",
            "oracle_mask_jaccard",
            "delta_to_oracle",
            "delta_to_global_static",
            "stability_spearman",
            "stability_top3_jaccard",
            "prompt_margin",
        ],
        seed=args.seed,
    )
    routing_summary = bootstrap_metric_table(
        routing_df,
        group_cols=["skip_count", "method"],
        metric_cols=[
            "continuation_loss",
            "decode_tokens_per_sec",
            "prefill_seconds",
            "decode_seconds",
            "routing_overhead_seconds",
            "end_to_end_seconds",
            "active_blocks",
        ],
        seed=args.seed,
    )

    loo_df.to_csv(oracles_dir / f"{tag_prefix}leave_one_out_alignment.csv", index=False)
    loo_summary.to_csv(oracles_dir / f"{tag_prefix}leave_one_out_alignment_summary.csv", index=False)
    feature_df.to_csv(oracles_dir / f"{tag_prefix}sequence_features.csv", index=False)
    mask_df.to_csv(oracles_dir / f"{tag_prefix}exhaustive_mask_losses.csv", index=False)
    oracle_df.to_csv(oracles_dir / f"{tag_prefix}oracle_mask_alignment.csv", index=False)
    oracle_summary.to_csv(oracles_dir / f"{tag_prefix}oracle_mask_alignment_summary.csv", index=False)
    routing_df.to_csv(routing_dir / f"{tag_prefix}routing_eval_per_sequence.csv", index=False)
    routing_summary.to_csv(routing_dir / f"{tag_prefix}routing_eval_summary.csv", index=False)
    save_yaml(
        routing_dir / f"{tag_prefix}routing_eval_config.yaml",
        {
            "checkpoint": str(args.checkpoint),
            "checkpoint_step": checkpoint_step,
            "num_sequences": args.num_sequences,
            "prompt_len": args.prompt_len,
            "decode_len": args.decode_len,
            "split": args.split,
            "sequence_offset": args.sequence_offset,
            "manifest_path": args.manifest_path,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "skip_counts": skip_counts,
            "score_mode": args.score_mode,
        },
    )

    if not args.skip_plots:
        plt.figure(figsize=(5, 4))
        plt.hist(loo_df["spearman"], bins=20)
        plt.xlabel("prompt-vs-LOO spearman")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(plot_dir / f"{tag_prefix}oracles_prompt_vs_loo_spearman_hist.png", dpi=160)
        plt.close()

        plt.figure(figsize=(6, 4))
        for skip_count in skip_counts:
            subset = routing_summary[
                (routing_summary["skip_count"] == skip_count) & (routing_summary["metric"] == "continuation_loss")
            ]
            x = subset["method"].tolist()
            y = subset["mean"].to_numpy()
            plt.plot(x, y, marker="o", label=f"skip={skip_count}")
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("continuation loss")
        plt.tight_layout()
        plt.legend()
        plt.savefig(plot_dir / f"{tag_prefix}routing_continuation_loss_by_budget.png", dpi=160)
        plt.close()

        plt.figure(figsize=(6, 4))
        for skip_count in skip_counts:
            subset = routing_summary[
                (routing_summary["skip_count"] == skip_count) & (routing_summary["metric"] == "decode_tokens_per_sec")
            ]
            x = subset["method"].tolist()
            y = subset["mean"].to_numpy()
            plt.plot(x, y, marker="o", label=f"skip={skip_count}")
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("decode tokens/s")
        plt.tight_layout()
        plt.legend()
        plt.savefig(plot_dir / f"{tag_prefix}routing_decode_toks_by_budget.png", dpi=160)
        plt.close()

    if not args.skip_docs:
        write_functional_oracles_doc(ROOT / "docs" / f"functional_oracles_{args.tag}.md", checkpoint_step, loo_summary, oracle_summary)
        write_routing_doc(ROOT / "docs" / f"routing_baselines_{args.tag}.md", routing_summary)


if __name__ == "__main__":
    main()
