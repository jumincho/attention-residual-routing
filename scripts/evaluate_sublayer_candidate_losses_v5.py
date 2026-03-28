#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attnres_routing.model import AttnResConfig, DecoderLM  # noqa: E402
from attnres_routing.routing import (  # noqa: E402
    continuation_losses_from_decode_logits,
    stack_past_key_values,
    teacher_forced_decode_from_past,
)
from attnres_routing.sequence_manifest import load_manifest_jsonl  # noqa: E402
from attnres_routing.sublayer_masks import (  # noqa: E402
    SublayerMask,
    edit_distance,
    enumerate_local_edits,
    estimated_decode_cost,
    estimated_reduction_ratio,
    from_block_mask,
)


def load_checkpoint(path: Path, device: torch.device, precision: str) -> DecoderLM:
    payload = torch.load(path, map_location=device)
    config = payload["config"]
    model = DecoderLM(AttnResConfig.from_dict(config["model"])).to(device)
    model.load_state_dict(payload["model_state"])
    if precision == "fp16" and device.type == "cuda":
        model = model.half()
    model.eval()
    return model


def parse_block_mask_id(mask_id: str, num_blocks: int) -> np.ndarray:
    mask = np.zeros(num_blocks, dtype=np.bool_)
    _, kept = mask_id.split(":", 1)
    if kept.strip():
        for token in kept.split(","):
            mask[int(token) - 1] = True
    return mask


def load_anchor_masks(bank_csv: Path, bank_size: int, skip_counts: list[int], num_blocks: int) -> dict[int, SublayerMask]:
    bank_df = pd.read_csv(bank_csv)
    anchors: dict[int, SublayerMask] = {}
    for skip_count in skip_counts:
        subset = bank_df[(bank_df["skip_count"] == skip_count) & (bank_df["bank_size"] == bank_size)].copy()
        if subset.empty:
            raise ValueError(f"No bank rows for skip_count={skip_count} bank_size={bank_size} in {bank_csv}")
        row = subset[subset["reasons"].str.contains("calib_global_static")].iloc[0]
        block_mask = parse_block_mask_id(str(row["mask_id"]), num_blocks)
        anchors[skip_count] = from_block_mask(block_mask.tolist())
    return anchors


def load_decode_costs(latency_csv: Path, prompt_len: int, decode_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(latency_csv)
    subset = df[
        (df["phase"] == "decode")
        & (df["prompt_len"] == prompt_len)
        & (df["decode_len"] == decode_len)
    ].copy()
    if subset.empty:
        raise ValueError(f"No decode latency rows for prompt_len={prompt_len}, decode_len={decode_len} in {latency_csv}")
    subset = subset.sort_values(["block_idx", "component"]).reset_index(drop=True)
    full = subset[subset["component"] == "full"].sort_values("block_idx")["median_seconds"].to_numpy(dtype=np.float64)
    attn = subset[subset["component"] == "attn_only"].sort_values("block_idx")["median_seconds"].to_numpy(dtype=np.float64)
    mlp = subset[subset["component"] == "mlp_only"].sort_values("block_idx")["median_seconds"].to_numpy(dtype=np.float64)
    return full, attn, mlp


def build_candidate_defs(
    anchors: dict[int, SublayerMask],
    full_cost: np.ndarray,
    attn_cost: np.ndarray,
    mlp_cost: np.ndarray,
    max_edits: int,
    min_reduction: float,
    max_reduction: float,
) -> pd.DataFrame:
    candidate_rows: dict[str, dict[str, Any]] = {}
    middle_blocks = list(range(len(full_cost) - 1))
    for anchor_skip_count, anchor in anchors.items():
        for candidate in enumerate_local_edits(anchor, middle_blocks, max_edits=max_edits):
            attn_mask, mlp_mask = candidate.to_arrays()
            attn_mask[-1] = True
            mlp_mask[-1] = True
            candidate = SublayerMask(attn_mask=tuple(attn_mask.tolist()), mlp_mask=tuple(mlp_mask.tolist()))
            reduction = estimated_reduction_ratio(candidate, full_cost, attn_cost, mlp_cost)
            if candidate.to_id() != from_block_mask(np.ones_like(full_cost, dtype=np.bool_)).to_id():
                if reduction < min_reduction or reduction > max_reduction:
                    continue
            candidate_cost = estimated_decode_cost(candidate, full_cost, attn_cost, mlp_cost)
            row = candidate_rows.setdefault(
                candidate.to_id(),
                {
                    "candidate_id": candidate.to_id(),
                    "attn_mask_json": json.dumps(attn_mask.astype(int).tolist()),
                    "mlp_mask_json": json.dumps(mlp_mask.astype(int).tolist()),
                    "estimated_decode_seconds": float(candidate_cost),
                    "estimated_reduction_ratio": float(reduction),
                    "anchor_skip_counts": [],
                    "min_anchor_edit_distance": None,
                    "attn_skip_count": int((~attn_mask[:-1]).sum()),
                    "mlp_skip_count": int((~mlp_mask[:-1]).sum()),
                    "whole_block_skip_count": int((~attn_mask[:-1] & ~mlp_mask[:-1]).sum()),
                },
            )
            row["anchor_skip_counts"].append(anchor_skip_count)
            candidate_edit_distance = edit_distance(candidate, anchor)
            current = row["min_anchor_edit_distance"]
            row["min_anchor_edit_distance"] = candidate_edit_distance if current is None else min(current, candidate_edit_distance)
    rows = []
    for row in candidate_rows.values():
        row["anchor_skip_counts"] = ",".join(str(value) for value in sorted(set(row["anchor_skip_counts"])))
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["estimated_reduction_ratio", "candidate_id"]).reset_index(drop=True)
    return out


def batch_rows(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [rows[start : start + batch_size] for start in range(0, len(rows), batch_size)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--manifest-path", type=str, required=True)
    parser.add_argument("--anchor-bank-csv", type=str, required=True)
    parser.add_argument("--latency-csv", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--prompt-len", type=int, required=True)
    parser.add_argument("--decode-len", type=int, required=True)
    parser.add_argument("--anchor-bank-size", type=int, default=32)
    parser.add_argument("--anchor-skip-counts", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--max-edits", type=int, default=2)
    parser.add_argument("--min-reduction", type=float, default=0.02)
    parser.add_argument("--max-reduction", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16"], default="fp16")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_checkpoint(Path(args.checkpoint), device=device, precision=args.precision)
    manifest_rows = load_manifest_jsonl(Path(args.manifest_path))
    if args.num_shards <= 0 or not (0 <= args.shard_index < args.num_shards):
        raise ValueError("Invalid shard settings")
    manifest_rows = [row for idx, row in enumerate(manifest_rows) if (idx % args.num_shards) == args.shard_index]

    num_blocks = model.config.num_blocks
    full_cost, attn_cost, mlp_cost = load_decode_costs(Path(args.latency_csv), args.prompt_len, args.decode_len)
    if len(full_cost) != num_blocks:
        raise ValueError(f"Latency CSV num_blocks mismatch: expected {num_blocks}, got {len(full_cost)}")
    anchors = load_anchor_masks(Path(args.anchor_bank_csv), args.anchor_bank_size, args.anchor_skip_counts, num_blocks)
    candidate_defs = build_candidate_defs(
        anchors=anchors,
        full_cost=full_cost,
        attn_cost=attn_cost,
        mlp_cost=mlp_cost,
        max_edits=args.max_edits,
        min_reduction=args.min_reduction,
        max_reduction=args.max_reduction,
    )

    out_dir = ROOT / "results" / "latency_budgeted_sublayer_v5"
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_suffix = f"_s{args.shard_index:02d}" if args.num_shards > 1 else ""
    candidate_defs.to_csv(out_dir / f"{args.tag}{shard_suffix}_candidate_defs.csv", index=False)

    sequence_cache: list[dict[str, Any]] = []
    for local_idx, row in enumerate(manifest_rows):
        tokens = torch.tensor(row["input_ids"], dtype=torch.long)
        if tokens.numel() < args.prompt_len + args.decode_len:
            continue
        prompt_ids = tokens[: args.prompt_len].unsqueeze(0).to(device)
        continuation_ids = tokens[args.prompt_len : args.prompt_len + args.decode_len].unsqueeze(0).to(device)
        with torch.no_grad():
            prompt_outputs = model(
                input_ids=prompt_ids,
                labels=None,
                use_cache=True,
                record_mode="none",
            )
        sequence_cache.append(
            {
                "sequence_idx": int(row["sequence_idx"]),
                "split": str(row.get("source_split", "")),
                "document_idx": int(row.get("document_idx", -1)),
                "document_title": str(row.get("document_title", "")),
                "window_idx": int(row.get("window_idx", 0)),
                "continuation_ids": continuation_ids,
                "past_key_values": prompt_outputs["past_key_values"],
                "prompt_last_logits": prompt_outputs["logits"][:, -1:, :],
            }
        )
        if (local_idx + 1) % 32 == 0 or (local_idx + 1) == len(manifest_rows):
            print(f"[sublayer-cache] {local_idx + 1}/{len(manifest_rows)} cached", flush=True)

    loss_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for cand_idx, cand_row in enumerate(candidate_defs.to_dict(orient="records"), start=1):
        attn_mask = torch.tensor(json.loads(cand_row["attn_mask_json"]), dtype=torch.bool, device=device)
        mlp_mask = torch.tensor(json.loads(cand_row["mlp_mask_json"]), dtype=torch.bool, device=device)
        total_elapsed = 0.0
        total_tokens = 0
        candidate_losses: list[float] = []
        for rows in batch_rows(sequence_cache, args.batch_size):
            continuation_ids = torch.cat([row["continuation_ids"] for row in rows], dim=0)
            prompt_last_logits = torch.cat([row["prompt_last_logits"] for row in rows], dim=0)
            past_key_values = stack_past_key_values([row["past_key_values"] for row in rows])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            decode_logits = teacher_forced_decode_from_past(
                model,
                continuation_ids=continuation_ids,
                past_key_values=past_key_values,
                prompt_last_logits=prompt_last_logits,
                active_attn_block_mask=attn_mask,
                active_mlp_block_mask=mlp_mask,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - t0
            total_elapsed += elapsed
            total_tokens += continuation_ids.size(0) * max(continuation_ids.size(1) - 1, 0)
            losses = continuation_losses_from_decode_logits(decode_logits, continuation_ids).detach().cpu().numpy()
            for row, loss in zip(rows, losses.tolist()):
                loss_value = float(loss)
                candidate_losses.append(loss_value)
                loss_rows.append(
                    {
                        "tag": args.tag,
                        "candidate_id": cand_row["candidate_id"],
                        "sequence_idx": int(row["sequence_idx"]),
                        "split": row["split"],
                        "document_idx": int(row["document_idx"]),
                        "document_title": row["document_title"],
                        "window_idx": int(row["window_idx"]),
                        "continuation_loss": loss_value,
                        "estimated_decode_seconds": float(cand_row["estimated_decode_seconds"]),
                        "estimated_reduction_ratio": float(cand_row["estimated_reduction_ratio"]),
                        "anchor_skip_counts": cand_row["anchor_skip_counts"],
                        "min_anchor_edit_distance": int(cand_row["min_anchor_edit_distance"]),
                        "attn_skip_count": int(cand_row["attn_skip_count"]),
                        "mlp_skip_count": int(cand_row["mlp_skip_count"]),
                        "whole_block_skip_count": int(cand_row["whole_block_skip_count"]),
                    }
                )
        summary_rows.append(
            {
                "tag": args.tag,
                "candidate_id": cand_row["candidate_id"],
                "estimated_decode_seconds": float(cand_row["estimated_decode_seconds"]),
                "estimated_reduction_ratio": float(cand_row["estimated_reduction_ratio"]),
                "anchor_skip_counts": cand_row["anchor_skip_counts"],
                "min_anchor_edit_distance": int(cand_row["min_anchor_edit_distance"]),
                "attn_skip_count": int(cand_row["attn_skip_count"]),
                "mlp_skip_count": int(cand_row["mlp_skip_count"]),
                "whole_block_skip_count": int(cand_row["whole_block_skip_count"]),
                "mean_continuation_loss": float(np.mean(candidate_losses)),
                "decode_seconds_per_sequence": total_elapsed / max(len(sequence_cache), 1),
                "decode_tokens_per_sec": total_tokens / max(total_elapsed, 1e-6),
            }
        )
        print(
            f"[sublayer-eval] {cand_idx}/{len(candidate_defs)} {cand_row['candidate_id']} "
            f"mean_loss={np.mean(candidate_losses):.6f} est_reduction={cand_row['estimated_reduction_ratio']:.4f}",
            flush=True,
        )

    pd.DataFrame(loss_rows).to_csv(out_dir / f"{args.tag}{shard_suffix}_candidate_losses.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out_dir / f"{args.tag}{shard_suffix}_candidate_summary.csv", index=False)


if __name__ == "__main__":
    main()
