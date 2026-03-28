#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attnres_routing.data import DataConfig, prepare_lm_dataset_splits  # noqa: E402
from attnres_routing.model import AttnResConfig, DecoderLM  # noqa: E402
from attnres_routing.sequence_manifest import load_manifest_jsonl  # noqa: E402


def load_checkpoint(path: Path, device: torch.device, precision: str) -> tuple[DecoderLM, dict[str, Any]]:
    payload = torch.load(path, map_location=device)
    config = payload["config"]
    model = DecoderLM(AttnResConfig.from_dict(config["model"])).to(device)
    model.load_state_dict(payload["model_state"])
    if precision == "fp16" and device.type == "cuda":
        model = model.half()
    model.eval()
    return model, config


def summarize_state(hidden: torch.Tensor) -> tuple[np.ndarray, np.ndarray, float, float]:
    mean_vec = hidden.mean(dim=1).squeeze(0).float().cpu().numpy()
    final_vec = hidden[:, -1, :].squeeze(0).float().cpu().numpy()
    token_norms = hidden.float().norm(dim=-1)
    return (
        mean_vec,
        final_vec,
        float(token_norms.mean().item()),
        float(token_norms.std(unbiased=False).item()),
    )


def prompt_difficulty_stats(prompt_ids: torch.Tensor, logits: torch.Tensor) -> dict[str, float]:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = prompt_ids[:, 1:].contiguous()
    token_losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view(shift_labels.size(0), shift_labels.size(1))
    losses = token_losses.squeeze(0).float()
    prompt_tokens = prompt_ids.squeeze(0)
    unique_ratio = float(prompt_tokens.unique().numel() / max(prompt_tokens.numel(), 1))
    adjacent_repeats = (prompt_tokens[1:] == prompt_tokens[:-1]).float() if prompt_tokens.numel() > 1 else torch.zeros(0)
    max_run = 1
    current_run = 1
    for idx in range(1, prompt_tokens.numel()):
        if int(prompt_tokens[idx].item()) == int(prompt_tokens[idx - 1].item()):
            current_run += 1
        else:
            max_run = max(max_run, current_run)
            current_run = 1
    max_run = max(max_run, current_run)
    return {
        "prompt_surprisal_mean": float(losses.mean().item()),
        "prompt_surprisal_std": float(losses.std(unbiased=False).item()),
        "prompt_surprisal_max": float(losses.max().item()),
        "prompt_ppl": float(torch.exp(losses.mean().clamp(max=20.0)).item()),
        "unique_token_ratio": unique_ratio,
        "adjacent_repeat_fraction": float(adjacent_repeats.mean().item()) if adjacent_repeats.numel() > 0 else 0.0,
        "max_repeat_run_fraction": float(max_run / max(prompt_tokens.numel(), 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num-sequences", type=int, required=True)
    parser.add_argument("--sequence-offset", type=int, default=0)
    parser.add_argument("--prompt-len", type=int, default=256)
    parser.add_argument("--decode-len", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--summary-blocks", type=int, nargs="*", default=None)
    parser.add_argument("--manifest-path", type=str, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, config = load_checkpoint(Path(args.checkpoint), device, args.precision)
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
        target_ds = split_datasets[args.split]

    feature_dir = ROOT / "results" / "rich_features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    num_blocks = model.config.num_blocks
    default_blocks = [0, num_blocks // 2, num_blocks]
    summary_blocks = sorted({int(idx) for idx in (args.summary_blocks or default_blocks) if 0 <= int(idx) <= num_blocks})

    rows = []
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
            window_idx = int(source_row.get("window_idx", 0))
            tokens = torch.tensor(source_row["input_ids"], dtype=torch.long)
        else:
            seq_idx = args.sequence_offset + local_idx
            split_name = args.split
            document_idx = -1
            window_idx = 0
            tokens = target_ds[seq_idx]["input_ids"]
        if tokens.numel() < args.prompt_len + args.decode_len:
            continue
        prompt_ids = tokens[: args.prompt_len].unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(
                prompt_ids,
                labels=None,
                use_cache=False,
                record_mode="none",
                return_block_states=True,
            )
        block_states = outputs["block_states"]
        final_hidden = outputs["final_hidden"]
        row: dict[str, Any] = {
            "sequence_idx": seq_idx,
            "split": split_name,
            "document_idx": document_idx,
            "window_idx": window_idx,
            "summary_blocks_json": json.dumps(summary_blocks),
        }
        row.update(prompt_difficulty_stats(prompt_ids, outputs["logits"]))
        for block_idx in summary_blocks:
            if block_idx == num_blocks:
                hidden = final_hidden
                prefix = "final"
            else:
                hidden = block_states[block_idx]
                prefix = f"block_{block_idx}"
            mean_vec, final_vec, norm_mean, norm_std = summarize_state(hidden)
            row[f"{prefix}_mean_json"] = json.dumps(mean_vec.tolist())
            row[f"{prefix}_final_json"] = json.dumps(final_vec.tolist())
            row[f"{prefix}_norm_mean"] = norm_mean
            row[f"{prefix}_norm_std"] = norm_std
        rows.append(row)
        if (local_idx + 1) % 32 == 0 or (local_idx + 1) == target_sequences:
            print(
                f"[hidden] processed {local_idx + 1}/{target_sequences} sequences from split={split_name}",
                flush=True,
            )

    pd.DataFrame(rows).to_csv(feature_dir / f"{args.tag}_hidden_prompt_features.csv", index=False)


if __name__ == "__main__":
    main()
