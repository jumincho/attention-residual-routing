#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attnres_routing.analysis import (
    compute_chunk_utility,
    compute_utility_from_records,
    save_json,
    summarize_prompt_decode_transfer,
    write_depth_support_outputs,
)
from attnres_routing.data import DataConfig, prepare_lm_datasets
from attnres_routing.model import AttnResConfig, DecoderLM
from attnres_routing.utils import ensure_dir, load_yaml


def load_checkpoint(path: Path, device: torch.device) -> tuple[DecoderLM, dict]:
    payload = torch.load(path, map_location=device)
    config = payload["config"]
    model = DecoderLM(AttnResConfig.from_dict(config["model"])).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-sequences", type=int, default=64)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--decode-len", type=int, default=128)
    parser.add_argument("--num-chunks", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda")
    model, config = load_checkpoint(Path(args.checkpoint), device)
    data_cfg = DataConfig(**config["data"])
    _, val_ds, _ = prepare_lm_datasets(data_cfg)
    out_dir = ensure_dir(ROOT / "results" / "depth_support")
    plot_dir = ensure_dir(ROOT / "results" / "plots")
    num_sources = model.config.num_blocks + 1

    rows = []
    raw_vectors = []

    for seq_idx in range(min(args.num_sequences, len(val_ds))):
        tokens = val_ds[seq_idx]["input_ids"]
        if tokens.numel() < args.prompt_len + args.decode_len:
            continue
        sample = tokens[: args.prompt_len + args.decode_len].unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(sample, labels=None, record_mode="full")
        records = outputs["stats"]["records"]
        prompt_summary = compute_chunk_utility(records, num_sources, prompt_len=args.prompt_len, num_chunks=args.num_chunks)
        decode_utility = compute_utility_from_records(
            records,
            num_sources=num_sources,
            token_start=args.prompt_len,
            token_end=args.prompt_len + args.decode_len,
        )
        prompt_dist = softmax(prompt_summary.full_utility[1:])
        transfer = summarize_prompt_decode_transfer(prompt_summary.full_utility, decode_utility)
        row = {
            "sequence_idx": seq_idx,
            "prompt_entropy": float(-(prompt_dist * np.log(prompt_dist + 1e-8)).sum()),
            "prompt_decode_corr_proxy": float(np.dot(prompt_summary.full_utility, decode_utility)),
            "chunk_variance_mean": float(prompt_summary.chunk_variance[1:].mean()),
            **transfer,
        }
        rows.append(row)
        raw_vectors.append(
            {
                "sequence_idx": seq_idx,
                "prompt_utility": prompt_summary.full_utility.tolist(),
                "decode_utility": decode_utility.tolist(),
                "chunk_utilities": prompt_summary.chunk_utilities.tolist(),
                "chunk_variance": prompt_summary.chunk_variance.tolist(),
                "topk_frequency": prompt_summary.topk_frequency.tolist(),
            }
        )

    df = pd.DataFrame(rows)
    depth_csv = out_dir / "depth_support_metrics.csv"
    write_depth_support_outputs(rows, depth_csv, plot_dir / "depth_support")
    save_json(out_dir / "depth_support_vectors.json", raw_vectors)

    summary_md = ROOT / "docs" / "depth_support_summary.md"
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# Depth Support Summary\n\n")
        if len(df) == 0:
            f.write("No valid sequences were analyzed.\n")
        else:
            f.write(f"- sequences: {len(df)}\n")
            f.write(f"- mean spearman: {df['spearman'].mean():.4f}\n")
            f.write(f"- mean kendall: {df['kendall'].mean():.4f}\n")
            f.write(f"- mean top-1 jaccard: {df['topk_jaccard_1'].mean():.4f}\n")
            f.write(f"- mean recall@3: {df['recall_at_3'].mean():.4f}\n")
            f.write(f"- mean chunk variance: {df['chunk_variance_mean'].mean():.4f}\n")


if __name__ == "__main__":
    main()
