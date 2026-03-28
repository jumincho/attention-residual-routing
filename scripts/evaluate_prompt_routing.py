#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attnres_routing.analysis import compute_chunk_utility, compute_utility_from_records
from attnres_routing.data import DataConfig, prepare_lm_datasets
from attnres_routing.model import AttnResConfig, DecoderLM
from attnres_routing.routing import (
    balanced_skip_route,
    compute_routing_scores,
    continuation_loss,
    random_skip_route,
    select_prompt_fixed_route,
    teacher_forced_decode_timing,
)
from attnres_routing.utils import ensure_dir


def load_checkpoint(path: Path, device: torch.device):
    payload = torch.load(path, map_location=device)
    config = payload["config"]
    model = DecoderLM(AttnResConfig.from_dict(config["model"])).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-sequences", type=int, default=32)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--decode-len", type=int, default=128)
    parser.add_argument("--skip-fractions", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    parser.add_argument("--score-mode", type=str, default="utility_over_variance")
    parser.add_argument("--num-chunks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda")
    model, config = load_checkpoint(Path(args.checkpoint), device)
    _, val_ds, _ = prepare_lm_datasets(DataConfig(**config["data"]))
    out_dir = ensure_dir(ROOT / "results")
    plot_dir = ensure_dir(ROOT / "results" / "plots")
    rng = np.random.default_rng(args.seed)
    rows = []

    for seq_idx in range(min(args.num_sequences, len(val_ds))):
        tokens = val_ds[seq_idx]["input_ids"]
        if tokens.numel() < args.prompt_len + args.decode_len:
            continue
        prompt_ids = tokens[: args.prompt_len].unsqueeze(0).to(device)
        continuation_ids = tokens[args.prompt_len : args.prompt_len + args.decode_len].unsqueeze(0).to(device)
        full_ids = tokens[: args.prompt_len + args.decode_len].unsqueeze(0).to(device)

        with torch.no_grad():
            prompt_outputs = model(prompt_ids, labels=None, use_cache=True, record_mode="full")
            full_outputs = model(full_ids, labels=full_ids, record_mode="full")
        prompt_summary = compute_chunk_utility(
            prompt_outputs["stats"]["records"],
            num_sources=model.config.num_blocks + 1,
            prompt_len=args.prompt_len,
            num_chunks=args.num_chunks,
        )
        decode_utility = compute_utility_from_records(
            full_outputs["stats"]["records"],
            num_sources=model.config.num_blocks + 1,
            token_start=args.prompt_len,
            token_end=args.prompt_len + args.decode_len,
        )
        prompt_scores = compute_routing_scores(
            prompt_summary.full_utility,
            prompt_summary.chunk_variance,
            prompt_summary.topk_frequency,
            mode=args.score_mode,
        )
        oracle_scores = decode_utility

        for skip_fraction in args.skip_fractions:
            routes = {
                "no_skip": np.ones(model.config.num_blocks, dtype=np.bool_),
                "balanced": balanced_skip_route(model.config.num_blocks, skip_fraction),
                "random": random_skip_route(model.config.num_blocks, skip_fraction, rng),
                "oracle": select_prompt_fixed_route(oracle_scores, model.config.num_blocks, skip_fraction),
                "prompt_fixed": select_prompt_fixed_route(prompt_scores, model.config.num_blocks, skip_fraction),
            }

            for method, route in routes.items():
                route_tensor = torch.tensor(route, device=device)
                with torch.no_grad():
                    routed = model(full_ids, labels=full_ids, active_block_mask=route_tensor, record_mode="none")
                timing = teacher_forced_decode_timing(
                    model,
                    prompt_ids=prompt_ids,
                    continuation_ids=continuation_ids,
                    active_block_mask=route_tensor,
                )
                rows.append(
                    {
                        "sequence_idx": seq_idx,
                        "method": method,
                        "skip_fraction": skip_fraction,
                        "prompt_len": args.prompt_len,
                        "decode_len": args.decode_len,
                        "continuation_loss": continuation_loss(routed["logits"], full_ids, args.prompt_len),
                        "prefill_seconds": timing.prefill_seconds,
                        "decode_seconds": timing.decode_seconds,
                        "decode_tokens_per_sec": timing.decode_tokens_per_sec,
                        "routing_overhead_seconds": timing.routing_overhead_seconds,
                        "active_blocks": int(route.sum()),
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "prompt_routing_eval.csv", index=False)
    if not df.empty:
        grouped = df.groupby(["method", "skip_fraction"], as_index=False).mean(numeric_only=True)
        grouped.to_csv(out_dir / "prompt_routing_eval_summary.csv", index=False)

        pivot = grouped.pivot(index="skip_fraction", columns="method", values="continuation_loss")
        ax = pivot.plot(marker="o", figsize=(6, 4))
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(plot_dir / "skip_budget_vs_loss.png", dpi=160)
        fig.clf()

        pivot = grouped.pivot(index="skip_fraction", columns="method", values="decode_tokens_per_sec")
        ax = pivot.plot(marker="o", figsize=(6, 4))
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(plot_dir / "skip_budget_vs_decode_toks.png", dpi=160)
        fig.clf()


if __name__ == "__main__":
    main()
