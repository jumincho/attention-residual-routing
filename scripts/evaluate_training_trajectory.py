#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attnres_routing.analysis import (  # noqa: E402
    compute_chunk_utility,
    compute_utility_from_records,
    summarize_prompt_decode_transfer,
)
from attnres_routing.data import DataConfig, LanguageModelCollator, prepare_lm_datasets  # noqa: E402
from attnres_routing.model import AttnResConfig, DecoderLM  # noqa: E402


def load_checkpoint(path: Path, device: torch.device) -> tuple[DecoderLM, dict[str, Any], int]:
    payload = torch.load(path, map_location=device)
    config = payload["config"]
    model = DecoderLM(AttnResConfig.from_dict(config["model"])).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, config, int(payload.get("step", 0))


def evaluate_val_loss(
    model: DecoderLM,
    val_ds,
    device: torch.device,
    eval_batch_size: int,
    eval_batches: int,
) -> tuple[float, float]:
    loader = DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=LanguageModelCollator(),
        num_workers=0,
    )
    losses = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= eval_batches:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(input_ids=input_ids, labels=labels, record_mode="summary")
            losses.append(float(outputs["loss"].detach().cpu()))
    val_loss = float(np.mean(losses)) if losses else float("nan")
    return val_loss, math.exp(min(val_loss, 20.0)) if np.isfinite(val_loss) else float("nan")


def trajectory_row_from_checkpoint(
    checkpoint_path: Path,
    val_ds,
    prompt_len: int,
    decode_len: int,
    num_sequences: int,
    num_chunks: int,
    eval_batches: int,
    device: torch.device,
) -> dict[str, Any]:
    model, config, step = load_checkpoint(checkpoint_path, device)
    val_loss, val_ppl = evaluate_val_loss(
        model,
        val_ds=val_ds,
        device=device,
        eval_batch_size=int(config["train"]["eval_batch_size"]),
        eval_batches=eval_batches,
    )
    row: dict[str, Any] = {
        "step": step,
        "checkpoint_path": str(checkpoint_path),
        "residual_mode": model.config.residual_mode,
        "val_loss": val_loss,
        "val_ppl": val_ppl,
        "mean_train_tokens_per_sec": float("nan"),
        "mean_memory_gb": float("nan"),
        "mean_depth_entropy": float("nan"),
        "mean_support_size": float("nan"),
        "mean_transfer_spearman": float("nan"),
        "mean_transfer_kendall": float("nan"),
        "mean_transfer_top1": float("nan"),
        "mean_transfer_recall3": float("nan"),
    }

    if model.config.residual_mode != "block_attnres":
        return row

    transfer_rows = []
    depth_entropies = []
    support_sizes = []
    for seq_idx in range(min(num_sequences, len(val_ds))):
        tokens = val_ds[seq_idx]["input_ids"]
        if tokens.numel() < prompt_len + decode_len:
            continue
        sample = tokens[: prompt_len + decode_len].unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(sample, labels=None, record_mode="full")
        records = outputs["stats"]["records"]
        prompt_summary = compute_chunk_utility(
            records,
            num_sources=model.config.num_blocks + 1,
            prompt_len=prompt_len,
            num_chunks=num_chunks,
            normalize=True,
            center_uniform=True,
        )
        decode_utility = compute_utility_from_records(
            records,
            num_sources=model.config.num_blocks + 1,
            token_start=prompt_len,
            token_end=prompt_len + decode_len,
            normalize=True,
            center_uniform=True,
        )
        transfer_rows.append(summarize_prompt_decode_transfer(prompt_summary.full_utility, decode_utility))
        depth_entropies.extend(outputs["stats"]["depth_entropies"])
        support_sizes.extend(outputs["stats"]["depth_support_sizes"])

    if transfer_rows:
        row["mean_transfer_spearman"] = float(np.nanmean([item["spearman"] for item in transfer_rows]))
        row["mean_transfer_kendall"] = float(np.nanmean([item["kendall"] for item in transfer_rows]))
        row["mean_transfer_top1"] = float(np.nanmean([item["topk_jaccard_1"] for item in transfer_rows]))
        row["mean_transfer_recall3"] = float(np.nanmean([item["recall_at_3"] for item in transfer_rows]))
    if depth_entropies:
        row["mean_depth_entropy"] = float(np.mean(depth_entropies))
    if support_sizes:
        row["mean_support_size"] = float(np.mean(support_sizes))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--prompt-len", type=int, default=256)
    parser.add_argument("--decode-len", type=int, default=64)
    parser.add_argument("--num-sequences", type=int, default=32)
    parser.add_argument("--num-chunks", type=int, default=4)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    run_dir = Path(args.run_dir)
    checkpoint_paths = sorted(run_dir.glob("checkpoint_step_*.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No step checkpoints found in {run_dir}")

    first_payload = torch.load(checkpoint_paths[0], map_location="cpu")
    config = first_payload["config"]
    _, val_ds, _ = prepare_lm_datasets(DataConfig(**config["data"]))
    metrics_path = run_dir / "metrics.csv"
    train_metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()

    rows = []
    for checkpoint_path in checkpoint_paths:
        row = trajectory_row_from_checkpoint(
            checkpoint_path,
            val_ds=val_ds,
            prompt_len=args.prompt_len,
            decode_len=args.decode_len,
            num_sequences=args.num_sequences,
            num_chunks=args.num_chunks,
            eval_batches=args.eval_batches,
            device=device,
        )
        if not train_metrics.empty:
            nearest = train_metrics[train_metrics["step"] <= row["step"]].sort_values("step").tail(1)
            if not nearest.empty:
                row["mean_train_tokens_per_sec"] = float(nearest.iloc[0]["tokens_per_sec"])
                row["mean_memory_gb"] = float(nearest.iloc[0]["memory_gb"])
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("step")
    out_dir = ROOT / "results" / "trajectory"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = ROOT / "results" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{run_dir.name}_trajectory.csv", index=False)

    plt.figure(figsize=(6, 4))
    plt.plot(df["step"], df["val_loss"], marker="o")
    plt.xlabel("step")
    plt.ylabel("val loss")
    plt.tight_layout()
    plt.savefig(plot_dir / f"trajectory_{run_dir.name}_val_loss.png", dpi=160)
    plt.close()

    if df["mean_transfer_spearman"].notna().any():
        plt.figure(figsize=(6, 4))
        plt.plot(df["step"], df["mean_transfer_spearman"], marker="o", label="spearman")
        plt.plot(df["step"], df["mean_transfer_recall3"], marker="o", label="recall@3")
        plt.xlabel("step")
        plt.ylabel("transfer")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"trajectory_{run_dir.name}_transfer.png", dpi=160)
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.plot(df["step"], df["mean_depth_entropy"], marker="o", label="depth entropy")
        plt.plot(df["step"], df["mean_support_size"], marker="o", label="support size")
        plt.xlabel("step")
        plt.ylabel("support stats")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"trajectory_{run_dir.name}_support.png", dpi=160)
        plt.close()


if __name__ == "__main__":
    main()
