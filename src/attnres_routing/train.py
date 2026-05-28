"""Base language-model training loop (no routing — just pretraining).

This is the trainer that ``scripts/train_lm.py`` calls. It owns the
"normal" half of the experiment: given a config, fit a ``DecoderLM`` to
next-token prediction on one of the registered datasets. Routing is *not*
trained here — routing is downstream evaluation on top of these
checkpoints (see :mod:`attnres_routing.routing` and the various
``scripts/evaluate_*`` and ``scripts/train_candidate_conditioned_ranker_*``
entrypoints).

What this module owns:

- :class:`TrainConfig` — full training-side hyperparameters (seed, lr,
  warmup / cosine schedule, grad accumulation, AMP dtype, checkpointing,
  routing-/usage-entropy regularizer knobs, the STP regularizer knobs).
- :func:`train_experiment` — the actual train/eval loop. Handles DDP
  init, dataset prep via :mod:`attnres_routing.data`, AdamW with
  param-group weight-decay split, AMP via ``GradScaler``, cosine LR via
  :func:`attnres_routing.utils.cosine_lr`, periodic eval, best-checkpoint
  tracking, per-step metrics CSV, loss-curve / source-usage plots, and
  the closure-friendly ``summary.json`` / ``summary.csv`` rows.
- :func:`routing_regularizer` — optional entropy floor on the per-layer
  depth weights (``routing_entropy_*``) and on the aggregate per-source
  usage distribution (``source_usage_entropy_*``). Encourages the
  attention-residual stack not to collapse onto a single depth source.
- :func:`stp_regularizer` — "smooth trajectory penalty": penalizes
  hidden-state trajectories whose forward and backward chord directions
  diverge in cosine. Optional, weight 0 disables it.
- :func:`evaluate` — held-out val loss / perplexity, plus the STP
  diagnostic.
- :func:`save_checkpoint`, :func:`save_metrics_csv`,
  :func:`append_summary`, :func:`plot_training`,
  :func:`save_source_usage_plot` — IO and diagnostic plot helpers.
- :func:`init_distributed` / :func:`cleanup_distributed` /
  :func:`unwrap_model` / :func:`reduce_scalar` — DDP plumbing.
"""
from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from .data import DataConfig, LanguageModelCollator, prepare_lm_datasets
from .model import AttnResConfig, DecoderLM
from .utils import cosine_lr, count_parameters, ensure_dir, save_yaml, set_seed

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass
class TrainConfig:
    experiment_name: str
    seed: int = 42
    output_root: str = "results"
    batch_size: int = 4
    eval_batch_size: int = 4
    grad_accum_steps: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 50
    max_steps: int = 200
    eval_interval: int = 50
    eval_batches: int = 20
    log_interval: int = 10
    num_workers: int = 2
    amp_dtype: str = "fp16"
    clip_grad_norm: float = 1.0
    compile: bool = False
    save_steps: tuple[int, ...] = ()
    save_initial_checkpoint: bool = False
    resume_from: Optional[str] = None
    routing_entropy_floor: float = 0.0
    routing_entropy_weight: float = 0.0
    source_usage_entropy_floor: float = 0.0
    source_usage_entropy_weight: float = 0.0
    stp_weight: float = 0.0
    stp_num_triplets: int = 2


def init_distributed() -> dict[str, Any]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return {"enabled": False, "rank": 0, "world_size": 1, "local_rank": 0}
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return {"enabled": True, "rank": rank, "world_size": world_size, "local_rank": local_rank}


def cleanup_distributed(enabled: bool) -> None:
    if enabled and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model: torch.nn.Module) -> DecoderLM:
    return model.module if isinstance(model, DDP) else model


def reduce_scalar(value: float, device: torch.device, enabled: bool) -> float:
    if not enabled:
        return value
    tensor = torch.tensor(value, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return float(tensor.item())


def build_optimizer(model: DecoderLM, config: TrainConfig) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "norm" in name or "query" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    param_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=config.learning_rate, betas=config.betas)


def _autocast_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported amp_dtype: {name}")


def grad_norm_by_layer(model: DecoderLM) -> list[float]:
    norms = []
    for layer in model.layers:
        total = 0.0
        count = 0
        for param in layer.parameters():
            if param.grad is not None:
                total += float(param.grad.detach().float().norm().item())
                count += 1
        norms.append(total / max(count, 1))
    return norms


def evaluate(
    model: DecoderLM,
    dataloader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    max_batches: int,
    train_config: Optional[TrainConfig] = None,
) -> dict[str, float]:
    model.eval()
    losses = []
    stp_losses = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                outputs = model(
                    input_ids=input_ids,
                    labels=labels,
                    record_mode="summary",
                    return_block_states=bool(train_config and train_config.stp_weight > 0.0),
                )
                _stp_loss, stp_metrics = stp_regularizer(
                    outputs.get("final_hidden"),
                    train_config,
                    device,
                )
            losses.append(float(outputs["loss"].detach().cpu()))
            if train_config and train_config.stp_weight > 0.0:
                stp_losses.append(float(stp_metrics["stp_loss_live"]))
    model.train()
    mean_loss = sum(losses) / max(len(losses), 1)
    return {
        "val_loss": mean_loss,
        "val_ppl": math.exp(min(mean_loss, 20.0)),
        "val_stp_loss": (sum(stp_losses) / max(len(stp_losses), 1)) if stp_losses else "",
    }


def save_checkpoint(
    path: Path,
    raw_model: DecoderLM,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: dict[str, Any],
    step: int,
    best_val_loss: float,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    payload = {
        "model_state": raw_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "config": config,
        "step": step,
        "best_val_loss": best_val_loss,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def save_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_summary(summary_path: Path, row: dict[str, Any]) -> None:
    exists = summary_path.exists()
    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def plot_training(metrics: list[dict[str, Any]], out_dir: Path, experiment_name: str) -> None:
    if not metrics:
        return
    steps = [row["step"] for row in metrics]
    train_losses = [row["train_loss"] for row in metrics]
    val_rows = [row for row in metrics if row["val_loss"] != ""]
    plt.figure(figsize=(6, 4))
    plt.plot(steps, train_losses, label="train_loss")
    if val_rows:
        plt.plot([row["step"] for row in val_rows], [row["val_loss"] for row in val_rows], label="val_loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{experiment_name}_loss_curve.png", dpi=160)
    plt.close()


def save_source_usage_plot(usages: list[list[float]], out_dir: Path, experiment_name: str) -> None:
    if not usages:
        return
    avg_usage = torch.tensor(usages).mean(dim=0).tolist()
    plt.figure(figsize=(6, 4))
    plt.bar(range(len(avg_usage)), avg_usage)
    plt.xlabel("source_id (0=embedding)")
    plt.ylabel("avg usage")
    plt.tight_layout()
    plt.savefig(out_dir / f"{experiment_name}_source_usage.png", dpi=160)
    plt.close()


def routing_regularizer(
    stats: dict[str, Any],
    config: TrainConfig,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    reg_loss = torch.zeros((), device=device)
    metrics = {
        "routing_reg_loss": 0.0,
        "routing_entropy_live": 0.0,
        "source_usage_entropy_live": 0.0,
    }

    entropy_tensors = stats.get("entropy_tensors", [])
    if config.routing_entropy_weight > 0.0 and entropy_tensors:
        mean_entropy = torch.stack(entropy_tensors).mean()
        penalty = torch.relu(mean_entropy.new_tensor(config.routing_entropy_floor) - mean_entropy)
        reg_loss = reg_loss + (config.routing_entropy_weight * penalty)
        metrics["routing_entropy_live"] = float(mean_entropy.detach().item())

    usage_tensors = stats.get("usage_tensors", [])
    if config.source_usage_entropy_weight > 0.0 and usage_tensors:
        usage = torch.stack(usage_tensors, dim=0).mean(dim=0)
        usage = usage / usage.sum().clamp_min(1e-8)
        usage_entropy = -(usage.clamp_min(1e-8) * usage.clamp_min(1e-8).log()).sum()
        penalty = torch.relu(usage_entropy.new_tensor(config.source_usage_entropy_floor) - usage_entropy)
        reg_loss = reg_loss + (config.source_usage_entropy_weight * penalty)
        metrics["source_usage_entropy_live"] = float(usage_entropy.detach().item())

    metrics["routing_reg_loss"] = float(reg_loss.detach().item())
    return reg_loss, metrics


def stp_regularizer(
    final_hidden: Optional[torch.Tensor],
    config: Optional[TrainConfig],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    zero = torch.zeros((), device=device)
    metrics = {
        "stp_loss_live": 0.0,
        "stp_cos_live": 0.0,
    }
    if config is None or config.stp_weight <= 0.0 or final_hidden is None:
        return zero, metrics

    batch_size, seq_len, hidden_dim = final_hidden.shape
    if batch_size <= 0 or seq_len < 3 or hidden_dim <= 0:
        return zero, metrics

    hidden = final_hidden.float()
    num_triplets = max(int(config.stp_num_triplets), 1)
    noise = torch.rand(batch_size, num_triplets, seq_len, device=hidden.device)
    triplets = noise.topk(k=3, dim=-1, largest=False).indices.sort(dim=-1).values
    s_idx = triplets[..., 0]
    r_idx = triplets[..., 1]
    t_idx = triplets[..., 2]

    batch_idx = torch.arange(batch_size, device=hidden.device).unsqueeze(1)
    h_s = hidden[batch_idx, s_idx]
    h_r = hidden[batch_idx, r_idx]
    h_t = hidden[batch_idx, t_idx]

    forward_diff = h_t - h_r
    backward_diff = h_r - h_s
    cos = F.cosine_similarity(forward_diff, backward_diff, dim=-1, eps=1e-8)
    stp_loss_unweighted = (1.0 - cos).mean()

    metrics["stp_loss_live"] = float(stp_loss_unweighted.detach().item())
    metrics["stp_cos_live"] = float(cos.mean().detach().item())
    return config.stp_weight * stp_loss_unweighted, metrics


def train_experiment(config: dict[str, Any]) -> dict[str, Any]:
    dist_info = init_distributed()
    rank = dist_info["rank"]
    is_main = rank == 0
    device = torch.device(f"cuda:{dist_info['local_rank']}")
    amp_dtype = _autocast_dtype(config["train"]["amp_dtype"])
    set_seed(config["train"]["seed"] + rank)

    model_config = AttnResConfig.from_dict(config["model"])
    train_config = TrainConfig(**config["train"])
    data_config = DataConfig(**config["data"])

    out_dir = ensure_dir(Path(train_config.output_root) / train_config.experiment_name)
    plots_dir = ensure_dir(out_dir / "plots")
    if is_main:
        save_yaml(out_dir / "config.yaml", config)

    train_ds, val_ds, _ = prepare_lm_datasets(data_config)
    collator = LanguageModelCollator()
    train_sampler = DistributedSampler(train_ds, shuffle=True) if dist_info["enabled"] else None
    train_loader = DataLoader(
        train_ds,
        batch_size=train_config.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=train_config.num_workers,
        pin_memory=True,
        collate_fn=collator,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_config.eval_batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=True,
        collate_fn=collator,
        drop_last=False,
    )

    model = DecoderLM(model_config).to(device)
    if train_config.compile:
        model = torch.compile(model)
    optimizer = build_optimizer(model, train_config)
    scaler = torch.amp.GradScaler("cuda", enabled=train_config.amp_dtype == "fp16")
    if dist_info["enabled"]:
        model = DDP(
            model,
            device_ids=[dist_info["local_rank"]],
            find_unused_parameters=model_config.residual_mode == "standard",
        )
    raw_model = unwrap_model(model)

    metrics_rows: list[dict[str, Any]] = []
    source_usage_rows: list[list[float]] = []
    best_val_loss = float("inf")
    tokens_since_log = 0
    t0 = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    save_steps = {int(step) for step in train_config.save_steps}
    step = 0
    start_epoch = 0
    if train_config.resume_from:
        payload = torch.load(train_config.resume_from, map_location=device)
        raw_model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        if "scaler_state" in payload:
            scaler.load_state_dict(payload["scaler_state"])
        step = int(payload.get("step", 0))
        start_epoch = int(payload.get("epoch", 0))
        best_val_loss = float(payload.get("best_val_loss", best_val_loss))
        if is_main:
            print(json.dumps({"event": "resume", "step": step, "epoch": start_epoch, "path": train_config.resume_from}))

    if is_main and train_config.save_initial_checkpoint:
        save_checkpoint(
            out_dir / "checkpoint_step_000000.pt",
            raw_model,
            optimizer,
            scaler,
            config,
            step=step,
            best_val_loss=best_val_loss,
            extra={"epoch": start_epoch},
        )

    epoch = start_epoch
    while step < train_config.max_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for batch in train_loader:
            if step >= train_config.max_steps:
                break
            step += 1

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            record_mode = "summary" if raw_model.config.residual_mode == "block_attnres" else "none"

            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                outputs = model(
                    input_ids=input_ids,
                    labels=labels,
                    record_mode=record_mode,
                    return_block_states=train_config.stp_weight > 0.0,
                )
                reg_loss, reg_metrics = routing_regularizer(outputs["stats"], train_config, device)
                stp_loss, stp_metrics = stp_regularizer(outputs.get("final_hidden"), train_config, device)
                total_loss = outputs["loss"] + reg_loss + stp_loss
                loss = total_loss / train_config.grad_accum_steps

            scaler.scale(loss).backward()
            tokens_since_log += int(input_ids.numel())

            current_grad_norms = grad_norm_by_layer(raw_model)
            if step % train_config.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                lr = cosine_lr(
                    step=step,
                    warmup_steps=train_config.warmup_steps,
                    max_steps=train_config.max_steps,
                    base_lr=train_config.learning_rate,
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr

            if step % train_config.log_interval == 0 or step == 1:
                elapsed = max(time.perf_counter() - t0, 1e-6)
                train_loss = reduce_scalar(float(outputs["loss"].detach().item()), device, dist_info["enabled"])
                activation_norms = outputs["stats"]["activation_norms"]
                memory_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                tokens_per_sec = tokens_since_log / elapsed
                row = {
                    "step": step,
                    "train_loss": train_loss,
                    "val_loss": "",
                    "val_ppl": "",
                    "val_stp_loss": "",
                    "lr": optimizer.param_groups[0]["lr"],
                    "tokens_per_sec": tokens_per_sec,
                    "memory_gb": memory_gb,
                    "mean_grad_norm": sum(current_grad_norms) / max(len(current_grad_norms), 1),
                    "mean_activation_norm": sum(activation_norms) / max(len(activation_norms), 1),
                    "routing_reg_loss": reg_metrics["routing_reg_loss"],
                    "routing_entropy_live": reg_metrics["routing_entropy_live"],
                    "source_usage_entropy_live": reg_metrics["source_usage_entropy_live"],
                    "stp_loss_live": stp_metrics["stp_loss_live"],
                    "stp_cos_live": stp_metrics["stp_cos_live"],
                    "stp_weight": train_config.stp_weight,
                    "mean_depth_entropy": (
                        sum(outputs["stats"]["depth_entropies"]) / max(len(outputs["stats"]["depth_entropies"]), 1)
                        if outputs["stats"]["depth_entropies"]
                        else ""
                    ),
                    "mean_support_size": (
                        sum(outputs["stats"]["depth_support_sizes"]) / max(len(outputs["stats"]["depth_support_sizes"]), 1)
                        if outputs["stats"]["depth_support_sizes"]
                        else ""
                    ),
                }
                metrics_rows.append(row)
                if raw_model.config.residual_mode == "block_attnres":
                    source_usage_rows.append(outputs["stats"]["source_usage"].detach().cpu().tolist())
                if is_main:
                    print(json.dumps({"event": "train_log", **row}))
                    save_metrics_csv(out_dir / "metrics.csv", metrics_rows)
                tokens_since_log = 0
                t0 = time.perf_counter()

            if step in save_steps and is_main:
                save_checkpoint(
                    out_dir / f"checkpoint_step_{step:06d}.pt",
                    raw_model,
                    optimizer,
                    scaler,
                    config,
                    step=step,
                    best_val_loss=best_val_loss,
                    extra={"epoch": epoch},
                )

            if step % train_config.eval_interval == 0 or step == train_config.max_steps:
                if dist_info["enabled"]:
                    dist.barrier()
                if is_main:
                    eval_metrics = evaluate(
                        raw_model,
                        val_loader,
                        device,
                        amp_dtype,
                        train_config.eval_batches,
                        train_config=train_config,
                    )
                    metrics_rows[-1]["val_loss"] = eval_metrics["val_loss"]
                    metrics_rows[-1]["val_ppl"] = eval_metrics["val_ppl"]
                    metrics_rows[-1]["val_stp_loss"] = eval_metrics["val_stp_loss"]
                    if eval_metrics["val_loss"] < best_val_loss:
                        best_val_loss = eval_metrics["val_loss"]
                        save_checkpoint(
                            out_dir / "best_checkpoint.pt",
                            raw_model,
                            optimizer,
                            scaler,
                            config,
                            step=step,
                            best_val_loss=best_val_loss,
                            extra={"epoch": epoch, "val_loss": best_val_loss},
                        )
                    print(json.dumps({"event": "eval_log", "step": step, **eval_metrics}))
                    save_metrics_csv(out_dir / "metrics.csv", metrics_rows)
                if dist_info["enabled"]:
                    dist.barrier()
        epoch += 1

    if is_main:
        save_checkpoint(
            out_dir / "last_checkpoint.pt",
            raw_model,
            optimizer,
            scaler,
            config,
            step=step,
            best_val_loss=best_val_loss,
            extra={"epoch": epoch},
        )
        save_metrics_csv(out_dir / "metrics.csv", metrics_rows)
        plot_training(metrics_rows, plots_dir, train_config.experiment_name)
        save_source_usage_plot(source_usage_rows, plots_dir, train_config.experiment_name)
        final_val_rows = [row for row in metrics_rows if row["val_loss"] != ""]
        final_val = final_val_rows[-1]["val_loss"] if final_val_rows else None
        summary_row = {
            "experiment_name": train_config.experiment_name,
            "residual_mode": model_config.residual_mode,
            "num_layers": model_config.num_layers,
            "d_model": model_config.d_model,
            "num_blocks": model_config.num_blocks,
            "dataset_name": data_config.dataset_name,
            "steps": train_config.max_steps,
            "parameter_count": count_parameters(raw_model),
            "best_val_loss": best_val_loss if best_val_loss < float("inf") else "",
            "final_logged_val_loss": final_val if final_val is not None else "",
        }
        append_summary(Path(train_config.output_root) / "summary.csv", summary_row)
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_row, f, indent=2)

    cleanup_distributed(dist_info["enabled"])
    return {"output_dir": str(out_dir), "best_val_loss": best_val_loss}
