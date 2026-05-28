"""Small utilities shared across the experiment.

This module is intentionally narrow: it owns the boring helpers that nearly
every other module in :mod:`attnres_routing` reaches for, and nothing else.
Keeping them in one place means training, evaluation, and analysis all agree
on seeding, config I/O, directory creation, and the learning-rate schedule.

What lives here:

- :func:`set_seed` — seeds Python, NumPy, and Torch (CPU + CUDA) together so
  that a single integer is the only knob a config has to set.
- :func:`load_yaml` / :func:`save_yaml` — round-trip the per-experiment YAML
  configs in ``configs/`` (including the ``scale_heterogeneity_v*`` rounds).
- :func:`ensure_dir` — create an output directory and return it as a
  :class:`pathlib.Path`; used for results / plots / checkpoints.
- :func:`resolve_hf_token` — read ``HF_TOKEN`` or
  ``HUGGINGFACE_HUB_TOKEN`` for gated datasets (FineWeb-Edu in particular).
- :func:`count_parameters` — parameter count used in training summaries.
- :func:`cosine_lr` — cosine learning-rate schedule with linear warmup that
  :mod:`attnres_routing.train` applies per step.
"""
from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    return data


def save_yaml(path: str | os.PathLike[str], data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def resolve_hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def count_parameters(module: torch.nn.Module) -> int:
    return sum(param.numel() for param in module.parameters())


def cosine_lr(step: int, warmup_steps: int, max_steps: int, base_lr: float, min_lr_ratio: float = 0.1) -> float:
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    min_lr = base_lr * min_lr_ratio
    return min_lr + (base_lr - min_lr) * cosine
