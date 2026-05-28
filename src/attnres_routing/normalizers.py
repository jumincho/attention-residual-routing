"""Depth-axis normalizers for the block-attention-residual mixing layer.

When the model runs in ``residual_mode="block_attnres"``, every block has a
:class:`~attnres_routing.model.DepthMix` head that turns per-source logits
into per-source weights along the depth axis. *This* module owns the
candidate normalizers that head can pick from. Which one is used at training
time is controlled by ``AttnResConfig.depth_normalizer`` and the per-mode
``depth_temperature`` / ``topk_softmax_k`` settings.

Modes:

- ``softmax`` / ``temperature_softmax`` — standard temperature-scaled softmax
  along the depth axis. Always-dense weights, easiest baseline.
- ``sparsemax`` — Martins & Astudillo's sparse projection. Can return exact
  zeros, so unused depth sources get masked out cleanly.
- ``entmax15`` — α=1.5 entmax. Sits between softmax (dense) and sparsemax
  (very sparse); used as the in-between knob in the late rounds.
- ``topk_softmax`` — softmax restricted to the top-``k`` depth sources;
  everything else is forced to zero pre-softmax. Drives the family of
  experiments where the model is *required* to read from few depths.

``depth_normalize`` is the single dispatcher all of those go through, so
:mod:`attnres_routing.model` only ever calls one entry point.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _transpose_last(x: torch.Tensor, dim: int) -> tuple[torch.Tensor, int]:
    if dim < 0:
        dim = x.dim() + dim
    if dim == x.dim() - 1:
        return x, dim
    perm = list(range(x.dim()))
    perm[dim], perm[-1] = perm[-1], perm[dim]
    return x.permute(*perm), dim


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    z, dim = _transpose_last(logits, dim)
    z_sorted, _ = torch.sort(z, descending=True, dim=-1)
    z_cumsum = z_sorted.cumsum(dim=-1)
    k = torch.arange(1, z.size(-1) + 1, device=z.device, dtype=z.dtype)
    support = 1 + k * z_sorted > z_cumsum
    support_size = support.sum(dim=-1, keepdim=True).clamp_min(1)
    tau = (z_cumsum.gather(-1, support_size - 1) - 1) / support_size.to(z.dtype)
    output = torch.clamp(z - tau, min=0.0)
    if dim != logits.dim() - 1:
        inv_perm = list(range(output.dim()))
        inv_perm[dim], inv_perm[-1] = inv_perm[-1], inv_perm[dim]
        output = output.permute(*inv_perm)
    return output


def entmax15(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x, dim = _transpose_last(logits, dim)
    x = x / 2.0
    x_sorted, _ = torch.sort(x, descending=True, dim=-1)
    rho = torch.arange(1, x.size(-1) + 1, device=x.device, dtype=x.dtype)
    mean = x_sorted.cumsum(dim=-1) / rho
    mean_sq = (x_sorted.square()).cumsum(dim=-1) / rho
    ss = rho * (mean_sq - mean.square())
    delta = (1 - ss) / rho
    delta = torch.clamp(delta, min=0.0)
    tau = mean - torch.sqrt(delta)
    support = tau <= x_sorted
    support_size = support.sum(dim=-1, keepdim=True).clamp_min(1)
    tau_star = tau.gather(-1, support_size - 1)
    output = torch.clamp(x - tau_star, min=0.0).square()
    if dim != logits.dim() - 1:
        inv_perm = list(range(output.dim()))
        inv_perm[dim], inv_perm[-1] = inv_perm[-1], inv_perm[dim]
        output = output.permute(*inv_perm)
    return output


def topk_softmax(logits: torch.Tensor, k: int, dim: int = -1) -> torch.Tensor:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if k >= logits.size(dim):
        return F.softmax(logits, dim=dim)
    values, indices = torch.topk(logits, k=k, dim=dim)
    masked = torch.full_like(logits, float("-inf"))
    masked.scatter_(dim, indices, values)
    return F.softmax(masked, dim=dim)


def depth_normalize(
    logits: torch.Tensor,
    mode: str,
    dim: int = -1,
    temperature: float = 1.0,
    topk: Optional[int] = None,
) -> torch.Tensor:
    scaled = logits / max(temperature, 1e-6)
    if mode == "softmax":
        return F.softmax(scaled, dim=dim)
    if mode == "temperature_softmax":
        return F.softmax(scaled, dim=dim)
    if mode == "sparsemax":
        return sparsemax(scaled, dim=dim)
    if mode == "entmax15":
        return entmax15(scaled, dim=dim)
    if mode == "topk_softmax":
        if topk is None:
            raise ValueError("topk_softmax requires topk")
        return topk_softmax(scaled, k=topk, dim=dim)
    raise ValueError(f"Unsupported depth normalizer: {mode}")
