"""Per-block sublayer activity masks — the routing vocabulary.

The whole experiment turns on *which sub-layers are active on this input*.
This module is the canonical place where that idea is encoded. Every routing
script and every evaluation script talks about decisions in terms of
:class:`SublayerMask` objects.

The vocabulary:

- :class:`SublayerMask` carries two boolean tuples of length ``num_blocks``:
  ``attn_mask`` (attention sub-layer active per block) and ``mlp_mask`` (MLP
  sub-layer active per block).
- :attr:`SublayerMask.action_types` maps each block to one of the four
  routing actions referenced everywhere: ``full`` (run both), ``skip_attn``
  (drop attention, keep MLP), ``skip_mlp`` (keep attention, drop MLP),
  ``skip_block`` (drop both).
- :meth:`SublayerMask.to_id` produces a stable string id of the form
  ``attn:1101|mlp:1011`` — that string is the key used in JSON / CSV /
  manifests for the candidate-conditioned ranker and the deployment reports.
  :func:`from_id` parses it back.

Convenience builders:

- :func:`from_block_mask` lifts a per-block on/off vector to a full mask.
- :func:`enumerate_local_edits` generates the candidate set around an anchor
  mask (used by the v7 candidate-conditioned ranker).
- :func:`edit_distance` / :func:`per_block_action` / :func:`apply_action`
  are the small graph helpers the search uses.

Cost helpers — :func:`estimated_decode_cost` and
:func:`estimated_reduction_ratio` — turn an arbitrary mask into a predicted
relative FLOP cost given measured per-block ``full`` / ``attn`` / ``mlp``
sub-costs; that is the input to the FLOP-budgeted analyses.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SublayerMask:
    attn_mask: tuple[bool, ...]
    mlp_mask: tuple[bool, ...]

    @property
    def num_blocks(self) -> int:
        return len(self.attn_mask)

    @property
    def action_types(self) -> tuple[str, ...]:
        actions = []
        for attn_on, mlp_on in zip(self.attn_mask, self.mlp_mask):
            if attn_on and mlp_on:
                actions.append("full")
            elif (not attn_on) and mlp_on:
                actions.append("skip_attn")
            elif attn_on and (not mlp_on):
                actions.append("skip_mlp")
            else:
                actions.append("skip_block")
        return tuple(actions)

    def to_id(self) -> str:
        attn_bits = "".join("1" if value else "0" for value in self.attn_mask)
        mlp_bits = "".join("1" if value else "0" for value in self.mlp_mask)
        return f"attn:{attn_bits}|mlp:{mlp_bits}"

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(self.attn_mask, dtype=np.bool_),
            np.asarray(self.mlp_mask, dtype=np.bool_),
        )


def from_id(mask_id: str) -> SublayerMask:
    attn_part, mlp_part = mask_id.split("|", 1)
    attn_bits = attn_part.split(":", 1)[1].strip()
    mlp_bits = mlp_part.split(":", 1)[1].strip()
    if len(attn_bits) != len(mlp_bits):
        raise ValueError(f"Mismatched mask lengths in {mask_id}")
    return SublayerMask(
        attn_mask=tuple(bit == "1" for bit in attn_bits),
        mlp_mask=tuple(bit == "1" for bit in mlp_bits),
    )


def from_block_mask(block_mask: Iterable[bool]) -> SublayerMask:
    block_mask = tuple(bool(value) for value in block_mask)
    return SublayerMask(attn_mask=block_mask, mlp_mask=block_mask)


def edit_distance(mask_a: SublayerMask, mask_b: SublayerMask, ignore_final: bool = True) -> int:
    attn_a, mlp_a = mask_a.to_arrays()
    attn_b, mlp_b = mask_b.to_arrays()
    if ignore_final:
        attn_a = attn_a[:-1]
        attn_b = attn_b[:-1]
        mlp_a = mlp_a[:-1]
        mlp_b = mlp_b[:-1]
    return int(np.logical_xor(attn_a, attn_b).sum() + np.logical_xor(mlp_a, mlp_b).sum())


def per_block_action(mask: SublayerMask, block_idx: int) -> str:
    return mask.action_types[block_idx]


def apply_action(mask: SublayerMask, block_idx: int, action: str) -> SublayerMask:
    attn = list(mask.attn_mask)
    mlp = list(mask.mlp_mask)
    if action == "full":
        attn[block_idx] = True
        mlp[block_idx] = True
    elif action == "skip_attn":
        attn[block_idx] = False
        mlp[block_idx] = True
    elif action == "skip_mlp":
        attn[block_idx] = True
        mlp[block_idx] = False
    elif action == "skip_block":
        attn[block_idx] = False
        mlp[block_idx] = False
    else:
        raise ValueError(f"Unsupported action: {action}")
    return SublayerMask(attn_mask=tuple(attn), mlp_mask=tuple(mlp))


def enumerate_local_edits(
    anchor: SublayerMask,
    block_indices: list[int],
    max_edits: int,
    candidate_actions: tuple[str, ...] = ("full", "skip_attn", "skip_mlp", "skip_block"),
) -> list[SublayerMask]:
    candidates = {anchor.to_id(): anchor}
    for edit_count in range(1, max_edits + 1):
        for blocks in combinations(block_indices, edit_count):
            action_choices = []
            for block_idx in blocks:
                current = per_block_action(anchor, block_idx)
                valid = [action for action in candidate_actions if action != current]
                action_choices.append(valid)
            for actions in product(*action_choices):
                mask = anchor
                for block_idx, action in zip(blocks, actions):
                    mask = apply_action(mask, block_idx, action)
                candidates[mask.to_id()] = mask
    return list(candidates.values())


def estimated_decode_cost(mask: SublayerMask, full_cost: np.ndarray, attn_cost: np.ndarray, mlp_cost: np.ndarray) -> float:
    total = 0.0
    for action, block_full, block_attn, block_mlp in zip(mask.action_types, full_cost, attn_cost, mlp_cost):
        if action == "full":
            total += float(block_full)
        elif action == "skip_attn":
            total += float(block_mlp)
        elif action == "skip_mlp":
            total += float(block_attn)
        elif action == "skip_block":
            total += 0.0
    return float(total)


def estimated_reduction_ratio(mask: SublayerMask, full_cost: np.ndarray, attn_cost: np.ndarray, mlp_cost: np.ndarray) -> float:
    baseline = float(np.sum(full_cost))
    candidate_cost = estimated_decode_cost(mask, full_cost, attn_cost, mlp_cost)
    if baseline <= 0.0:
        return 0.0
    return float((baseline - candidate_cost) / baseline)
