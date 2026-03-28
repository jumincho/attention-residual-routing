from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def _maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def compute_routing_scores(
    utility: np.ndarray,
    variance: np.ndarray,
    topk_frequency: np.ndarray,
    mode: str,
    eps: float = 1e-6,
) -> np.ndarray:
    if mode == "utility":
        return utility.copy()
    if mode == "utility_over_variance":
        return utility / (eps + variance)
    if mode == "utility_times_topk_frequency":
        return utility * topk_frequency
    raise ValueError(f"Unknown routing score mode: {mode}")


def _tertiled_block_ids(num_blocks: int) -> list[np.ndarray]:
    blocks = np.arange(1, num_blocks + 1)
    return [chunk for chunk in np.array_split(blocks, 3) if len(chunk) > 0]


def select_prompt_fixed_route(
    scores: np.ndarray,
    num_blocks: int,
    skip_fraction: float,
    keep_final: bool = True,
    enforce_tertiles: bool = True,
) -> np.ndarray:
    computational_scores = scores[1 : num_blocks + 1]
    candidate_ids = np.arange(1, num_blocks + 1)
    active_ids = set()
    if keep_final:
        active_ids.add(num_blocks)
    if enforce_tertiles:
        for tertile in _tertiled_block_ids(num_blocks):
            best = tertile[np.argmax(computational_scores[tertile - 1])]
            active_ids.add(int(best))

    target_keep = max(1, int(round((1.0 - skip_fraction) * num_blocks)))
    while len(active_ids) < target_keep:
        remaining = [idx for idx in candidate_ids.tolist() if idx not in active_ids]
        if not remaining:
            break
        best = max(remaining, key=lambda idx: computational_scores[idx - 1])
        active_ids.add(int(best))

    mask = np.zeros(num_blocks, dtype=np.bool_)
    for block_id in active_ids:
        mask[block_id - 1] = True
    return mask


def balanced_skip_route(num_blocks: int, skip_fraction: float) -> np.ndarray:
    target_keep = max(1, int(round((1.0 - skip_fraction) * num_blocks)))
    mask = np.zeros(num_blocks, dtype=np.bool_)
    if target_keep >= num_blocks:
        mask[:] = True
        return mask
    remaining_keep = max(target_keep - 1, 0)
    if remaining_keep > 0:
        keep_ids = np.linspace(1, max(num_blocks - 1, 1), remaining_keep, dtype=int)
        mask[np.unique(keep_ids) - 1] = True
    mask[-1] = True
    return mask


def random_skip_route(num_blocks: int, skip_fraction: float, rng: np.random.Generator) -> np.ndarray:
    target_keep = max(1, int(round((1.0 - skip_fraction) * num_blocks)))
    mask = np.zeros(num_blocks, dtype=np.bool_)
    if target_keep >= num_blocks:
        mask[:] = True
        return mask
    remaining_keep = max(target_keep - 1, 0)
    if remaining_keep > 0:
        ids = np.arange(1, num_blocks)
        chosen = rng.choice(ids, size=remaining_keep, replace=False)
        mask[chosen - 1] = True
    mask[-1] = True
    return mask


def continuation_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    prompt_len: int,
) -> float:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    token_losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view_as(shift_labels)
    valid = torch.zeros_like(shift_labels, dtype=torch.bool)
    valid[:, max(prompt_len - 1, 0) :] = True
    masked = token_losses[valid]
    return float(masked.mean().item())


def continuation_loss_from_decode_logits(
    decode_logits: torch.Tensor,
    continuation_ids: torch.Tensor,
) -> float:
    if decode_logits.numel() == 0:
        return float("nan")
    token_losses = F.cross_entropy(
        decode_logits.reshape(-1, decode_logits.size(-1)),
        continuation_ids.reshape(-1),
        reduction="none",
    )
    return float(token_losses.mean().item())


def continuation_losses_from_decode_logits(
    decode_logits: torch.Tensor,
    continuation_ids: torch.Tensor,
) -> torch.Tensor:
    if decode_logits.numel() == 0:
        return torch.full(
            (continuation_ids.size(0),),
            float("nan"),
            device=continuation_ids.device,
            dtype=torch.float32,
        )
    token_losses = F.cross_entropy(
        decode_logits.reshape(-1, decode_logits.size(-1)),
        continuation_ids.reshape(-1),
        reduction="none",
    ).view(continuation_ids.size(0), continuation_ids.size(1))
    return token_losses.mean(dim=1)


@dataclass
class TimingResult:
    prefill_seconds: float
    decode_seconds: float
    decode_tokens_per_sec: float
    routing_overhead_seconds: float


def filter_past_key_values(
    past_key_values: Optional[list[Optional[tuple[torch.Tensor, torch.Tensor]]]],
    active_block_mask: Optional[torch.Tensor],
    layer_to_block: list[int],
    active_attn_block_mask: Optional[torch.Tensor] = None,
) -> Optional[list[Optional[tuple[torch.Tensor, torch.Tensor]]]]:
    if past_key_values is None or active_block_mask is None:
        if past_key_values is None or active_attn_block_mask is None:
            return past_key_values
    filtered: list[Optional[tuple[torch.Tensor, torch.Tensor]]] = []
    for layer_idx, kv in enumerate(past_key_values):
        block_idx = layer_to_block[layer_idx]
        block_active = True if active_block_mask is None else bool(active_block_mask[block_idx].item())
        attn_active = block_active and (
            True if active_attn_block_mask is None else bool(active_attn_block_mask[block_idx].item())
        )
        if not attn_active:
            filtered.append(None)
        else:
            filtered.append(kv)
    return filtered


def stack_past_key_values(
    batch_past_key_values: list[Optional[list[Optional[tuple[torch.Tensor, torch.Tensor]]]]],
) -> Optional[list[Optional[tuple[torch.Tensor, torch.Tensor]]]]:
    if not batch_past_key_values:
        return None
    reference = next((item for item in batch_past_key_values if item is not None), None)
    if reference is None:
        return None
    num_layers = len(reference)
    stacked: list[Optional[tuple[torch.Tensor, torch.Tensor]]] = []
    for layer_idx in range(num_layers):
        layer_entries = [
            item[layer_idx]
            for item in batch_past_key_values
            if item is not None
        ]
        if not layer_entries or all(entry is None for entry in layer_entries):
            stacked.append(None)
            continue
        if any(entry is None for entry in layer_entries):
            raise ValueError("Cannot stack mixed None/non-None past key values within the same layer.")
        keys = torch.cat([entry[0] for entry in layer_entries], dim=0)
        values = torch.cat([entry[1] for entry in layer_entries], dim=0)
        stacked.append((keys, values))
    return stacked


def teacher_forced_decode_logits(
    model,
    prompt_ids: torch.Tensor,
    continuation_ids: torch.Tensor,
    prompt_active_block_mask: Optional[torch.Tensor] = None,
    decode_active_block_mask: Optional[torch.Tensor] = None,
    prompt_active_attn_block_mask: Optional[torch.Tensor] = None,
    prompt_active_mlp_block_mask: Optional[torch.Tensor] = None,
    decode_active_attn_block_mask: Optional[torch.Tensor] = None,
    decode_active_mlp_block_mask: Optional[torch.Tensor] = None,
    prompt_record_mode: str = "none",
):
    with torch.no_grad():
        prompt_outputs = model(
            input_ids=prompt_ids,
            labels=None,
            use_cache=True,
            active_block_mask=prompt_active_block_mask,
            active_attn_block_mask=prompt_active_attn_block_mask,
            active_mlp_block_mask=prompt_active_mlp_block_mask,
            record_mode=prompt_record_mode,
        )
        past = filter_past_key_values(
            prompt_outputs["past_key_values"],
            decode_active_block_mask,
            model.layer_to_block,
            active_attn_block_mask=decode_active_attn_block_mask,
        )
        decode_tokens = continuation_ids.size(1)
        logits_steps = [prompt_outputs["logits"][:, -1:, :]]
        current = continuation_ids[:, :1]
        for step in range(max(decode_tokens - 1, 0)):
            out = model(
                input_ids=current,
                labels=None,
                use_cache=True,
                past_key_values=past,
                active_block_mask=decode_active_block_mask,
                active_attn_block_mask=decode_active_attn_block_mask,
                active_mlp_block_mask=decode_active_mlp_block_mask,
                record_mode="none",
            )
            past = out["past_key_values"]
            logits_steps.append(out["logits"])
            if step + 1 < decode_tokens - 1:
                current = continuation_ids[:, step + 1 : step + 2]
    decode_logits = torch.cat(logits_steps, dim=1) if logits_steps else prompt_outputs["logits"][:, :0, :]
    return prompt_outputs, decode_logits


def teacher_forced_decode_from_past(
    model,
    continuation_ids: torch.Tensor,
    past_key_values: Optional[list[Optional[tuple[torch.Tensor, torch.Tensor]]]],
    prompt_last_logits: torch.Tensor,
    active_block_mask: Optional[torch.Tensor] = None,
    active_attn_block_mask: Optional[torch.Tensor] = None,
    active_mlp_block_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    with torch.no_grad():
        past = filter_past_key_values(
            past_key_values,
            active_block_mask,
            model.layer_to_block,
            active_attn_block_mask=active_attn_block_mask,
        )
        decode_tokens = continuation_ids.size(1)
        logits_steps = [prompt_last_logits]
        current = continuation_ids[:, :1]
        for step in range(max(decode_tokens - 1, 0)):
            out = model(
                input_ids=current,
                labels=None,
                use_cache=True,
                past_key_values=past,
                active_block_mask=active_block_mask,
                active_attn_block_mask=active_attn_block_mask,
                active_mlp_block_mask=active_mlp_block_mask,
                record_mode="none",
            )
            past = out["past_key_values"]
            logits_steps.append(out["logits"])
            if step + 1 < decode_tokens - 1:
                current = continuation_ids[:, step + 1 : step + 2]
    return torch.cat(logits_steps, dim=1) if logits_steps else prompt_last_logits[:, :0, :]


def decode_timing_from_past(
    model,
    continuation_ids: torch.Tensor,
    past_key_values: Optional[list[Optional[tuple[torch.Tensor, torch.Tensor]]]],
    prompt_last_logits: torch.Tensor,
    active_block_mask: Optional[torch.Tensor] = None,
    active_attn_block_mask: Optional[torch.Tensor] = None,
    active_mlp_block_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, float, float]:
    device = continuation_ids.device
    past = filter_past_key_values(
        past_key_values,
        active_block_mask,
        model.layer_to_block,
        active_attn_block_mask=active_attn_block_mask,
    )
    decode_tokens = continuation_ids.size(1)
    logits_steps = [prompt_last_logits]
    _maybe_sync(device)
    t0 = time.perf_counter()
    current = continuation_ids[:, :1]
    with torch.no_grad():
        for step in range(max(decode_tokens - 1, 0)):
            out = model(
                input_ids=current,
                labels=None,
                use_cache=True,
                past_key_values=past,
                active_block_mask=active_block_mask,
                active_attn_block_mask=active_attn_block_mask,
                active_mlp_block_mask=active_mlp_block_mask,
                record_mode="none",
            )
            past = out["past_key_values"]
            logits_steps.append(out["logits"])
            if step + 1 < decode_tokens - 1:
                current = continuation_ids[:, step + 1 : step + 2]
    _maybe_sync(device)
    elapsed = time.perf_counter() - t0
    logits = torch.cat(logits_steps, dim=1) if logits_steps else prompt_last_logits[:, :0, :]
    decode_steps = max(decode_tokens - 1, 0)
    effective_tokens = continuation_ids.size(0) * decode_steps
    return logits, elapsed, effective_tokens / max(elapsed, 1e-6)


def teacher_forced_decode_timing(
    model,
    prompt_ids: torch.Tensor,
    continuation_ids: torch.Tensor,
    active_block_mask: Optional[torch.Tensor],
    active_attn_block_mask: Optional[torch.Tensor] = None,
    active_mlp_block_mask: Optional[torch.Tensor] = None,
    route_fn=None,
) -> TimingResult:
    device = prompt_ids.device
    _maybe_sync(device)
    route_overhead = 0.0
    if route_fn is not None:
        t_route = time.perf_counter()
        route_result = route_fn()
        _maybe_sync(device)
        route_overhead = time.perf_counter() - t_route
        if isinstance(route_result, tuple):
            if len(route_result) == 3:
                active_block_mask, active_attn_block_mask, active_mlp_block_mask = route_result
            elif len(route_result) == 2:
                active_attn_block_mask, active_mlp_block_mask = route_result
            elif len(route_result) == 1:
                active_block_mask = route_result[0]
        else:
            active_block_mask = route_result

    t0 = time.perf_counter()
    prefill = model(
        input_ids=prompt_ids,
        labels=None,
        use_cache=True,
        active_block_mask=active_block_mask,
        active_attn_block_mask=active_attn_block_mask,
        active_mlp_block_mask=active_mlp_block_mask,
        record_mode="none",
    )
    _maybe_sync(device)
    prefill_seconds = time.perf_counter() - t0

    past = filter_past_key_values(
        prefill["past_key_values"],
        active_block_mask,
        model.layer_to_block,
        active_attn_block_mask=active_attn_block_mask,
    )
    decode_tokens = continuation_ids.size(1)
    t1 = time.perf_counter()
    current = continuation_ids[:, :1]
    for step in range(max(decode_tokens - 1, 0)):
        out = model(
            input_ids=current,
            labels=None,
            use_cache=True,
            past_key_values=past,
            active_block_mask=active_block_mask,
            active_attn_block_mask=active_attn_block_mask,
            active_mlp_block_mask=active_mlp_block_mask,
            record_mode="none",
        )
        past = out["past_key_values"]
        if step + 1 < decode_tokens - 1:
            current = continuation_ids[:, step + 1 : step + 2]
    _maybe_sync(device)
    decode_seconds = time.perf_counter() - t1
    tok_s = decode_tokens / max(decode_seconds, 1e-6)
    return TimingResult(
        prefill_seconds=prefill_seconds,
        decode_seconds=decode_seconds,
        decode_tokens_per_sec=tok_s,
        routing_overhead_seconds=route_overhead,
    )


def teacher_forced_decode_timing_split_masks(
    model,
    prompt_ids: torch.Tensor,
    continuation_ids: torch.Tensor,
    prompt_active_block_mask: Optional[torch.Tensor] = None,
    decode_active_block_mask: Optional[torch.Tensor] = None,
    prompt_active_attn_block_mask: Optional[torch.Tensor] = None,
    prompt_active_mlp_block_mask: Optional[torch.Tensor] = None,
    decode_active_attn_block_mask: Optional[torch.Tensor] = None,
    decode_active_mlp_block_mask: Optional[torch.Tensor] = None,
    route_fn=None,
) -> TimingResult:
    device = prompt_ids.device
    _maybe_sync(device)
    route_overhead = 0.0
    if route_fn is not None:
        t_route = time.perf_counter()
        route_result = route_fn()
        _maybe_sync(device)
        route_overhead = time.perf_counter() - t_route
        if isinstance(route_result, tuple):
            if len(route_result) == 6:
                (
                    prompt_active_block_mask,
                    decode_active_block_mask,
                    prompt_active_attn_block_mask,
                    prompt_active_mlp_block_mask,
                    decode_active_attn_block_mask,
                    decode_active_mlp_block_mask,
                ) = route_result
            elif len(route_result) == 4:
                (
                    prompt_active_attn_block_mask,
                    prompt_active_mlp_block_mask,
                    decode_active_attn_block_mask,
                    decode_active_mlp_block_mask,
                ) = route_result

    _maybe_sync(device)
    t0 = time.perf_counter()
    prompt_outputs = model(
        input_ids=prompt_ids,
        labels=None,
        use_cache=True,
        active_block_mask=prompt_active_block_mask,
        active_attn_block_mask=prompt_active_attn_block_mask,
        active_mlp_block_mask=prompt_active_mlp_block_mask,
        record_mode="none",
    )
    _maybe_sync(device)
    prefill_seconds = time.perf_counter() - t0

    past = filter_past_key_values(
        prompt_outputs["past_key_values"],
        decode_active_block_mask,
        model.layer_to_block,
        active_attn_block_mask=decode_active_attn_block_mask,
    )
    decode_tokens = continuation_ids.size(1)
    t1 = time.perf_counter()
    current = continuation_ids[:, :1]
    for step in range(max(decode_tokens - 1, 0)):
        out = model(
            input_ids=current,
            labels=None,
            use_cache=True,
            past_key_values=past,
            active_block_mask=decode_active_block_mask,
            active_attn_block_mask=decode_active_attn_block_mask,
            active_mlp_block_mask=decode_active_mlp_block_mask,
            record_mode="none",
        )
        past = out["past_key_values"]
        if step + 1 < decode_tokens - 1:
            current = continuation_ids[:, step + 1 : step + 2]
    _maybe_sync(device)
    decode_seconds = time.perf_counter() - t1
    tok_s = decode_tokens / max(decode_seconds, 1e-6)
    return TimingResult(
        prefill_seconds=prefill_seconds,
        decode_seconds=decode_seconds,
        decode_tokens_per_sec=tok_s,
        routing_overhead_seconds=route_overhead,
    )
