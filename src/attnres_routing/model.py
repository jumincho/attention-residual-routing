from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .normalizers import depth_normalize


@dataclass
class AttnResConfig:
    vocab_size: int = 50257
    max_seq_len: int = 1024
    d_model: int = 768
    num_heads: int = 12
    num_layers: int = 24
    mlp_hidden_dim: int = 2048
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    rope_base: float = 10000.0
    residual_mode: str = "standard"
    num_blocks: int = 8
    depth_normalizer: str = "softmax"
    depth_temperature: float = 1.0
    topk_softmax_k: int = 2
    tie_embeddings: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AttnResConfig":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mean_token_norm(x: torch.Tensor) -> torch.Tensor:
    return x.float().norm(dim=-1).mean()


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Rotary dim must be even, got {dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        self._cos_cached: Optional[torch.Tensor] = None
        self._sin_cached: Optional[torch.Tensor] = None

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(dtype=dtype)
        sin = emb.sin().to(dtype=dtype)
        return cos, sin

    def get_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._cos_cached.size(0) < seq_len
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
        ):
            cos, sin = self._build_cache(seq_len, device, dtype)
            self._cos_cached = cos
            self._sin_cached = sin
        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    out = torch.stack((-x2, x1), dim=-1)
    return out.flatten(start_dim=-2)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos[offset : offset + q.size(1)].unsqueeze(0).unsqueeze(2)
    sin = sin[offset : offset + q.size(1)].unsqueeze(0).unsqueeze(2)
    q_out = (q * cos) + (_rotate_half(q) * sin)
    k_cos = cos
    k_sin = sin
    k_out = (k * k_cos) + (_rotate_half(k) * k_sin)
    return q_out, k_out


class CausalSelfAttention(nn.Module):
    def __init__(self, config: AttnResConfig) -> None:
        super().__init__()
        if config.d_model % config.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.attn_dropout = config.attn_dropout
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.resid_dropout = nn.Dropout(config.resid_dropout)
        self.rotary = RotaryEmbedding(self.head_dim, config.max_seq_len, config.rope_base)

    def _causal_mask(self, q_len: int, kv_len: int, past_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        mask = torch.full((q_len, kv_len), float("-inf"), device=device, dtype=dtype)
        return torch.triu(mask, diagonal=1 + past_len)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        bsz, seq_len, dim = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim)

        past_len = 0 if past_key_value is None else past_key_value[0].size(2)
        cos, sin = self.rotary.get_cos_sin(past_len + seq_len, hidden_states.device, hidden_states.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, offset=past_len)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        attn_mask = None
        is_causal = past_len == 0
        if not is_causal:
            attn_mask = self._causal_mask(seq_len, k.size(2), past_len, hidden_states.device, hidden_states.dtype)
            is_causal = False

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        new_kv = (k, v) if use_cache else None
        return out, new_kv


class SwiGLU(nn.Module):
    def __init__(self, config: AttnResConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.mlp_hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.mlp_hidden_dim, bias=False)
        self.down_proj = nn.Linear(config.mlp_hidden_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.resid_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        return self.dropout(x)


class DepthMix(nn.Module):
    def __init__(self, config: AttnResConfig) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(config.d_model))
        self.key_norm = nn.RMSNorm(config.d_model)
        self.mode = config.depth_normalizer
        self.temperature = config.depth_temperature
        self.topk = config.topk_softmax_k

    def forward(
        self,
        sources: torch.Tensor,
        source_ids: torch.Tensor,
        record_mode: str = "none",
    ) -> tuple[torch.Tensor, Optional[dict[str, Any]]]:
        keys = self.key_norm(sources)
        logits = torch.einsum("d,sbtd->sbt", self.query, keys)
        weights = depth_normalize(
            logits,
            mode=self.mode,
            dim=0,
            temperature=self.temperature,
            topk=self.topk,
        )
        mixed = torch.einsum("sbt,sbtd->btd", weights, sources)

        if record_mode == "none":
            return mixed, None

        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=0)
        support_size = (weights > 1e-4).float().sum(dim=0)
        info: dict[str, Any] = {
            "source_ids": source_ids.detach().cpu(),
            "entropy": entropy.mean().detach().cpu(),
            "support_size": support_size.mean().detach().cpu(),
            "entropy_tensor": entropy.mean(),
        }
        if record_mode == "full":
            info["weights"] = weights.detach().cpu()
        else:
            usage = torch.zeros(int(source_ids.max().item()) + 1, device=weights.device, dtype=weights.dtype)
            for idx, source_id in enumerate(source_ids.tolist()):
                usage[source_id] += weights[idx].sum()
            info["usage"] = usage.detach().cpu()
            info["usage_tensor"] = usage
        return mixed, info


class TransformerLayer(nn.Module):
    def __init__(self, config: AttnResConfig) -> None:
        super().__init__()
        self.attn_norm = nn.RMSNorm(config.d_model)
        self.mlp_norm = nn.RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.mlp = SwiGLU(config)
        self.attn_res = DepthMix(config)
        self.mlp_res = DepthMix(config)


def _stack_sources(sources: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(sources, dim=0)


class DecoderLM(nn.Module):
    def __init__(self, config: AttnResConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.num_layers)])
        self.final_norm = nn.RMSNorm(config.d_model)
        self.final_res = DepthMix(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embeddings.weight
        self.block_sizes = self._build_block_sizes(config.num_layers, config.num_blocks)
        self.layer_to_block = []
        for block_idx, size in enumerate(self.block_sizes):
            self.layer_to_block.extend([block_idx] * size)
        self.apply(self._init_weights)

    @staticmethod
    def _build_block_sizes(num_layers: int, num_blocks: int) -> list[int]:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        base = num_layers // num_blocks
        rem = num_layers % num_blocks
        return [base + (1 if idx < rem else 0) for idx in range(num_blocks)]

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _finalize_loss(self, logits: torch.Tensor, labels: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if labels is None:
            return None
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    def _empty_stats(self, batch_size: int, device: torch.device) -> dict[str, Any]:
        return {
            "activation_norms": [],
            "depth_entropies": [],
            "depth_support_sizes": [],
            "source_usage": torch.zeros(self.config.num_blocks + 1, device=device),
            "entropy_tensors": [],
            "usage_tensors": [],
            "records": [],
            "num_sequences": batch_size,
        }

    def _update_stats(
        self,
        stats: dict[str, Any],
        record: Optional[dict[str, Any]],
        layer_idx: int,
        sublayer: str,
        block_idx: int,
        record_mode: str,
    ) -> None:
        if record is None:
            return
        stats["depth_entropies"].append(float(record["entropy"]))
        stats["depth_support_sizes"].append(float(record["support_size"]))
        if "entropy_tensor" in record:
            stats["entropy_tensors"].append(record["entropy_tensor"])
        if "usage" in record:
            usage = record["usage"].to(stats["source_usage"].device)
            if usage.numel() < stats["source_usage"].numel():
                padded = torch.zeros_like(stats["source_usage"])
                padded[: usage.numel()] = usage
                usage = padded
            stats["source_usage"] += usage
        if "usage_tensor" in record:
            usage_tensor = record["usage_tensor"]
            if usage_tensor.numel() < stats["source_usage"].numel():
                padded = torch.zeros_like(stats["source_usage"])
                padded[: usage_tensor.numel()] = usage_tensor
                usage_tensor = padded
            stats["usage_tensors"].append(usage_tensor)
        if record_mode == "full":
            stats["records"].append(
                {
                    "layer_idx": layer_idx,
                    "sublayer": sublayer,
                    "block_idx": block_idx,
                    "source_ids": record["source_ids"],
                    "weights": record["weights"],
                }
            )

    @staticmethod
    def _resolve_sublayer_activity(
        block_idx: int,
        active_block_mask: Optional[torch.Tensor],
        active_attn_block_mask: Optional[torch.Tensor],
        active_mlp_block_mask: Optional[torch.Tensor],
    ) -> tuple[bool, bool]:
        block_active = True if active_block_mask is None else bool(active_block_mask[block_idx].item())
        attn_active = block_active and (
            True if active_attn_block_mask is None else bool(active_attn_block_mask[block_idx].item())
        )
        mlp_active = block_active and (
            True if active_mlp_block_mask is None else bool(active_mlp_block_mask[block_idx].item())
        )
        return attn_active, mlp_active

    def _baseline_forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor],
        past_key_values: Optional[list[Optional[tuple[torch.Tensor, torch.Tensor]]]],
        use_cache: bool,
        active_block_mask: Optional[torch.Tensor],
        active_attn_block_mask: Optional[torch.Tensor],
        active_mlp_block_mask: Optional[torch.Tensor],
        record_mode: str,
        return_block_states: bool,
    ) -> dict[str, Any]:
        x = self.token_embeddings(input_ids)
        next_past: list[Optional[tuple[torch.Tensor, torch.Tensor]]] = []
        stats = self._empty_stats(input_ids.size(0), input_ids.device)
        block_states = [x] if return_block_states else None
        prev_block_idx: Optional[int] = None
        for layer_idx, layer in enumerate(self.layers):
            block_idx = self.layer_to_block[layer_idx]
            attn_active, mlp_active = self._resolve_sublayer_activity(
                block_idx,
                active_block_mask,
                active_attn_block_mask,
                active_mlp_block_mask,
            )
            if not attn_active and not mlp_active:
                next_past.append(None if not use_cache else past_key_values[layer_idx] if past_key_values else None)
                stats["activation_norms"].append(float(_mean_token_norm(x)))
                prev_block_idx = block_idx
                continue
            kv = None if past_key_values is None else past_key_values[layer_idx]
            if attn_active:
                attn_out, new_kv = layer.attn(layer.attn_norm(x), past_key_value=kv, use_cache=use_cache)
                x = x + attn_out
            else:
                new_kv = kv if use_cache else None
            if mlp_active:
                x = x + layer.mlp(layer.mlp_norm(x))
            next_past.append(new_kv)
            stats["activation_norms"].append(float(_mean_token_norm(x)))
            next_block_idx = self.layer_to_block[layer_idx + 1] if (layer_idx + 1) < len(self.layers) else None
            if return_block_states and (next_block_idx is None or next_block_idx != block_idx):
                block_states.append(x)
            prev_block_idx = block_idx
        logits = self.lm_head(self.final_norm(x))
        outputs = {
            "logits": logits,
            "loss": self._finalize_loss(logits, labels),
            "past_key_values": next_past if use_cache else None,
            "stats": stats,
        }
        if return_block_states:
            outputs["block_states"] = block_states
            outputs["final_hidden"] = x
        return outputs

    def _attnres_forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor],
        past_key_values: Optional[list[Optional[tuple[torch.Tensor, torch.Tensor]]]],
        use_cache: bool,
        active_block_mask: Optional[torch.Tensor],
        active_attn_block_mask: Optional[torch.Tensor],
        active_mlp_block_mask: Optional[torch.Tensor],
        record_mode: str,
        return_block_states: bool,
    ) -> dict[str, Any]:
        embedding = self.token_embeddings(input_ids)
        completed_blocks: list[torch.Tensor] = [embedding]
        completed_ids: list[int] = [0]
        current_block_sum: Optional[torch.Tensor] = None
        current_block_id: Optional[int] = None
        next_past: list[Optional[tuple[torch.Tensor, torch.Tensor]]] = []
        stats = self._empty_stats(input_ids.size(0), input_ids.device)
        prev_block_idx = None
        current_active = True
        block_states = [embedding] if return_block_states else None

        for layer_idx, layer in enumerate(self.layers):
            block_idx = self.layer_to_block[layer_idx]
            if prev_block_idx is None or block_idx != prev_block_idx:
                if current_block_sum is not None and current_block_id is not None:
                    completed_blocks.append(current_block_sum)
                    completed_ids.append(current_block_id)
                    if return_block_states:
                        block_states.append(current_block_sum)
                current_block_sum = None
                current_block_id = block_idx + 1
                attn_active, mlp_active = self._resolve_sublayer_activity(
                    block_idx,
                    active_block_mask,
                    active_attn_block_mask,
                    active_mlp_block_mask,
                )
                current_active = attn_active or mlp_active
                prev_block_idx = block_idx

            if not current_active:
                next_past.append(None if not use_cache else past_key_values[layer_idx] if past_key_values else None)
                stats["activation_norms"].append(float(_mean_token_norm(completed_blocks[-1])))
                continue

            attn_sources = completed_blocks if current_block_sum is None else completed_blocks + [current_block_sum]
            attn_ids = torch.tensor(
                completed_ids if current_block_sum is None else completed_ids + [current_block_id],
                device=input_ids.device,
                dtype=torch.long,
            )
            h_attn, attn_record = layer.attn_res(_stack_sources(attn_sources), attn_ids, record_mode=record_mode)
            kv = None if past_key_values is None else past_key_values[layer_idx]
            if attn_active:
                attn_out, new_kv = layer.attn(layer.attn_norm(h_attn), past_key_value=kv, use_cache=use_cache)
                partial_after_attn = attn_out if current_block_sum is None else current_block_sum + attn_out
            else:
                new_kv = kv if use_cache else None
                partial_after_attn = h_attn
            self._update_stats(stats, attn_record, layer_idx, "attn", block_idx, record_mode)

            mlp_sources = completed_blocks + [partial_after_attn]
            mlp_ids = torch.tensor(completed_ids + [current_block_id], device=input_ids.device, dtype=torch.long)
            h_mlp, mlp_record = layer.mlp_res(_stack_sources(mlp_sources), mlp_ids, record_mode=record_mode)
            if mlp_active:
                mlp_out = layer.mlp(layer.mlp_norm(h_mlp))
                current_block_sum = partial_after_attn + mlp_out
            else:
                current_block_sum = h_mlp
            next_past.append(new_kv)
            self._update_stats(stats, mlp_record, layer_idx, "mlp", block_idx, record_mode)
            stats["activation_norms"].append(float(_mean_token_norm(current_block_sum)))

        final_sources = completed_blocks if current_block_sum is None else completed_blocks + [current_block_sum]
        final_ids = torch.tensor(
            completed_ids if current_block_sum is None else completed_ids + [current_block_id],
            device=input_ids.device,
            dtype=torch.long,
        )
        final_hidden, final_record = self.final_res(_stack_sources(final_sources), final_ids, record_mode=record_mode)
        self._update_stats(stats, final_record, self.config.num_layers, "final", self.config.num_blocks - 1, record_mode)
        logits = self.lm_head(self.final_norm(final_hidden))
        outputs = {
            "logits": logits,
            "loss": self._finalize_loss(logits, labels),
            "past_key_values": next_past if use_cache else None,
            "stats": stats,
        }
        if return_block_states:
            if current_block_sum is not None:
                block_states.append(current_block_sum)
            outputs["block_states"] = block_states
            outputs["final_hidden"] = final_hidden
        return outputs

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[list[Optional[tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: bool = False,
        active_block_mask: Optional[torch.Tensor] = None,
        active_attn_block_mask: Optional[torch.Tensor] = None,
        active_mlp_block_mask: Optional[torch.Tensor] = None,
        record_mode: str = "none",
        return_block_states: bool = False,
    ) -> dict[str, Any]:
        if self.config.residual_mode == "standard":
            return self._baseline_forward(
                input_ids=input_ids,
                labels=labels,
                past_key_values=past_key_values,
                use_cache=use_cache,
                active_block_mask=active_block_mask,
                active_attn_block_mask=active_attn_block_mask,
                active_mlp_block_mask=active_mlp_block_mask,
                record_mode=record_mode,
                return_block_states=return_block_states,
            )
        if self.config.residual_mode == "block_attnres":
            return self._attnres_forward(
                input_ids=input_ids,
                labels=labels,
                past_key_values=past_key_values,
                use_cache=use_cache,
                active_block_mask=active_block_mask,
                active_attn_block_mask=active_attn_block_mask,
                active_mlp_block_mask=active_mlp_block_mask,
                record_mode=record_mode,
                return_block_states=return_block_states,
            )
        raise ValueError(f"Unsupported residual mode: {self.config.residual_mode}")
