#!/usr/bin/env python
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attnres_routing.model import AttnResConfig, DecoderLM  # noqa: E402


def load_model(path: Path, device: torch.device) -> DecoderLM:
    payload = torch.load(path, map_location=device)
    model = DecoderLM(AttnResConfig.from_dict(payload["config"]["model"])).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def block_layer_indices(model: DecoderLM, block_idx: int) -> list[int]:
    return [layer_idx for layer_idx, blk in enumerate(model.layer_to_block) if blk == block_idx]


def run_prompt_component(
    model: DecoderLM,
    layer_indices: list[int],
    component: str,
    prompt_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor] | None]]:
    x = torch.randn(1, prompt_len, model.config.d_model, device=device, dtype=dtype)
    pasts: list[tuple[torch.Tensor, torch.Tensor] | None] = []
    with torch.no_grad():
        for layer_idx in layer_indices:
            layer = model.layers[layer_idx]
            if component in {"full", "attn_only"}:
                attn_out, kv = layer.attn(layer.attn_norm(x), use_cache=True)
                x = x + attn_out
                pasts.append(kv)
            else:
                pasts.append(None)
            if component in {"full", "mlp_only"}:
                x = x + layer.mlp(layer.mlp_norm(x))
    return x, pasts


def run_decode_component(
    model: DecoderLM,
    layer_indices: list[int],
    component: str,
    prompt_len: int,
    decode_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    _, pasts = run_prompt_component(model, layer_indices, component, prompt_len, dtype, device)
    current = torch.randn(1, 1, model.config.d_model, device=device, dtype=dtype)
    with torch.no_grad():
        for _ in range(decode_len):
            x = current
            next_pasts: list[tuple[torch.Tensor, torch.Tensor] | None] = []
            for layer_pos, layer_idx in enumerate(layer_indices):
                layer = model.layers[layer_idx]
                if component in {"full", "attn_only"}:
                    attn_out, kv = layer.attn(
                        layer.attn_norm(x),
                        past_key_value=pasts[layer_pos],
                        use_cache=True,
                    )
                    x = x + attn_out
                    next_pasts.append(kv)
                else:
                    next_pasts.append(None)
                if component in {"full", "mlp_only"}:
                    x = x + layer.mlp(layer.mlp_norm(x))
            pasts = next_pasts
            current = torch.randn_like(current)


def benchmark(
    fn,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    timings = []
    for _ in range(repeats):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        timings.append(time.perf_counter() - t0)
    return (
        float(statistics.median(timings)),
        float(np.quantile(timings, 0.1)),
        float(np.quantile(timings, 0.9)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt-lens", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--decode-lens", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--block-ids", type=int, nargs="*", default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tag", type=str, default="main")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(Path(args.checkpoint), device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    block_ids = args.block_ids if args.block_ids else list(range(model.config.num_blocks))
    out_dir = ROOT / "results" / "headroom"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for block_idx in block_ids:
        layer_indices = block_layer_indices(model, block_idx)
        for prompt_len in args.prompt_lens:
            for component in ["full", "attn_only", "mlp_only"]:
                median_s, p10_s, p90_s = benchmark(
                    lambda block_idx=block_idx, prompt_len=prompt_len, component=component: run_prompt_component(
                        model,
                        layer_indices,
                        component,
                        prompt_len,
                        dtype,
                        device,
                    ),
                    warmup=args.warmup,
                    repeats=args.repeats,
                    device=device,
                )
                rows.append(
                    {
                        "tag": args.tag,
                        "block_idx": block_idx,
                        "phase": "prompt",
                        "component": component,
                        "prompt_len": prompt_len,
                        "decode_len": 0,
                        "median_seconds": median_s,
                        "p10_seconds": p10_s,
                        "p90_seconds": p90_s,
                    }
                )
            for decode_len in args.decode_lens:
                for component in ["full", "attn_only", "mlp_only"]:
                    median_s, p10_s, p90_s = benchmark(
                        lambda block_idx=block_idx, prompt_len=prompt_len, decode_len=decode_len, component=component: run_decode_component(
                            model,
                            layer_indices,
                            component,
                            prompt_len,
                            decode_len,
                            dtype,
                            device,
                        ),
                        warmup=max(2, args.warmup // 2),
                        repeats=args.repeats,
                        device=device,
                    )
                    rows.append(
                        {
                            "tag": args.tag,
                            "block_idx": block_idx,
                            "phase": "decode",
                            "component": component,
                            "prompt_len": prompt_len,
                            "decode_len": decode_len,
                            "median_seconds": median_s,
                            "p10_seconds": p10_s,
                            "p90_seconds": p90_s,
                        }
                    )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{args.tag}_latency_microbench.csv", index=False)


if __name__ == "__main__":
    main()
