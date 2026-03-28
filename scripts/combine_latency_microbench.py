#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True)
    args = parser.parse_args()

    headroom_dir = ROOT / "results" / "headroom"
    plot_dir = ROOT / "results" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(headroom_dir.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched {args.pattern!r} in {headroom_dir}")

    df = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    df = df.sort_values(["block_idx", "phase", "prompt_len", "decode_len", "component"]).reset_index(drop=True)
    df.to_csv(headroom_dir / f"{args.tag}_latency_microbench.csv", index=False)

    summary = (
        df.groupby(["phase", "prompt_len", "decode_len", "component"], as_index=False)[
            ["median_seconds", "p10_seconds", "p90_seconds"]
        ]
        .mean()
        .sort_values(["phase", "prompt_len", "decode_len", "component"])
    )

    full = summary[summary["component"] == "full"][
        ["phase", "prompt_len", "decode_len", "median_seconds"]
    ].rename(columns={"median_seconds": "full_median_seconds"})
    summary = summary.merge(full, on=["phase", "prompt_len", "decode_len"], how="left")
    summary["fraction_of_full"] = summary["median_seconds"] / summary["full_median_seconds"].clip(lower=1e-9)
    summary.to_csv(headroom_dir / f"{args.tag}_latency_microbench_summary.csv", index=False)

    prompt_df = summary[summary["phase"] == "prompt"].copy()
    plt.figure(figsize=(6, 4))
    for component in ["full", "attn_only", "mlp_only"]:
        subset = prompt_df[prompt_df["component"] == component].sort_values("prompt_len")
        plt.plot(subset["prompt_len"], subset["median_seconds"] * 1e3, marker="o", label=component)
    plt.xlabel("prompt length")
    plt.ylabel("median ms per block")
    plt.tight_layout()
    plt.legend()
    plt.savefig(plot_dir / f"headroom_latency_prompt_{args.tag}.png", dpi=160)
    plt.close()

    decode_df = summary[summary["phase"] == "decode"].copy()
    for decode_len in sorted(decode_df["decode_len"].unique().tolist()):
        plt.figure(figsize=(6, 4))
        subset_decode = decode_df[decode_df["decode_len"] == decode_len]
        for component in ["full", "attn_only", "mlp_only"]:
            subset = subset_decode[subset_decode["component"] == component].sort_values("prompt_len")
            plt.plot(subset["prompt_len"], subset["median_seconds"] * 1e3, marker="o", label=component)
        plt.xlabel("prompt length")
        plt.ylabel(f"median ms per block for decode_len={decode_len}")
        plt.tight_layout()
        plt.legend()
        plt.savefig(plot_dir / f"headroom_latency_decode_{args.tag}_d{decode_len}.png", dpi=160)
        plt.close()


if __name__ == "__main__":
    main()
