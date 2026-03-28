#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--winners-csv", required=True)
    parser.add_argument("--final-splits", nargs="+", default=["final_A", "final_B", "final_C"])
    parser.add_argument("--template-limits", nargs="+", type=int, default=[0, 2, 4])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    winners = pd.read_csv(args.winners_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, row in winners.iterrows():
        seed = int(row["seed"])
        step = int(row["step"])
        bank_size = int(row["bank_size"])
        model_name = str(row["model_name"])
        for split in args.final_splits:
            for template_limit in args.template_limits:
                path = ROOT / "results" / "systems_routing_v7" / (
                    f"v8_systems_seed{seed}_{split}_step{step}_b{bank_size}_{model_name}_tpl{template_limit}_summary.csv"
                )
                if not path.exists():
                    continue
                df = pd.read_csv(path)
                df["seed"] = seed
                df["final_split"] = split
                df["step"] = step
                df["bank_size"] = bank_size
                df["template_limit"] = template_limit
                rows.append(df)

    all_rows = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    all_rows.to_csv(output_dir / "ccnews_v8_systems_rows.csv", index=False)

    summary_rows = []
    if not all_rows.empty:
        for template_limit, group_df in all_rows.groupby("template_limit"):
            static_df = group_df[group_df["method"] == "global_static"].copy()
            dynamic_df = group_df[group_df["method"] == "dynamic_selector"].copy()
            if static_df.empty or dynamic_df.empty:
                continue
            merged = dynamic_df.merge(
                static_df[
                    [
                        "seed",
                        "final_split",
                        "end_to_end_seconds_per_sequence",
                        "decode_seconds_per_sequence",
                        "decode_tokens_per_sec",
                    ]
                ],
                on=["seed", "final_split"],
                suffixes=("_dynamic", "_static"),
            )
            merged["latency_delta_vs_static"] = (
                merged["end_to_end_seconds_per_sequence_dynamic"] - merged["end_to_end_seconds_per_sequence_static"]
            )
            merged["decode_toks_gain_vs_static"] = (
                merged["decode_tokens_per_sec_dynamic"] - merged["decode_tokens_per_sec_static"]
            )
            summary_rows.extend(
                [
                    {
                        "template_limit": template_limit,
                        "metric": "dynamic_delta_to_static_quality",
                        "mean": float(dynamic_df["delta_to_global_static"].mean()),
                        "n": len(dynamic_df),
                    },
                    {
                        "template_limit": template_limit,
                        "metric": "dynamic_latency_delta_vs_static",
                        "mean": float(merged["latency_delta_vs_static"].mean()),
                        "n": len(merged),
                    },
                    {
                        "template_limit": template_limit,
                        "metric": "dynamic_decode_toks_gain_vs_static",
                        "mean": float(merged["decode_toks_gain_vs_static"].mean()),
                        "n": len(merged),
                    },
                    {
                        "template_limit": template_limit,
                        "metric": "dynamic_route_count",
                        "mean": float(dynamic_df["route_count"].mean()),
                        "n": len(dynamic_df),
                    },
                    {
                        "template_limit": template_limit,
                        "metric": "dynamic_selector_overhead_total",
                        "mean": float(dynamic_df["selector_overhead_seconds_total"].mean()),
                        "n": len(dynamic_df),
                    },
                ]
            )

    pd.DataFrame(summary_rows).to_csv(output_dir / "ccnews_v8_systems_template_summary.csv", index=False)
    print(all_rows.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
