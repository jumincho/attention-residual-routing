#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_manifest(path: Path, final_split: str) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            rows.append(
                {
                    "sequence_idx": int(row["sequence_idx"]),
                    "document_uid": row["document_uid"],
                    "source_split": row["source_split"],
                    "document_idx": int(row["document_idx"]),
                    "final_split": final_split,
                }
            )
    return pd.DataFrame(rows)


def summarize(group_df: pd.DataFrame, subgroup_name: str, subgroup_value: str) -> dict:
    return {
        "subgroup_name": subgroup_name,
        "subgroup_value": subgroup_value,
        "n": int(len(group_df)),
        "delta_to_static_mean": float(group_df["actual_delta_to_static"].mean()),
        "regret_to_bank_mean": float(group_df["delta_to_bank_upper_bound"].mean()),
        "fraction_improved": float(group_df["improved_over_static"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pooled-per-seq", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    pooled = pd.read_csv(args.pooled_per_seq)
    meta = pd.read_csv(args.metadata_csv)

    manifests = []
    manifest_dir = Path(args.manifest_dir)
    for split in ("final_A", "final_B", "final_C"):
        manifests.append(load_manifest(manifest_dir / f"v8_ccnews_p256d64_lockbox_{split}.jsonl", split))
    manifest_df = pd.concat(manifests, ignore_index=True)

    merged = pooled.merge(manifest_df, on=["sequence_idx", "final_split"], how="left")
    merged = merged.merge(meta, on=["document_uid", "source_split", "document_idx"], how="left")

    top_domains = merged["domain"].fillna("unknown").value_counts().head(10).index.tolist()
    merged["domain_group"] = merged["domain"].fillna("unknown").where(merged["domain"].fillna("unknown").isin(top_domains), "other")
    merged["year"] = merged["date"].fillna("").str.slice(0, 4).replace("", "unknown")
    try:
        merged["length_bin"] = pd.qcut(merged["text_char_len"].fillna(0), q=4, labels=["q1", "q2", "q3", "q4"], duplicates="drop")
    except ValueError:
        merged["length_bin"] = "all"

    rows = [summarize(merged, "overall", "all")]
    for subgroup_name in ("final_split", "seed", "domain_group", "year", "length_bin"):
        for subgroup_value, group_df in merged.groupby(subgroup_name):
            rows.append(summarize(group_df, subgroup_name, str(subgroup_value)))

    out_df = pd.DataFrame(rows)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(out_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
