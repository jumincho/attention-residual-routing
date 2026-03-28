#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def load_manifest(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            rows.append(
                {
                    "document_uid": row["document_uid"],
                    "source_split": row["source_split"],
                    "document_idx": int(row["document_idx"]),
                    "split": row["split"],
                }
            )
    return pd.DataFrame(rows).drop_duplicates()


def load_ccnews_metadata(split: str, indices: list[int]) -> pd.DataFrame:
    ds = load_dataset("cc_news", split=split)
    rows = []
    for idx in sorted(set(indices)):
        item = ds[int(idx)]
        rows.append(
            {
                "source_split": split,
                "document_idx": int(idx),
                "title": item.get("title", ""),
                "domain": item.get("domain", ""),
                "date": item.get("date", ""),
                "description": item.get("description", ""),
                "url": item.get("url", ""),
                "text_char_len": len(item.get("text", "") or ""),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest_df = pd.concat([load_manifest(Path(p)) for p in args.manifests], ignore_index=True)
    frames = []
    for source_split, group in manifest_df.groupby("source_split"):
        frames.append(load_ccnews_metadata(source_split, group["document_idx"].tolist()))
    meta_df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["source_split", "document_idx"])
    merged = manifest_df.merge(meta_df, on=["source_split", "document_idx"], how="left")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(merged.head(10).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
