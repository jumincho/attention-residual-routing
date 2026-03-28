#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attnres_routing.sequence_manifest import build_window_records, save_manifest_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="wikitext103")
    parser.add_argument("--tokenizer-name", type=str, default="openai-community/gpt2")
    parser.add_argument("--prompt-len", type=int, required=True)
    parser.add_argument("--decode-len", type=int, required=True)
    parser.add_argument("--train-target", type=int, default=8192)
    parser.add_argument("--validation-target", type=int, default=1024)
    parser.add_argument("--test-target", type=int, default=1024)
    parser.add_argument("--max-windows-per-doc", type=int, default=8)
    parser.add_argument("--stride", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, required=True)
    args = parser.parse_args()

    out_dir = ROOT / "results" / "selector_data_scale"
    out_dir.mkdir(parents=True, exist_ok=True)
    total_stride = args.stride if args.stride > 0 else None

    summaries = []
    split_to_target = {
        "train": args.train_target,
        "validation": args.validation_target,
        "test": args.test_target,
    }
    for split, target in split_to_target.items():
        records, summary = build_window_records(
            dataset_name=args.dataset_name,
            tokenizer_name=args.tokenizer_name,
            source_split=split,
            prompt_len=args.prompt_len,
            decode_len=args.decode_len,
            target_count=target,
            seed=args.seed,
            max_windows_per_doc=args.max_windows_per_doc,
            stride=total_stride,
        )
        manifest_path = out_dir / f"{args.tag}_{split}.jsonl"
        save_manifest_jsonl(records, manifest_path)
        summary["tag"] = args.tag
        summary["manifest_path"] = str(manifest_path)
        summaries.append(summary)
        print(
            f"[manifest] split={split} created={summary['created_count']} "
            f"docs_used={summary['documents_used']} total_tokens={summary['total_tokens']} "
            f"path={manifest_path}",
            flush=True,
        )

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / f"{args.tag}_manifest_summary.csv", index=False)
    with (out_dir / f"{args.tag}_manifest_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)


if __name__ == "__main__":
    main()
