#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attnres_routing.sequence_manifest import (  # noqa: E402
    build_document_window_records,
    document_uid,
    load_all_documents,
    load_tokenizer,
    save_manifest_jsonl,
)


def stable_doc_key(seed: int, document: dict[str, object]) -> str:
    payload = f"{seed}::{document_uid(document)}"
    return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()


def parse_partition_spec(values: list[str]) -> list[tuple[str, int]]:
    partitions: list[tuple[str, int]] = []
    seen: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"Partition spec must look like split=count, got {value!r}")
        split, count_text = value.split("=", 1)
        split = split.strip()
        if not split:
            raise ValueError(f"Empty split name in partition spec {value!r}")
        if split in seen:
            raise ValueError(f"Duplicate split name {split!r}")
        partitions.append((split, int(count_text)))
        seen.add(split)
    if not partitions:
        raise ValueError("At least one --partition split=count pair is required")
    return partitions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--tokenizer-name", type=str, default="openai-community/gpt2")
    parser.add_argument("--prompt-len", type=int, required=True)
    parser.add_argument("--decode-len", type=int, required=True)
    parser.add_argument("--partition", type=str, action="append", required=True)
    parser.add_argument("--max-windows-per-doc", type=int, default=16)
    parser.add_argument("--stride", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    partitions = parse_partition_spec(args.partition)

    out_dir = ROOT / "results" / "lockbox_manifests_v8"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = ROOT / "results" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    stride = args.stride if args.stride > 0 else None
    tokenizer = load_tokenizer(args.tokenizer_name)
    tokenizer.model_max_length = max(int(getattr(tokenizer, "model_max_length", 0)), 1_000_000_000)

    documents = load_all_documents(args.dataset_name, seed=args.seed)
    docs_sorted = sorted(documents, key=lambda doc: stable_doc_key(args.seed, doc))

    partition_rows: dict[str, list[dict[str, object]]] = {name: [] for name, _ in partitions}
    partition_docs: dict[str, set[str]] = {name: set() for name, _ in partitions}
    partition_targets = {name: int(target) for name, target in partitions}
    summary_rows: list[dict[str, object]] = []
    current_partition_idx = 0

    for document in docs_sorted:
        while current_partition_idx < len(partitions):
            active_name, active_target = partitions[current_partition_idx]
            if len(partition_rows[active_name]) < active_target:
                break
            current_partition_idx += 1
        if current_partition_idx >= len(partitions):
            break

        active_name, active_target = partitions[current_partition_idx]
        doc_windows = build_document_window_records(
            tokenizer=tokenizer,
            document=document,
            prompt_len=args.prompt_len,
            decode_len=args.decode_len,
            max_windows_per_doc=args.max_windows_per_doc,
            stride=stride,
        )
        if not doc_windows:
            continue

        remaining = active_target - len(partition_rows[active_name])
        if remaining <= 0:
            continue
        chosen_windows = doc_windows[:remaining]
        if not chosen_windows:
            continue
        doc_uid = str(chosen_windows[0]["document_uid"])
        partition_docs[active_name].add(doc_uid)
        base_offset = len(partition_rows[active_name])
        for local_idx, row in enumerate(chosen_windows):
            row_out = dict(row)
            row_out["sequence_idx"] = int(base_offset + local_idx)
            row_out["split"] = active_name
            row_out["partition_seed"] = int(args.seed)
            partition_rows[active_name].append(row_out)

    for split_name, target in partitions:
        manifest_path = out_dir / f"{args.tag}_{split_name}.jsonl"
        save_manifest_jsonl(partition_rows[split_name], manifest_path)
        summary_rows.append(
            {
                "tag": args.tag,
                "dataset_name": args.dataset_name,
                "split": split_name,
                "target_count": int(target),
                "created_count": len(partition_rows[split_name]),
                "documents_used": len(partition_docs[split_name]),
                "avg_windows_per_doc": (
                    float(len(partition_rows[split_name]) / max(len(partition_docs[split_name]), 1))
                ),
                "manifest_path": str(manifest_path),
            }
        )
        print(
            f"[lockbox-manifest-v8] split={split_name} created={len(partition_rows[split_name])} "
            f"docs={len(partition_docs[split_name])} path={manifest_path}",
            flush=True,
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / f"{args.tag}_manifest_summary.csv", index=False)
    with (out_dir / f"{args.tag}_manifest_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2)

    split_sizes = {row["split"]: int(row["created_count"]) for row in summary_rows}
    plt.figure(figsize=(8, 4))
    plt.bar(list(split_sizes.keys()), list(split_sizes.values()))
    plt.ylabel("sequence windows")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(plot_dir / f"lockbox_split_checks_v8_{args.tag}_window_counts.png", dpi=160)
    plt.close()

    split_docs = {row["split"]: int(row["documents_used"]) for row in summary_rows}
    plt.figure(figsize=(8, 4))
    plt.bar(list(split_docs.keys()), list(split_docs.values()))
    plt.ylabel("documents used")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(plot_dir / f"lockbox_split_checks_v8_{args.tag}_document_counts.png", dpi=160)
    plt.close()

    overlap_rows: list[dict[str, object]] = []
    split_names = [name for name, _ in partitions]
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            left_docs = partition_docs[left]
            right_docs = partition_docs[right]
            overlap_rows.append(
                {
                    "tag": args.tag,
                    "left_split": left,
                    "right_split": right,
                    "shared_documents": int(len(left_docs & right_docs)),
                    "left_documents": int(len(left_docs)),
                    "right_documents": int(len(right_docs)),
                }
            )
    overlap_df = pd.DataFrame(overlap_rows)
    overlap_df.to_csv(out_dir / f"{args.tag}_overlap_checks.csv", index=False)
    if not overlap_df.empty:
        labels = [f"{row.left_split}\nvs\n{row.right_split}" for row in overlap_df.itertuples()]
        plt.figure(figsize=(max(8, len(labels) * 0.75), 4))
        plt.bar(labels, overlap_df["shared_documents"].to_numpy(dtype=float))
        plt.ylabel("shared documents")
        plt.tight_layout()
        plt.savefig(plot_dir / f"lockbox_split_checks_v8_{args.tag}_overlap.png", dpi=160)
        plt.close()

    ledger_stub = out_dir / f"{args.tag}_selection_freeze_stub.csv"
    with ledger_stub.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "tag",
                "decision_order",
                "decision_name",
                "choice",
                "basis",
                "dev_only",
                "final_test_opened",
            ],
        )
        writer.writeheader()
        for order, decision_name in enumerate(
            [
                "checkpoint",
                "bank_size",
                "budget",
                "selector_family",
                "feature_set",
                "deployment_mode",
            ],
            start=1,
        ):
            writer.writerow(
                {
                    "tag": args.tag,
                    "decision_order": order,
                    "decision_name": decision_name,
                    "choice": "",
                    "basis": "to_fill_after_dev_selection",
                    "dev_only": 1,
                    "final_test_opened": 0,
                }
            )


if __name__ == "__main__":
    main()
