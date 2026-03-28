#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attnres_routing.data import _resolve_dataset_spec  # noqa: E402
from attnres_routing.utils import resolve_hf_token  # noqa: E402
from datasets import load_dataset  # noqa: E402


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--train-count", type=int, default=10000)
    parser.add_argument("--validation-count", type=int, default=2000)
    parser.add_argument("--test-count", type=int, default=2000)
    args = parser.parse_args()

    spec = _resolve_dataset_spec(args.dataset_name)
    if not spec.get("streaming", False):
        raise ValueError(f"{args.dataset_name} is not a streaming-backed dataset alias")

    token = resolve_hf_token()
    source_split = str(spec.get("stream_source_split", "train"))
    base_offsets = spec.get("stream_offsets", {})
    target_counts = {
        "train": int(args.train_count),
        "validation": int(args.validation_count),
        "test": int(args.test_count),
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, count in target_counts.items():
        offset = int(base_offsets.get(split_name, 0))
        stream_ds = load_dataset(
            spec["path"],
            spec.get("name"),
            split=source_split,
            streaming=True,
            token=token,
        )
        rows: list[dict[str, str]] = []
        seen = 0
        for idx, row in enumerate(stream_ds):
            if idx < offset:
                continue
            text = None
            for column in ("text", "content", "document"):
                if column in row and row[column] is not None:
                    text = str(row[column])
                    break
            if text is None or not text.strip():
                continue
            rows.append({"text": text})
            seen += 1
            if seen >= count:
                break
        out_path = out_dir / f"fineweb_edu_sample10bt_{split_name}.jsonl"
        write_jsonl(out_path, rows)
        print(f"[materialize-stream-v7] split={split_name} rows={len(rows)} path={out_path}", flush=True)


if __name__ == "__main__":
    main()
