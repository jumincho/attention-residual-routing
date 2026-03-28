#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--target-count", type=int, required=True)
    parser.add_argument("--mode", choices=["head", "stride"], default="stride")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    lines = [line for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.target_count <= 0:
        raise ValueError("target-count must be positive")
    if len(lines) <= args.target_count:
        chosen = lines
    elif args.mode == "head":
        chosen = lines[: args.target_count]
    else:
        stride = max(len(lines) / float(args.target_count), 1.0)
        indices = []
        cursor = 0.0
        while len(indices) < args.target_count and round(cursor) < len(lines):
            idx = int(round(cursor))
            if idx >= len(lines):
                break
            indices.append(idx)
            cursor += stride
        indices = sorted(set(indices))[: args.target_count]
        while len(indices) < args.target_count:
            next_idx = indices[-1] + 1 if indices else 0
            if next_idx >= len(lines):
                break
            indices.append(next_idx)
        chosen = [lines[idx] for idx in indices[: args.target_count]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(chosen) + "\n", encoding="utf-8")
    print(f"[subset_manifest] input={input_path} output={output_path} rows={len(chosen)}", flush=True)


if __name__ == "__main__":
    main()
