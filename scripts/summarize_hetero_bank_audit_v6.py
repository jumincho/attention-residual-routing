#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def parse_bank_tag(tag: str) -> dict[str, str | int]:
    parts = tag.split("_")
    step_part = next(part for part in parts if part.startswith("step"))
    setting = "_".join(parts[:4])
    return {
        "setting_tag": setting,
        "checkpoint_step": int(step_part.removeprefix("step")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-tags", type=str, nargs="+", required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    bank_dir = ROOT / "results" / "bank_hygiene"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_frames = []
    per_sequence_frames = []
    candidate_bank_frames = []

    for tag in args.bank_tags:
        summary_path = bank_dir / f"{tag}_summary.csv"
        per_sequence_path = bank_dir / f"{tag}_per_sequence.csv"
        candidate_bank_path = bank_dir / f"{tag}_candidate_bank.csv"
        if not summary_path.exists():
            continue
        meta = parse_bank_tag(tag)
        summary_df = pd.read_csv(summary_path)
        for key, value in meta.items():
            summary_df[key] = value
        summary_df["bank_tag"] = tag
        summary_frames.append(summary_df)

        if per_sequence_path.exists():
            per_sequence_df = pd.read_csv(per_sequence_path)
            for key, value in meta.items():
                per_sequence_df[key] = value
            per_sequence_df["bank_tag"] = tag
            per_sequence_frames.append(per_sequence_df)

        if candidate_bank_path.exists():
            candidate_bank_df = pd.read_csv(candidate_bank_path)
            for key, value in meta.items():
                candidate_bank_df[key] = value
            candidate_bank_df["bank_tag"] = tag
            candidate_bank_frames.append(candidate_bank_df)

    if summary_frames:
        pd.concat(summary_frames, ignore_index=True).to_csv(out_dir / "bank_summary.csv", index=False)
    if per_sequence_frames:
        pd.concat(per_sequence_frames, ignore_index=True).to_csv(out_dir / "bank_per_sequence.csv", index=False)
    if candidate_bank_frames:
        pd.concat(candidate_bank_frames, ignore_index=True).to_csv(out_dir / "candidate_bank.csv", index=False)


if __name__ == "__main__":
    main()
