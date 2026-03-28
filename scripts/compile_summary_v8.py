#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def add_rows(summary_rows: list[dict], section: str, path: Path, extra: dict | None = None) -> None:
    if not path.exists():
        return
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        payload = {"section": section}
        if extra:
            payload.update(extra)
        payload.update(row.to_dict())
        summary_rows.append(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.repo_root)
    rows: list[dict] = []

    add_rows(
        rows,
        "locked_main",
        root / "results" / "ccnews_multiseed_multisplit_v8" / "ccnews_v8_locked_pooled_summary.csv",
    )
    add_rows(
        rows,
        "locked_per_seed_split",
        root / "results" / "ccnews_multiseed_multisplit_v8" / "ccnews_v8_locked_per_seed_split.csv",
    )
    add_rows(
        rows,
        "necessity",
        root / "results" / "ccnews_necessity_multiseed_v8" / "ccnews_v8_necessity_pooled_summary.csv",
    )
    add_rows(
        rows,
        "necessity_per_seed_split",
        root / "results" / "ccnews_necessity_multiseed_v8" / "ccnews_v8_necessity_per_seed_split.csv",
    )
    add_rows(
        rows,
        "systems",
        root / "results" / "systems_speedup_v8" / "ccnews_v8_systems_template_summary.csv",
    )
    add_rows(
        rows,
        "boundary",
        root / "results" / "boundary_analysis_v8" / "v8_ccnews_subgroup_summary.csv",
    )

    winners_path = root / "results" / "ccnews_multiseed_multisplit_v8" / "v8_ccnews_dev_frozen_selection_winners.csv"
    if winners_path.exists():
        winners = pd.read_csv(winners_path)
        for _, row in winners.iterrows():
            payload = {"section": "frozen_winners"}
            payload.update(row.to_dict())
            rows.append(payload)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(pd.DataFrame(rows).head(40).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
