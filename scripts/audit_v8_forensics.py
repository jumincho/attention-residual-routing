#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is best effort
    plt = None


MANIFEST_NAMES = [
    "train",
    "validation",
    "dev_select",
    "final_A",
    "final_B",
    "final_C",
]


def sha256sum(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def mtime_iso(path: Path) -> str:
    return pd.Timestamp(path.stat().st_mtime, unit="s").tz_localize("UTC").tz_convert("Asia/Seoul").isoformat()


def mtime_epoch(path: Path) -> float:
    return float(path.stat().st_mtime)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def git_tracked(path: Path) -> bool:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(path.relative_to(ROOT))],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def fmt5(value: float) -> str:
    return f"{float(value):.5f}"


def load_manifest(path: Path) -> tuple[pd.DataFrame, set[str], set[str]]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    doc_set = set(df["document_uid"].astype(str).tolist()) if not df.empty else set()
    win_set = set((df["document_uid"].astype(str) + "::" + df["window_idx"].astype(str)).tolist()) if not df.empty else set()
    return df, doc_set, win_set


def compare_csvs(left: Path, right: Path) -> tuple[bool | None, str]:
    if not left.exists() or not right.exists():
        return None, "missing"
    left_df = pd.read_csv(left)
    right_df = pd.read_csv(right)
    exact = left_df.equals(right_df)
    return exact, f"rows={left_df.shape[0]} cols={left_df.shape[1]}"


def maybe_make_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_timeline(df: pd.DataFrame, output_path: Path) -> None:
    if plt is None or df.empty:
        return
    plot_df = df.sort_values("mtime_epoch").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, max(4, 0.45 * len(plot_df))))
    y = list(range(len(plot_df)))
    ax.scatter(plot_df["mtime_epoch"], y, s=36)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"])
    ax.set_xlabel("Epoch Seconds")
    ax.set_title("V8 Forensics Timeline")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_overlap(df: pd.DataFrame, output_path: Path) -> None:
    if plt is None or df.empty:
        return
    heat = df.pivot(index="left_split", columns="right_split", values="doc_overlap")
    heat = heat.reindex(index=MANIFEST_NAMES, columns=MANIFEST_NAMES)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(heat.fillna(0).to_numpy(), cmap="Blues")
    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels(heat.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index)
    ax.set_title("V8 Manifest Document Overlap")
    for i in range(len(heat.index)):
        for j in range(len(heat.columns)):
            val = heat.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, str(int(val)), ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(RESULTS_DIR / "v8_forensics"))
    parser.add_argument("--plots-dir", default=str(RESULTS_DIR / "plots"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    plots_dir = Path(args.plots_dir)
    maybe_make_dir(output_dir)
    maybe_make_dir(plots_dir)

    manifest_dir = RESULTS_DIR / "lockbox_manifests_v8"
    manifest_paths = {name: manifest_dir / f"v8_ccnews_p256d64_lockbox_{name}.jsonl" for name in MANIFEST_NAMES}

    hash_rows = []
    manifest_cache: dict[str, tuple[pd.DataFrame, set[str], set[str]]] = {}
    for name, path in manifest_paths.items():
        df, doc_set, win_set = load_manifest(path)
        manifest_cache[name] = (df, doc_set, win_set)
        hash_rows.append(
            {
                "artifact": name,
                "path": str(path),
                "sha256": sha256sum(path),
                "size_bytes": path.stat().st_size,
                "mtime_iso": mtime_iso(path),
                "rows": len(df),
                "documents": len(doc_set),
                "windows": len(win_set),
            }
        )

    overlap_rows = []
    for left_name in MANIFEST_NAMES:
        for right_name in MANIFEST_NAMES:
            _, left_docs, left_wins = manifest_cache[left_name]
            _, right_docs, right_wins = manifest_cache[right_name]
            overlap_rows.append(
                {
                    "left_split": left_name,
                    "right_split": right_name,
                    "left_documents": len(left_docs),
                    "right_documents": len(right_docs),
                    "doc_overlap": len(left_docs & right_docs),
                    "left_windows": len(left_wins),
                    "right_windows": len(right_wins),
                    "window_overlap": len(left_wins & right_wins),
                }
            )

    tracked_paths = [
        Path("docs/selection_ledger_v8.md"),
        Path("docs/final_report_v8.md"),
        Path("docs/paper_verdict_v8.md"),
        Path("scripts/select_ccnews_v8_frozen_configs.py"),
        Path("scripts/aggregate_ccnews_locked_v8.py"),
        Path("scripts/run_v8_remaining_pipeline.sh"),
        Path("scripts/run_ccnews_lockbox_after_followup_v8.sh"),
        Path("scripts/write_v8_docs.py"),
    ]
    tracking_rows = [
        {
            "path": str(path),
            "exists": (ROOT / path).exists(),
            "git_tracked": git_tracked(ROOT / path) if (ROOT / path).exists() else False,
        }
        for path in tracked_paths
    ]

    timeline_targets = [
        ("selection_ledger_v8", ROOT / "docs" / "selection_ledger_v8.md"),
        ("lockbox_protocol_v8", ROOT / "docs" / "lockbox_protocol_v8.md"),
        ("selection_freeze_stub_v8", manifest_dir / "v8_ccnews_p256d64_lockbox_selection_freeze_stub.csv"),
        ("selection_winners_v8", RESULTS_DIR / "ccnews_multiseed_multisplit_v8" / "v8_ccnews_dev_frozen_selection_winners.csv"),
        ("first_locked_seed42_final_A_summary", RESULTS_DIR / "regret_reduction_v8" / "v8_locked_seed42_final_A_step5500_b32_hgb_pair_attnres_attnres_summary.csv"),
        ("last_locked_seed44_final_C_summary", RESULTS_DIR / "regret_reduction_v8" / "v8_locked_seed44_final_C_step3500_b32_retrieval_rerank_top4_attnres_attnres_summary.csv"),
        ("locked_pooled_summary_v8", RESULTS_DIR / "ccnews_multiseed_multisplit_v8" / "ccnews_v8_locked_pooled_summary.csv"),
        ("boundary_subgroup_summary_v8", RESULTS_DIR / "boundary_analysis_v8" / "v8_ccnews_subgroup_summary.csv"),
        ("systems_template_summary_v8", RESULTS_DIR / "systems_speedup_v8" / "ccnews_v8_systems_template_summary.csv"),
        ("necessity_pooled_summary_v8", RESULTS_DIR / "ccnews_necessity_multiseed_v8" / "ccnews_v8_necessity_pooled_summary.csv"),
        ("summary_v8", RESULTS_DIR / "summary_v8.csv"),
        ("paper_verdict_v8", ROOT / "docs" / "paper_verdict_v8.md"),
        ("final_report_v8", ROOT / "docs" / "final_report_v8.md"),
    ]
    timeline_rows = []
    for label, path in timeline_targets:
        if path.exists():
            timeline_rows.append(
                {
                    "label": label,
                    "path": str(path),
                    "mtime_iso": mtime_iso(path),
                    "mtime_epoch": mtime_epoch(path),
                }
            )

    recomputed_pairs = [
        (
            "locked_pooled_summary",
            RESULTS_DIR / "ccnews_multiseed_multisplit_v8" / "ccnews_v8_locked_pooled_summary.csv",
            RESULTS_DIR / "v8_forensics" / "recomputed" / "locked" / "ccnews_v8_locked_pooled_summary.csv",
        ),
        (
            "locked_per_seed_split",
            RESULTS_DIR / "ccnews_multiseed_multisplit_v8" / "ccnews_v8_locked_per_seed_split.csv",
            RESULTS_DIR / "v8_forensics" / "recomputed" / "locked" / "ccnews_v8_locked_per_seed_split.csv",
        ),
        (
            "necessity_pooled_summary",
            RESULTS_DIR / "ccnews_necessity_multiseed_v8" / "ccnews_v8_necessity_pooled_summary.csv",
            RESULTS_DIR / "v8_forensics" / "recomputed" / "necessity" / "ccnews_v8_necessity_pooled_summary.csv",
        ),
        (
            "necessity_per_seed_split",
            RESULTS_DIR / "ccnews_necessity_multiseed_v8" / "ccnews_v8_necessity_per_seed_split.csv",
            RESULTS_DIR / "v8_forensics" / "recomputed" / "necessity" / "ccnews_v8_necessity_per_seed_split.csv",
        ),
        (
            "systems_template_summary",
            RESULTS_DIR / "systems_speedup_v8" / "ccnews_v8_systems_template_summary.csv",
            RESULTS_DIR / "v8_forensics" / "recomputed" / "systems" / "ccnews_v8_systems_template_summary.csv",
        ),
        (
            "summary_v8",
            RESULTS_DIR / "summary_v8.csv",
            RESULTS_DIR / "v8_forensics" / "recomputed" / "summary_v8_recompiled.csv",
        ),
    ]
    reagg_rows = []
    for name, original, recomputed in recomputed_pairs:
        exact, notes = compare_csvs(original, recomputed)
        reagg_rows.append(
            {
                "artifact": name,
                "original_path": str(original),
                "recomputed_path": str(recomputed),
                "exact_match": exact,
                "notes": notes,
            }
        )

    locked_pooled = pd.read_csv(RESULTS_DIR / "ccnews_multiseed_multisplit_v8" / "ccnews_v8_locked_pooled_summary.csv")
    necessity_pooled = pd.read_csv(RESULTS_DIR / "ccnews_necessity_multiseed_v8" / "ccnews_v8_necessity_pooled_summary.csv")
    systems_summary = pd.read_csv(RESULTS_DIR / "systems_speedup_v8" / "ccnews_v8_systems_template_summary.csv")
    locked_per_seed = pd.read_csv(RESULTS_DIR / "ccnews_multiseed_multisplit_v8" / "ccnews_v8_locked_per_seed_split.csv")

    delta_row = locked_pooled[locked_pooled["metric"] == "pooled_delta_to_static"].iloc[0]
    regret_row = locked_pooled[locked_pooled["metric"] == "pooled_regret_to_bank"].iloc[0]
    frac_row = locked_pooled[locked_pooled["metric"] == "pooled_fraction_improved"].iloc[0]

    attn_dyn = necessity_pooled[(necessity_pooled["family"] == "attnres_dynamic") & (necessity_pooled["metric"] == "delta_to_static")].iloc[0]
    attn_hidden = necessity_pooled[(necessity_pooled["family"] == "attnres_hidden") & (necessity_pooled["metric"] == "delta_to_static")].iloc[0]
    std_hidden = necessity_pooled[(necessity_pooled["family"] == "standard_hidden") & (necessity_pooled["metric"] == "delta_to_static")].iloc[0]

    best_tpl = systems_summary.pivot_table(index="template_limit", columns="metric", values="mean", aggfunc="first").reset_index()
    best_tpl = best_tpl.sort_values("dynamic_latency_delta_vs_static").iloc[0]

    final_report_text = read_text(ROOT / "docs" / "final_report_v8.md")
    paper_verdict_text = read_text(ROOT / "docs" / "paper_verdict_v8.md")

    expected_final_delta = f"`{fmt5(delta_row['mean'])} [{fmt5(delta_row['ci_low'])}, {fmt5(delta_row['ci_high'])}]`"
    expected_final_regret = f"`{fmt5(regret_row['mean'])} [{fmt5(regret_row['ci_low'])}, {fmt5(regret_row['ci_high'])}]`"
    expected_final_frac = f"`{fmt5(frac_row['mean'])} [{fmt5(frac_row['ci_low'])}, {fmt5(frac_row['ci_high'])}]`"

    doc_rows = [
        {
            "doc_path": "docs/final_report_v8.md",
            "check_name": "locked_delta_line",
            "expected": expected_final_delta,
            "passed": expected_final_delta in final_report_text,
        },
        {
            "doc_path": "docs/final_report_v8.md",
            "check_name": "locked_regret_line",
            "expected": expected_final_regret,
            "passed": expected_final_regret in final_report_text,
        },
        {
            "doc_path": "docs/final_report_v8.md",
            "check_name": "locked_fraction_line",
            "expected": expected_final_frac,
            "passed": expected_final_frac in final_report_text,
        },
        {
            "doc_path": "docs/final_report_v8.md",
            "check_name": "necessity_attnres_dynamic",
            "expected": f"`{fmt5(attn_dyn['mean'])}`",
            "passed": f"`{fmt5(attn_dyn['mean'])}`" in final_report_text,
        },
        {
            "doc_path": "docs/final_report_v8.md",
            "check_name": "necessity_attnres_hidden",
            "expected": f"`{fmt5(attn_hidden['mean'])}`",
            "passed": f"`{fmt5(attn_hidden['mean'])}`" in final_report_text,
        },
        {
            "doc_path": "docs/final_report_v8.md",
            "check_name": "necessity_standard_hidden",
            "expected": f"`{fmt5(std_hidden['mean'])}`",
            "passed": f"`{fmt5(std_hidden['mean'])}`" in final_report_text,
        },
        {
            "doc_path": "docs/final_report_v8.md",
            "check_name": "systems_latency_delta",
            "expected": f"`{fmt5(best_tpl['dynamic_latency_delta_vs_static'])}`",
            "passed": f"`{fmt5(best_tpl['dynamic_latency_delta_vs_static'])}`" in final_report_text,
        },
        {
            "doc_path": "docs/final_report_v8.md",
            "check_name": "systems_quality_delta",
            "expected": f"`{fmt5(best_tpl['dynamic_delta_to_static_quality'])}`",
            "passed": f"`{fmt5(best_tpl['dynamic_delta_to_static_quality'])}`" in final_report_text,
        },
        {
            "doc_path": "docs/paper_verdict_v8.md",
            "check_name": "locked_delta_line",
            "expected": expected_final_delta,
            "passed": expected_final_delta in paper_verdict_text,
        },
        {
            "doc_path": "docs/paper_verdict_v8.md",
            "check_name": "v8_pooled_regret_line",
            "expected": f"`{fmt5(regret_row['mean'])}`",
            "passed": f"`{fmt5(regret_row['mean'])}`" in paper_verdict_text,
        },
    ]

    for _, row in locked_per_seed.iterrows():
        expected_tokens = [
            f"| {int(row['seed'])} |",
            f"| {row['final_split']} |",
            f"| {int(row['step'])} |",
            f"| {int(row['bank_size'])} |",
            f"| {row['feature_mode']} |",
            f"| {row['model_name']} |",
            f"{row['delta_to_static']}",
            f"{row['regret_to_bank']}",
            f"{row['fraction_improved']}",
            f"{row['oracle_in_bank_match']}",
        ]
        passed = any(all(token in line for token in expected_tokens) for line in paper_verdict_text.splitlines())
        doc_rows.append(
            {
                "doc_path": "docs/paper_verdict_v8.md",
                "check_name": f"per_seed_row_seed{int(row['seed'])}_{row['final_split']}",
                "expected": "row_tokens_present",
                "passed": passed,
            }
        )

    pipeline_log = read_text(RESULTS_DIR / "ccnews_multiseed_multisplit_v8" / "v8_pipeline.log")
    finalize_log = read_text(RESULTS_DIR / "ccnews_multiseed_multisplit_v8" / "v8_finalize_after_locked_eval.log")
    bash_history = read_text(Path.home() / ".bash_history") if (Path.home() / ".bash_history").exists() else ""

    winners_path = RESULTS_DIR / "ccnews_multiseed_multisplit_v8" / "v8_ccnews_dev_frozen_selection_winners.csv"
    first_final_path = RESULTS_DIR / "regret_reduction_v8" / "v8_locked_seed42_final_A_step5500_b32_hgb_pair_attnres_attnres_summary.csv"
    final_summary_path = RESULTS_DIR / "summary_v8.csv"

    finding_rows = [
        {
            "finding": "manifests_document_disjoint",
            "severity": "info",
            "passed": all(row["doc_overlap"] == 0 for row in overlap_rows if row["left_split"] != row["right_split"]),
            "evidence": "all non-diagonal manifest doc overlaps are zero",
        },
        {
            "finding": "selection_winners_before_first_locked_final",
            "severity": "info",
            "passed": mtime_epoch(winners_path) < mtime_epoch(first_final_path),
            "evidence": f"winners={mtime_iso(winners_path)} first_final={mtime_iso(first_final_path)}",
        },
        {
            "finding": "summary_v8_reaggregates_exactly",
            "severity": "info",
            "passed": all(row["exact_match"] is True for row in reagg_rows if row["artifact"] != "summary_v8") and compare_csvs(final_summary_path, RESULTS_DIR / 'v8_forensics' / 'recomputed' / 'summary_v8_recompiled.csv')[0] is True,
            "evidence": "locked/necessity/systems/summary exact-match canonical outputs",
        },
        {
            "finding": "pipeline_log_contains_abort_marker",
            "severity": "warn",
            "passed": "unsupported seed output_tag" not in pipeline_log,
            "evidence": "unsupported seed output_tag" if "unsupported seed output_tag" in pipeline_log else "not found",
        },
        {
            "finding": "finalize_log_contains_aggregation_step",
            "severity": "warn",
            "passed": "aggregating locked final results" in finalize_log,
            "evidence": "aggregation marker present" if "aggregating locked final results" in finalize_log else "finalize watcher log stops at wait loop",
        },
        {
            "finding": "selection_freeze_stub_filled",
            "severity": "warn",
            "passed": "to_fill_after_dev_selection" not in read_text(manifest_dir / "v8_ccnews_p256d64_lockbox_selection_freeze_stub.csv"),
            "evidence": "selection_freeze_stub still placeholder" if "to_fill_after_dev_selection" in read_text(manifest_dir / "v8_ccnews_p256d64_lockbox_selection_freeze_stub.csv") else "filled",
        },
        {
            "finding": "key_v8_artifacts_git_tracked",
            "severity": "warn",
            "passed": all(row["git_tracked"] for row in tracking_rows),
            "evidence": "one or more key scripts/docs are untracked",
        },
        {
            "finding": "bash_history_contains_v8_commands",
            "severity": "warn",
            "passed": bool(re.search(r"run_v8|_v8|lockbox_manifests_v8|selection_ledger_v8|run_ccnews_.*_v8", bash_history)),
            "evidence": "bash_history has no recoverable V8 protocol commands" if not re.search(r"run_v8|_v8|lockbox_manifests_v8|selection_ledger_v8|run_ccnews_.*_v8", bash_history) else "bash_history contains V8-related tokens",
        },
    ]

    pd.DataFrame(hash_rows).to_csv(output_dir / "v8_manifest_hashes.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(output_dir / "v8_manifest_overlap.csv", index=False)
    pd.DataFrame(tracking_rows).to_csv(output_dir / "v8_git_tracking.csv", index=False)
    pd.DataFrame(timeline_rows).to_csv(output_dir / "v8_artifact_timeline.csv", index=False)
    pd.DataFrame(reagg_rows).to_csv(output_dir / "v8_reaggregation_check.csv", index=False)
    pd.DataFrame(doc_rows).to_csv(output_dir / "v8_doc_consistency.csv", index=False)
    pd.DataFrame(finding_rows).to_csv(output_dir / "v8_protocol_findings.csv", index=False)

    plot_timeline(pd.DataFrame(timeline_rows), plots_dir / "v8_forensics_timeline.png")
    plot_overlap(pd.DataFrame(overlap_rows), plots_dir / "v8_forensics_manifest_overlap.png")

    print(pd.DataFrame(finding_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
