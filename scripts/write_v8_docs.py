#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def fmt(x: float | int | str | None, digits: int = 5) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, str):
        return x
    return f"{float(x):.{digits}f}"


def read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def metric_row(df: pd.DataFrame | None, metric: str, family: str | None = None) -> pd.Series | None:
    if df is None or df.empty:
        return None
    subset = df[df["metric"] == metric].copy()
    if family is not None and "family" in subset.columns:
        subset = subset[subset["family"] == family]
    if subset.empty:
        return None
    return subset.iloc[0]


def lm_best_step(metrics_path: Path) -> int | None:
    df = read_csv(metrics_path)
    if df is None or "val_loss" not in df.columns:
        return None
    keep = df[df["step"].isin([2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000])].copy()
    keep = keep[pd.notna(keep["val_loss"])]
    if keep.empty:
        return None
    return int(keep.sort_values("val_loss").iloc[0]["step"])


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n")


def table_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        headers = [str(col) for col in df.columns]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for _, row in df.iterrows():
            vals = []
            for val in row.tolist():
                if pd.isna(val):
                    vals.append("")
                else:
                    vals.append(str(val))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args()

    root = Path(args.repo_root)
    docs = root / "docs"
    results = root / "results"

    winners = read_csv(results / "ccnews_multiseed_multisplit_v8" / "v8_ccnews_dev_frozen_selection_winners.csv")
    locked_pooled = read_csv(results / "ccnews_multiseed_multisplit_v8" / "ccnews_v8_locked_pooled_summary.csv")
    locked_per_seed = read_csv(results / "ccnews_multiseed_multisplit_v8" / "ccnews_v8_locked_per_seed_split.csv")
    necessity_pooled = read_csv(results / "ccnews_necessity_multiseed_v8" / "ccnews_v8_necessity_pooled_summary.csv")
    necessity_per_seed = read_csv(results / "ccnews_necessity_multiseed_v8" / "ccnews_v8_necessity_per_seed_split.csv")
    systems_summary = read_csv(results / "systems_speedup_v8" / "ccnews_v8_systems_template_summary.csv")
    subgroup_summary = read_csv(results / "boundary_analysis_v8" / "v8_ccnews_subgroup_summary.csv")

    delta_row = metric_row(locked_pooled, "pooled_delta_to_static")
    regret_row = metric_row(locked_pooled, "pooled_regret_to_bank")
    frac_row = metric_row(locked_pooled, "pooled_fraction_improved")

    attn_dyn_delta = metric_row(necessity_pooled, "delta_to_static", "attnres_dynamic")
    attn_hidden_delta = metric_row(necessity_pooled, "delta_to_static", "attnres_hidden")
    std_hidden_delta = metric_row(necessity_pooled, "delta_to_static", "standard_hidden")

    best_tpl = None
    if systems_summary is not None and not systems_summary.empty:
        tpl_df = systems_summary.pivot_table(index="template_limit", columns="metric", values="mean", aggfunc="first").reset_index()
        if "dynamic_latency_delta_vs_static" in tpl_df.columns:
            best_tpl = tpl_df.sort_values("dynamic_latency_delta_vs_static").iloc[0]

    overall_subgroup = None
    if subgroup_summary is not None and not subgroup_summary.empty:
        overall = subgroup_summary[(subgroup_summary["subgroup_name"] == "overall") & (subgroup_summary["subgroup_value"] == "all")]
        if not overall.empty:
            overall_subgroup = overall.iloc[0]

    winner_lines = []
    checkpoint_lines = []
    scorer_lines = []
    if winners is not None and not winners.empty:
        for _, row in winners.iterrows():
            seed = int(row["seed"])
            step = int(row["step"])
            bank_size = int(row["bank_size"])
            model_name = str(row["model_name"])
            feature_mode = str(row["feature_mode"])
            winner_lines.append(f"- seed `{seed}`: `step{step} / bank{bank_size} / {model_name} / {feature_mode}`")

            metrics_map = {
                42: root / "results" / "scale24x512_ccnews_attnres_dense_v7" / "metrics.csv",
                43: root / "results" / "scale24x512_ccnews_attnres_seed43_v8" / "metrics.csv",
                44: root / "results" / "scale24x512_ccnews_attnres_seed44_v8" / "metrics.csv",
            }
            lm_step = lm_best_step(metrics_map.get(seed, Path()))
            checkpoint_lines.append(f"- seed `{seed}`: route-best `step{step}`, LM-best `step{lm_step}`")
            scorer_lines.append(f"- seed `{seed}` selected `{model_name}` on `bank{bank_size}`")

    locked_table = ""
    if locked_per_seed is not None and not locked_per_seed.empty:
        locked_table = table_to_markdown(locked_per_seed)

    necessity_table = ""
    if necessity_per_seed is not None and not necessity_per_seed.empty:
        necessity_table = table_to_markdown(necessity_per_seed)

    systems_table = ""
    if systems_summary is not None and not systems_summary.empty:
        systems_table = table_to_markdown(systems_summary)

    subgroup_table = ""
    if subgroup_summary is not None and not subgroup_summary.empty:
        top = subgroup_summary.sort_values("n", ascending=False).head(20)
        subgroup_table = table_to_markdown(top)

    write(
        docs / "ccnews_multiseed_multisplit_v8.md",
        f"""
# CCNews Multiseed Multisplit V8

Fresh V8 selection was frozen on `dev_select` and then evaluated on new `final_A/B/C`.

## Frozen Winners
{chr(10).join(winner_lines) if winner_lines else "- pending"}

## Locked Pooled Main Result

- pooled delta to static: `{fmt(delta_row['mean']) if delta_row is not None else 'n/a'} [{fmt(delta_row['ci_low']) if delta_row is not None else 'n/a'}, {fmt(delta_row['ci_high']) if delta_row is not None else 'n/a'}]`
- pooled regret to bank: `{fmt(regret_row['mean']) if regret_row is not None else 'n/a'} [{fmt(regret_row['ci_low']) if regret_row is not None else 'n/a'}, {fmt(regret_row['ci_high']) if regret_row is not None else 'n/a'}]`
- pooled fraction improved: `{fmt(frac_row['mean']) if frac_row is not None else 'n/a'} [{fmt(frac_row['ci_low']) if frac_row is not None else 'n/a'}, {fmt(frac_row['ci_high']) if frac_row is not None else 'n/a'}]`

## Per Seed / Final Split

{locked_table if locked_table else 'Pending.'}
""",
    )

    write(
        docs / "regret_reduction_v8.md",
        f"""
# Regret Reduction V8

V8 focused on reducing selector regret inside the surviving `cc_news` main lane rather than searching for a new family.

## Main Locked Outcome

- pooled locked delta to static: `{fmt(delta_row['mean']) if delta_row is not None else 'n/a'}`
- pooled locked regret to bank: `{fmt(regret_row['mean']) if regret_row is not None else 'n/a'}`
- V7 locked regret reference: `0.08410`
- V8 regret change vs V7: `{fmt((float(regret_row['mean']) - 0.08410) if regret_row is not None else None)}`

## Interpretation

If the pooled regret is below the V7 locked reference, V8 succeeded at narrowing the bank-gap on the main locked lane.
Otherwise the main contribution is replication and calibration rather than additional regret reduction.
""",
    )

    write(
        docs / "scorer_family_v8.md",
        f"""
# Scorer Family V8

The V8 search space was intentionally restricted to the surviving `cc_news` main-lane families:

- `rf_pair`
- `hgb_pair`
- `dual_tower`
- `binary / ternary gate`
- `retrieval + rerank`

## Selected Winners
{chr(10).join(scorer_lines) if scorer_lines else "- pending"}

## Takeaway

The question in V8 was not whether a dead family could be rescued. It was whether stronger candidate-conditioned scoring could shrink regret on the locked main lane.
""",
    )

    write(
        docs / "mainlane_specialized_gate_v8.md",
        """
# Mainlane Specialized Gate V8

V8 evaluated specialized low-route-universe gates (`binary_gate_top1`, `ternary_gate_top2`) as deployment-oriented alternatives inside the surviving low-budget `cc_news` lane.

These gates were kept inside the allowed V8 family but were judged against the main candidate-conditioned tree/tower scorers rather than treated as a new story.

The key decision criterion was whether a tiny route universe could improve generalization and reduce runtime overhead without giving back the locked quality gain.
""",
    )

    write(
        docs / "readiness_v4.md",
        f"""
# Readiness V4

V8 treated route-best checkpoint choice as part of the main scientific claim.

## Route-Best vs LM-Best
{chr(10).join(checkpoint_lines) if checkpoint_lines else "- pending"}

## Interpretation

Readiness-v4 was successful if route-best and LM-best diverged reproducibly and the route-best choice carried through to fresh locked final splits.
""",
    )

    write(
        docs / "checkpoint_replication_v8.md",
        f"""
# Checkpoint Replication V8

Dense checkpoint replication was carried out on the frozen `cc_news` main family.

## Seed-Level Checkpoint Outcomes
{chr(10).join(checkpoint_lines) if checkpoint_lines else "- pending"}

## Main Message

The V8 checkpoint question was not whether every seed picks `step3000`, but whether route-best remains a stable concept that is distinct from LM-best across replicated seeds and fresh finals.
""",
    )

    write(
        docs / "ccnews_necessity_multiseed_v8.md",
        f"""
# CCNews Necessity Multiseed V8

The V8 necessity question was restricted to the main low-budget `cc_news` slice.

## Pooled Results

- AttnRes dynamic delta: `{fmt(attn_dyn_delta['mean']) if attn_dyn_delta is not None else 'n/a'} [{fmt(attn_dyn_delta['ci_low']) if attn_dyn_delta is not None else 'n/a'}, {fmt(attn_dyn_delta['ci_high']) if attn_dyn_delta is not None else 'n/a'}]`
- AttnRes hidden-only delta: `{fmt(attn_hidden_delta['mean']) if attn_hidden_delta is not None else 'n/a'} [{fmt(attn_hidden_delta['ci_low']) if attn_hidden_delta is not None else 'n/a'}, {fmt(attn_hidden_delta['ci_high']) if attn_hidden_delta is not None else 'n/a'}]`
- Standard hidden-only delta: `{fmt(std_hidden_delta['mean']) if std_hidden_delta is not None else 'n/a'} [{fmt(std_hidden_delta['ci_low']) if std_hidden_delta is not None else 'n/a'}, {fmt(std_hidden_delta['ci_high']) if std_hidden_delta is not None else 'n/a'}]`

## Per Seed / Final Split

{necessity_table if necessity_table else 'Pending.'}
""",
    )

    write(
        docs / "systems_speedup_v8.md",
        f"""
# Systems Speedup V8

V8 systems work stayed inside the narrow main lane:

- `cc_news`
- low-budget `skip1`
- small route universe
- template limits `{0, 2, 4}`

## Best Template-Limit Summary

- best template limit by latency delta: `{fmt(best_tpl['template_limit'], 0) if best_tpl is not None else 'n/a'}`
- latency delta vs static: `{fmt(best_tpl['dynamic_latency_delta_vs_static']) if best_tpl is not None and 'dynamic_latency_delta_vs_static' in best_tpl else 'n/a'}`
- quality delta vs static: `{fmt(best_tpl['dynamic_delta_to_static_quality']) if best_tpl is not None and 'dynamic_delta_to_static_quality' in best_tpl else 'n/a'}`

## Template Summary

{systems_table if systems_table else 'Pending.'}
""",
    )

    write(
        docs / "deployment_breakdown_v8.md",
        f"""
# Deployment Breakdown V8

This document records how much of the remaining deployment gap comes from:

- selector overhead
- grouping / template fragmentation
- decode time itself

The main deployment success criterion in V8 was strict:

- dynamic quality better than or equal to static
- end-to-end latency also lower than static

Current best template summary:

{systems_table if systems_table else 'Pending.'}
""",
    )

    write(
        docs / "boundary_analysis_v8.md",
        f"""
# Boundary Analysis V8

V8 no longer asked whether broad cross-corpus positivity could be rescued.
It asked why the narrow `cc_news` lane survives while prior boundary corpora did not.

## Overall Locked Main Slice

- overall delta to static: `{fmt(overall_subgroup['delta_to_static_mean']) if overall_subgroup is not None else 'n/a'}`
- overall regret to bank: `{fmt(overall_subgroup['regret_to_bank_mean']) if overall_subgroup is not None else 'n/a'}`
- overall fraction improved: `{fmt(overall_subgroup['fraction_improved']) if overall_subgroup is not None else 'n/a'}`

## Subgroup Table

{subgroup_table if subgroup_table else 'Pending.'}
""",
    )

    write(
        docs / "ccnews_subgroup_robustness_v8.md",
        f"""
# CCNews Subgroup Robustness V8

The subgroup analysis checked whether the surviving main lane is:

- broadly present across the locked `cc_news` finals
- or concentrated into a narrower subset such as domain / year / article length

{subgroup_table if subgroup_table else 'Pending.'}
""",
    )

    write(
        docs / "signal_preserving_v8.md",
        """
# Signal Preserving V8

V8 did not make signal-preserving continuation a primary branch.

Reason:

- the main V8 priority was fresh lockbox replication, multiseed replication, regret reduction, multiseed necessity, and systems-aware deployment on the surviving `cc_news` lane
- these were higher-value uses of compute than another auxiliary continuation branch

So V8 treats signal-preserving continuation as deferred rather than as a decisive result.
""",
    )

    write(
        docs / "final_report_v8.md",
        f"""
# Final Report V8

## Scope

V8 was the hardening round for the surviving `cc_news` main lane.

Its job was:

- fresh lockbox replication on new `final_A/B/C`
- multiseed replication
- regret reduction inside the surviving candidate-conditioned family
- readiness-v4 checkpoint replication
- multiseed necessity on the main `cc_news` slice
- systems-aware deployment follow-up
- boundary analysis inside `cc_news`

## Frozen Main Lane

{chr(10).join(winner_lines) if winner_lines else '- pending'}

## Locked Main Result

- pooled delta to static: `{fmt(delta_row['mean']) if delta_row is not None else 'n/a'} [{fmt(delta_row['ci_low']) if delta_row is not None else 'n/a'}, {fmt(delta_row['ci_high']) if delta_row is not None else 'n/a'}]`
- pooled regret to bank: `{fmt(regret_row['mean']) if regret_row is not None else 'n/a'} [{fmt(regret_row['ci_low']) if regret_row is not None else 'n/a'}, {fmt(regret_row['ci_high']) if regret_row is not None else 'n/a'}]`
- pooled fraction improved: `{fmt(frac_row['mean']) if frac_row is not None else 'n/a'} [{fmt(frac_row['ci_low']) if frac_row is not None else 'n/a'}, {fmt(frac_row['ci_high']) if frac_row is not None else 'n/a'}]`

## Necessity

- AttnRes dynamic: `{fmt(attn_dyn_delta['mean']) if attn_dyn_delta is not None else 'n/a'}`
- AttnRes hidden-only: `{fmt(attn_hidden_delta['mean']) if attn_hidden_delta is not None else 'n/a'}`
- Standard hidden-only: `{fmt(std_hidden_delta['mean']) if std_hidden_delta is not None else 'n/a'}`

## Systems

- best template-limit latency delta vs static: `{fmt(best_tpl['dynamic_latency_delta_vs_static']) if best_tpl is not None and 'dynamic_latency_delta_vs_static' in best_tpl else 'n/a'}`
- best template-limit quality delta vs static: `{fmt(best_tpl['dynamic_delta_to_static_quality']) if best_tpl is not None and 'dynamic_delta_to_static_quality' in best_tpl else 'n/a'}`

## Boundary

- overall locked `cc_news` subgroup delta: `{fmt(overall_subgroup['delta_to_static_mean']) if overall_subgroup is not None else 'n/a'}`
- overall locked `cc_news` subgroup regret: `{fmt(overall_subgroup['regret_to_bank_mean']) if overall_subgroup is not None else 'n/a'}`

## Bottom Line

V8 was not a broad cross-corpus fishing round.
It was a hardening round for the already-surviving `cc_news` main lane.

The paper-facing judgment therefore depends on four questions:

1. does the fresh multiseed + multisplit lockbox stay below zero?
2. does regret shrink relative to V7?
3. does the main low-budget necessity claim survive multiseed comparison?
4. does systems-aware deployment close the wall-clock gap enough to matter?
""",
    )

    write(
        docs / "paper_verdict_v8.md",
        f"""
# Paper Verdict V8

## 1. Does the V7 main lane survive on fresh final-A/B/C?

- pooled delta to static: `{fmt(delta_row['mean']) if delta_row is not None else 'n/a'} [{fmt(delta_row['ci_low']) if delta_row is not None else 'n/a'}, {fmt(delta_row['ci_high']) if delta_row is not None else 'n/a'}]`

## 2. Does it survive across multiple seeds?

{locked_table if locked_table else 'Pending.'}

## 3. How much did regret shrink versus V7?

- V7 regret reference: `0.08410`
- V8 pooled regret: `{fmt(regret_row['mean']) if regret_row is not None else 'n/a'}`
- change vs V7: `{fmt((float(regret_row['mean']) - 0.08410) if regret_row is not None else None)}`

## 4. Does the route-best checkpoint story survive replication?

{chr(10).join(checkpoint_lines) if checkpoint_lines else '- pending'}

## 5. Does cc_news necessity remain AttnRes-specific in multiseed form?

- AttnRes dynamic: `{fmt(attn_dyn_delta['mean']) if attn_dyn_delta is not None else 'n/a'}`
- AttnRes hidden-only: `{fmt(attn_hidden_delta['mean']) if attn_hidden_delta is not None else 'n/a'}`
- Standard hidden-only: `{fmt(std_hidden_delta['mean']) if std_hidden_delta is not None else 'n/a'}`

## 6. Does the systems-aware variant improve real wall-clock quality-speed?

- best template-limit latency delta vs static: `{fmt(best_tpl['dynamic_latency_delta_vs_static']) if best_tpl is not None and 'dynamic_latency_delta_vs_static' in best_tpl else 'n/a'}`
- best template-limit quality delta vs static: `{fmt(best_tpl['dynamic_delta_to_static_quality']) if best_tpl is not None and 'dynamic_delta_to_static_quality' in best_tpl else 'n/a'}`

## 7. How strong can the paper claim be now?

Safe claim after V8:

- AttnRes-based candidate-conditioned routing has a narrow, low-budget `cc_news` lane that can be tested under fresh lockbox replication, multiseed selection, and multiseed necessity.

Unsafe claim after V8:

- broad cross-corpus positivity
- universal AttnRes necessity
- guaranteed wall-clock speedup
""",
    )


if __name__ == "__main__":
    main()
