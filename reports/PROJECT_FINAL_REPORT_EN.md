# AttnRes Routing Research: Final Preservation Report

## 1. What This Project Was

This repository studied a specific question:

Can Attention Residuals inside a decoder LM be used as an internal routing signal to decide when to skip depth without losing quality?

The project started from a fairly strong hope:

- Attention Residuals might expose sequence-specific depth demand.
- A routing policy might then skip work adaptively.
- That could improve quality-vs-compute tradeoffs, and possibly end-to-end latency.

By the end of the project, the answer was narrower than the original ambition.

- The signal is real.
- A broad prompt-fixed routing win did not survive.
- A narrow `cc_news` quality-positive lane did survive repeated hardening.
- That lane still did not become a real deployment-speed win.

## 2. What Is Preserved In This Archive

This archive keeps the code needed to understand and rerun the project logic:

- `src/attnres_routing/`: core library code for models, routing, data, analysis, training, and manifests
- `scripts/`: experiment runners, evaluators, sharding helpers, aggregation scripts, and pipeline wrappers
- `configs/`: YAML training and experiment configs across the major rounds
- `README.md`
- `requirements.txt`
- this final report in English and Korean

This archive does not include:

- `.git/`
- `.venv/`
- `data/`
- `results/`
- `logs/`
- `external/`

Those were excluded because the user requested a code-only preservation package plus final reports.

## 3. Code Map For A New Reader

Start here if you need to orient quickly:

- `src/attnres_routing/model.py`: decoder LM and AttnRes model components
- `src/attnres_routing/routing.py`: routing and mask-selection logic
- `src/attnres_routing/analysis.py`: summary metrics and bootstrap utilities
- `src/attnres_routing/data.py`: dataset preparation
- `src/attnres_routing/sequence_manifest.py`: lockbox/train/dev manifest construction
- `scripts/train_lm.py`: base LM training entrypoint
- `scripts/evaluate_functional_oracles.py`: oracle evaluation
- `scripts/evaluate_prompt_routing.py`: early prompt-routing evaluation
- `scripts/train_candidate_conditioned_ranker_v7.py`: later candidate-conditioned selector training
- `scripts/evaluate_deployment_measurement_v7.py`: deployment-quality and latency measurement
- `scripts/build_lockbox_manifests_v9.py`: fresh V9 split construction

The final operational fixes left in the code are:

- `scripts/evaluate_deployment_measurement_v7.py`
- `scripts/run_compare_v8_seed_repro_v9.sh`

## 4. How The Research Progressed

### Phase A. Early Reproduction And Signal Discovery

The project began with faithful Block AttnRes reproduction work and early prompt-routing experiments on smaller settings such as WikiText.

What was learned:

- Attention Residuals did contain a real depth-related signal.
- The signal was measurable with leave-one-block-out agreement metrics.
- But prompt-fixed sequence routing did not beat a strong global static skip schedule.

This was the first major correction to the original thesis.

### Phase B. Controls And Negative-Result Consolidation

The project then added transfer-integrity controls, longer 4-GPU trajectory runs, batched functional-oracle evaluation, and stability-gated routing.

The important conclusion from this stage was already clear:

- AttnRes carried meaningful information.
- The then-current routing policy still failed to turn that information into a reliable routing win.

This is the state summarized in `docs/current_status_v2.md` in the original repository.

### Phase C. Scale-Up To `cc_news` And Candidate-Conditioned Routing

The project then pivoted from the failing prompt-fixed policy to a stronger family of candidate-conditioned selectors on larger `24x512` models and `cc_news`.

This was the key pivot that kept the project alive.

Instead of asking only whether prompt features could directly choose depth, the later work evaluated:

- candidate-conditioned selectors
- richer feature families
- bank-based routing
- readiness and checkpoint-selection procedures
- lockbox split protocols
- systems-aware deployment measurement

### Phase D. V8 Hardening

V8 was the hardening round for the surviving `cc_news` lane.

What V8 established:

- the main lane survived fresh `final_A/B/C` lockbox evaluation
- the lane replicated across seeds `42/43/44`
- necessity analyses supported the claim that the strongest lane was not reproduced by the matched standard hidden-only control
- deployment-aware evaluation still failed to show a wall-clock win

V8 therefore supported a narrow claim, not a broad one:

AttnRes-based candidate-conditioned routing had a replicated low-budget quality-positive lane on `cc_news`, but not a general adaptive-routing victory.

### Phase E. V8 Forensics

Because V8 had interrupted-run provenance weaknesses, the project then audited whether the V8 claim was numerically and procedurally trustworthy.

The forensic result was:

- V8 summary numbers were numerically reproducible from raw artifacts.
- The lockbox itself appeared clean and disjoint.
- Documentation around freeze logging and interrupted-run provenance was weaker than it should have been.

That is why a fresh V9 lockbox remained necessary.

### Phase F. V9 Fresh Lockbox Closure

V9 was the final closure round.

It built a fresh `dev_select_v9` plus unopened `final_D/E/F` lockbox, froze winners on dev only, and then ran all three final splits.

V9 completed on `2026-03-28 13:39:41 KST`.

Frozen winners:

| seed | step | bank | feature | selector |
| --- | ---: | ---: | --- | --- |
| 42 | 5500 | 32 | `attnres` | `rf_pair` |
| 43 | 6000 | 32 | `stp_scalar` | `hgb_pair` |
| 44 | 3500 | 32 | `attnres` | `rf_pair` |

## 5. Final Findings

### Main Scientific Finding

The strongest defensible final claim is:

There is a narrow, reproducible quality-positive routing lane on `cc_news`.

More concretely:

- V9 frozen winners stayed below static on all three locked splits `final_D/E/F`.
- Mean locked `delta_to_static` across the nine seed-split evaluations was `-0.02339`.
- Mean `fraction_improved` was `0.16800`.

### What Did Not Hold

The project does not support the stronger claims that originally motivated it.

It does not support:

- broad cross-corpus generality
- a general prompt-fixed routing win
- a low-regret approximation to the oracle bank
- a deployment-speed win over static

### Deployment Result

The deployment side remained the limiting factor.

Across the V9 locked deployment summaries:

- mean deploy `delta_to_global_static`: `-0.01643`
- mean dynamic end-to-end seconds/sequence: `0.22158`
- mean latency delta vs static: `+0.03174 s/sequence`

So quality stayed slightly better than static, but the dynamic policy still ran slower than static.

## 6. Final Interpretation

The project did discover something real:

- Attention Residuals can carry endogenous routing signal.
- Under the later `cc_news` setup, that signal can support a narrow quality-positive selector family.

But the original larger dream did not close:

- the result is narrow, not general
- the quality gain is modest
- regret remains far from the oracle upper bound
- deployment latency does not improve

The final paper-facing sentence should therefore be:

AttnRes-based internal routing signals can produce a narrow, reproducible quality-positive lane on `cc_news`, but they do not currently deliver a deployment-speed win.

## 7. Operational Notes From The Final Round

The last active engineering issues that were fixed during V9 were:

- `scripts/evaluate_deployment_measurement_v7.py`: fixed a missing `stp_diff_components` handoff that crashed deployment measurement
- `scripts/run_compare_v8_seed_repro_v9.sh`: fixed malformed default compare-template strings

There was also storage pressure late in the project:

- heavy result directories were moved to `/var/tmp/chojm-attnres-routing-research-results/`
- symlinks were left in `results/` so the pipeline could continue

Those result artifacts are not included in this archive.

## 8. Recommended Reading Order

If someone opens this package cold, the fastest useful order is:

1. read this report
2. read `README.md`
3. inspect `src/attnres_routing/`
4. inspect `scripts/`
5. use `configs/scale_heterogeneity_v8/` and `configs/scale_heterogeneity_v9/` for the late-stage setup

## 9. Bottom Line

This project should be remembered as a mixed result:

- it found real internal routing signal
- it falsified several broader hopes
- it eventually isolated one narrow, replicated quality-positive lane
- it still failed to turn that lane into a convincing systems win

That is the most accurate end state of the research.
