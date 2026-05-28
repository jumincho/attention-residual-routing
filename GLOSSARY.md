# GLOSSARY

Internal vocabulary that lives in this repo's source, configs, and final
reports. The reports name datasets, splits, mask formats, score modes, and
config rounds by short on-disk identifiers — this file decodes them so a
reader who opens the repo cold can follow what each one means and where it
is defined.

Cross-reference for everything below:

- Code library: [`src/attnres_routing/`](src/attnres_routing/)
- Per-round configs: [`configs/scale_heterogeneity_v5..v9/`](configs/)
- Closure reports:
  [`reports/PROJECT_FINAL_REPORT_KO.md`](reports/PROJECT_FINAL_REPORT_KO.md) ·
  [`reports/PROJECT_FINAL_REPORT_EN.md`](reports/PROJECT_FINAL_REPORT_EN.md)

---

## Dataset aliases

Every dataset is registered in
[`src/attnres_routing/data.py`](src/attnres_routing/data.py) under the
`DATASET_ALIASES` table. The alias string is the only surface configs and
CLI args use to name a dataset.

| Alias | Underlying HF dataset | Notes |
|---|---|---|
| `tinystories` | `roneneldan/TinyStories` | Smallest smoke / debug corpus. Used in `configs/smoke_tinystories_*.yaml` for round-zero sanity. |
| `wikitext103` | `wikitext` / `wikitext-103-raw-v1` | The screening / trajectory baseline corpus. Used in the `screen_wikitext_*` and `trajectory24x384_wikitext_*` configs and in the v5/v6 standard-baseline rows of the closure reports. |
| `openwebtext10k` | `stas/openwebtext-10k` | Mid-size web corpus added in the second-corpus rounds (v6/v7). |
| `c4_en` | `c4` / `en` | English C4. Used as a second-corpus diversity check. |
| `cc_news` | `cc_news` | The narrow corpus where the only surviving quality edge held across seeds and on the lockbox split. Heavy presence in v7/v8/v9. |
| `fineweb_edu_sample10bt` | `HuggingFaceFW/fineweb-edu` / `sample-10BT` | Big-corpus generalization round. Uses HF range slices `train[:12000]` / `train[12000:14000]` / `train[14000:16000]` so train / validation / test are disjoint slices of the same sample. |
| `fineweb_edu_sample10bt_stream` | same | Streaming variant — same offsets / limits but consumed via `datasets` streaming when the full sample doesn't fit on disk. |
| `fineweb_edu_sample10bt_local_v7` | local JSONL under `data/third_corpus_v7/*.jsonl` | The v7 frozen offline snapshot. |

---

## Sublayer-mask vocabulary

Every routing decision in the codebase is expressed as a
`SublayerMask` (see
[`src/attnres_routing/sublayer_masks.py`](src/attnres_routing/sublayer_masks.py)).
A mask is two boolean tuples of length `num_blocks`:

- `attn_mask` — for each block, is the attention sub-layer active.
- `mlp_mask` — for each block, is the MLP sub-layer active.

### Action types

For each block, `(attn_mask[b], mlp_mask[b])` collapses to one of four
*action types* — this is the routing vocabulary used everywhere from
configs to reports:

| Action | `attn` | `mlp` | What happens |
|---|---|---|---|
| `full` | on | on | Run both attention and MLP sub-layers normally. |
| `skip_attn` | off | on | Attention sub-layer is skipped (residual passes the depth-mixed input through unchanged); MLP runs. |
| `skip_mlp` | on | off | Attention runs; MLP sub-layer is skipped. |
| `skip_block` | off | off | Both sub-layers skipped — the block is effectively a pass-through (or a depth-mix-only pass-through in `block_attnres` mode). |

### `SublayerMask.to_id()` format

`to_id()` produces a deterministic string id of the form:

```
attn:1101|mlp:1011
```

- `attn:` is followed by `num_blocks` bits — `1` means attention on, `0`
  means attention off, in block order.
- `|mlp:` is followed by `num_blocks` bits in the same convention.

This string is the key used in manifests, JSON, CSV outputs, and the
candidate-conditioned ranker. `from_id()` parses it back into a mask.

### Reserved masks and helpers

- `from_block_mask(per_block_on_off)` → a mask where attention and MLP
  follow the same per-block on/off vector (so per-block `True` is `full`,
  `False` is `skip_block`).
- `enumerate_local_edits(anchor, block_indices, max_edits, candidate_actions)`
  — generate the candidate set within `max_edits` action flips of an anchor
  mask. This drives the candidate-conditioned ranker (v7+) and the
  functional-oracle search.
- `edit_distance(a, b, ignore_final=True)` — Hamming distance between two
  masks; `ignore_final=True` excludes the last block (the last block is
  always kept in practice).

---

## Routing score modes

Defined in
[`src/attnres_routing/routing.py`](src/attnres_routing/routing.py) under
`compute_routing_scores(mode=...)`. The score modes turn the
per-source utility / variance / top-k frequency vectors (produced by
[`analysis.py`](src/attnres_routing/analysis.py)) into a single per-source
score that the route-selection helpers consume.

| Mode | Formula | What it prefers |
|---|---|---|
| `utility` | `utility[s]` | Highest raw mean utilization. Simplest signal. |
| `utility_over_variance` | `utility[s] / (eps + variance[s])` | Depths that are *consistently* useful across prompt chunks — penalizes "spiky" depths. |
| `utility_times_topk_frequency` | `utility[s] * topk_frequency[s]` | Depths that show up in the per-chunk top-k often — penalizes depths that are useful only once. |

Both `_over_variance` and `_times_topk_frequency` modes were attempts to
make the score more stable across input distributions; closure reports
include comparison tables for all three modes.

---

## Normalizers (depth-axis)

Defined in
[`src/attnres_routing/normalizers.py`](src/attnres_routing/normalizers.py)
via the single dispatch `depth_normalize(logits, mode=..., temperature=...,
topk=...)`. This is what
[`model.DepthMix`](src/attnres_routing/model.py) calls to turn
per-source logits into per-source weights along the depth axis.

| `mode` | What it does |
|---|---|
| `softmax` | Standard temperature-scaled softmax. Always-dense output. |
| `temperature_softmax` | Same as `softmax`, kept as a separate name because some configs reference it explicitly. |
| `sparsemax` | Martins & Astudillo's sparse projection. Can emit exact zeros, so unused depths are masked out. |
| `entmax15` | α=1.5 entmax. Between softmax (dense) and sparsemax (very sparse); used as the in-between knob. |
| `topk_softmax` | Softmax restricted to the top-`k` depth sources, with everything else forced to `-inf` before softmax. Controlled by `AttnResConfig.topk_softmax_k`. |

### `depth_temperature`

Scalar divider applied to the logits *before* the chosen normalizer. Set
on `AttnResConfig.depth_temperature`. Lower temperature → sharper weights;
higher temperature → flatter weights. Used to control how committed the
depth attention is to its top picks.

---

## Config rounds (`configs/scale_heterogeneity_v*/`)

The pilot scaled up round by round. Each round's configs live in their own
folder and are referenced by closure reports under their `_v*` name.

| Folder | What this round was for |
|---|---|
| [`configs/scale_heterogeneity_v5/`](configs/scale_heterogeneity_v5/) | First scaling-up round. Same WikiText-103 corpus, larger model, longer schedule than the screening / trajectory benchmarks. Establishes the "real but narrow" signal on a single corpus. |
| [`configs/scale_heterogeneity_v6/`](configs/scale_heterogeneity_v6/) | Adds a second corpus — `openwebtext10k` (and supporting C4 / FineWeb runs) — to test whether the signal transfers across corpora at the same model size. |
| [`configs/scale_heterogeneity_v7/`](configs/scale_heterogeneity_v7/) | Switches the main corpus to `cc_news` (the only corpus that survived) and to `fineweb_edu_sample10bt_local_v7` for the third-corpus diversity check. Introduces the *candidate-conditioned ranker* line — `train_candidate_conditioned_ranker_v7.py` consumes these configs. Includes resume-from-checkpoint variants for incremental scaling. |
| [`configs/scale_heterogeneity_v8/`](configs/scale_heterogeneity_v8/) | Tightens the v7 setup: STP regularizer sweeps, longer schedules, fresh selectors and ranker trainings on the narrow corpus to confirm the quality edge holds across seeds. |
| [`configs/scale_heterogeneity_v9/`](configs/scale_heterogeneity_v9/) | Final round. Introduces *lockbox* validation splits (validation documents the selector and the ranker have never seen). Confirms the narrow quality edge survives the lockbox check, and produces the joint quality + latency measurement that shows the wall-clock loss. |

`v5 → v9` is roughly: *one corpus, bigger model* (v5) → *more corpora,
same model* (v6) → *narrow corpus + candidate-conditioned selector*
(v7) → *seed-stability and stronger selector* (v8) → *lockbox split +
quality vs latency* (v9).

---

## "Lockbox" splits

A *lockbox* split is a validation split the routing selectors and the
candidate-conditioned ranker have **never** trained on or been selected
on. The lockbox manifests are built by
`scripts/build_lockbox_manifests_v9.py`, using
[`sequence_manifest.py`](src/attnres_routing/sequence_manifest.py) so each
document carries a stable `document_uid`. The point of the lockbox is to
catch the case where the v7/v8 quality edge was a model-selection artifact
on the same validation pool the ranker was tuned against. The narrow
`cc_news` edge survives this test.

---

## Candidate-conditioned ranker, functional oracles, oracle policies

These three terms come up in scripts and reports as different evaluation
regimes for the same underlying object — a per-input routing decision.

- **Functional oracle** — an *idealized* policy: for each input, enumerate
  candidate masks and pick the one that achieves the lowest
  continuation loss. This is the upper bound any per-input router could
  hope to match. `scripts/evaluate_functional_oracles.py` computes it.
- **Oracle policy** — the same idea but consumed by routing-eval scripts
  as a target for selectors to imitate. In code this is the
  oracle-labeled subset of `(input, mask) → loss` table.
- **Candidate-conditioned ranker** — the v7+ design. Instead of training
  a free-form selector that picks an action per block, train a *ranker*
  that takes (input features, candidate mask) and predicts the loss
  reduction. At inference time the ranker scores a small candidate set
  generated by `enumerate_local_edits` and the best-scoring candidate is
  used. `scripts/train_candidate_conditioned_ranker_v7.py` is the
  training entrypoint; the closure reports cite it as "CCR".

---

## Sub-cost vocabulary (FLOP budget)

In [`sublayer_masks.py`](src/attnres_routing/sublayer_masks.py),
`estimated_decode_cost` and `estimated_reduction_ratio` consume three
per-block cost arrays:

- `full_cost[b]` — measured cost of running both sub-layers in block `b`.
- `attn_cost[b]` — cost of running attention only.
- `mlp_cost[b]` — cost of running MLP only.

Action `full` adds `full_cost[b]`, `skip_attn` adds `mlp_cost[b]`,
`skip_mlp` adds `attn_cost[b]`, `skip_block` adds `0`. The reduction ratio
is `(baseline - cost) / baseline`. These numbers drive the FLOP-budgeted
"how much could we save?" tables but, as the closure says, FLOPs and
wall-clock do not agree under dynamic routing.

---

## STP regularizer

"Smooth Trajectory Penalty." In
[`train.py`](src/attnres_routing/train.py)::`stp_regularizer`. Samples
random sorted triplets `(s, r, t)` from the final hidden states over the
sequence and penalizes `1 - cos(h_t - h_r, h_r - h_s)`. The intent is to
keep hidden-state trajectories locally smooth so that block-skipping is a
small perturbation rather than a discontinuity. Controlled by
`TrainConfig.stp_weight` (0 disables) and `TrainConfig.stp_num_triplets`.
