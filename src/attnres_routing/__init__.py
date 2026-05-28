"""Public package surface for the attention-residual-routing experiment.

This pilot asked whether a model-internal signal — the way each block's
attention-residual sub-layer weights earlier depth sources — can be read at
inference time to decide *per input* which layers to skip. The package owns
the decoder LM, the routing logic, the data plumbing, and the analyses that
the closure reports cite.

Only the two main entrypoints are re-exported here:

- ``AttnResConfig`` — dataclass that captures both the standard decoder
  hyperparameters and the attention-residual extras (``num_blocks``,
  ``depth_normalizer``, ``depth_temperature``, ``topk_softmax_k``,
  ``residual_mode``). See :mod:`attnres_routing.model`.
- ``DecoderLM`` — the actual model. Runs either a standard pre-LN residual
  stack (``residual_mode="standard"``) or the block-attention-residual stack
  (``residual_mode="block_attnres"``) that exposes the per-block source
  attention used by routing and analysis.

The rest of the modules (data, normalizers, sublayer_masks, routing,
sequence_manifest, analysis, train, utils) are imported directly from their
submodules — see ``GLOSSARY.md`` at the repo root for vocabulary.
"""
from .model import AttnResConfig, DecoderLM

__all__ = ["AttnResConfig", "DecoderLM"]
