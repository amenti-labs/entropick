"""MinPStage -- dynamic probability floor filtering."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from qr_sampler.pipeline.registry import StageRegistry
from qr_sampler.stages._utils import stable_softmax

if TYPE_CHECKING:
    from qr_sampler.pipeline.context import SamplingContext


@StageRegistry.register("min_p")
class MinPStage:
    """Filter tokens whose probability falls below min_p * max(p).

    Unlike top-k (fixed count) or top-p (fixed cumulative mass), min-p
    scales the threshold dynamically with model confidence.  When the
    model is confident (high max-p), the threshold is high and few
    tokens survive.  When uncertain (low max-p), more tokens survive.

    Operates on logits directly: computes softmax internally, then sets
    logits to ``-inf`` for tokens below the threshold.  This preserves
    the logit-space representation for downstream stages.

    No-ops when ``config.min_p <= 0``.
    """

    name: str = "min_p"

    def __call__(self, ctx: SamplingContext) -> None:
        if ctx.config.min_p <= 0.0:
            return

        probs = stable_softmax(ctx.row)
        if probs is None:
            return

        p_max = float(np.max(probs))
        threshold = ctx.config.min_p * p_max

        # Mask tokens below threshold.
        below = probs < threshold
        # Always keep at least one token (the highest-probability one).
        if np.all(below):
            return

        ctx.row = np.where(below, -np.inf, ctx.row)
