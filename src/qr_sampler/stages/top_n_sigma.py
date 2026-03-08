"""TopNSigmaStage -- keep logits within n standard deviations of the max."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from qr_sampler.pipeline.registry import StageRegistry

if TYPE_CHECKING:
    from qr_sampler.pipeline.context import SamplingContext


@StageRegistry.register("top_n_sigma")
class TopNSigmaStage:
    """Filter tokens whose logits fall more than n std below the maximum.

    Operates on raw logits (pre-softmax):
        threshold = max(logits) - n * std(logits)
    Any token with logit below the threshold is masked to ``-inf``.

    This provides an adaptive cutoff that scales with model confidence:
    when the logit distribution is tight (low std), even a small n
    removes many tokens.  When spread is large, more tokens survive.

    Always keeps at least one token.

    No-ops when ``config.top_n_sigma <= 0``.
    """

    name: str = "top_n_sigma"

    def __call__(self, ctx: SamplingContext) -> None:
        if ctx.config.top_n_sigma <= 0.0:
            return

        finite_mask = np.isfinite(ctx.row)
        if not np.any(finite_mask):
            return

        finite_logits = ctx.row[finite_mask]
        max_logit = float(np.max(finite_logits))
        std_logit = float(np.std(finite_logits))

        threshold = max_logit - ctx.config.top_n_sigma * std_logit

        below = ctx.row < threshold
        # Always keep at least one token (the highest-logit one).
        if np.all(below | ~finite_mask):
            return

        ctx.row = np.where(below, -np.inf, ctx.row)
