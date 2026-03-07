"""AdaptiveInjectionStage -- scale injection intensity by model uncertainty."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qr_sampler.pipeline.registry import StageRegistry
from qr_sampler.stages._utils import shannon_entropy_from_probs, stable_softmax

if TYPE_CHECKING:
    from qr_sampler.pipeline.context import SamplingContext


@StageRegistry.register("adaptive_injection")
class AdaptiveInjectionStage:
    """Scale injection intensity based on distribution entropy.

    Computes Shannon entropy H of the current logit distribution and
    maps it to an ``injection_scale`` in [0, 1] via linear interpolation
    between ``adaptive_injection_low_h`` (scale=0) and
    ``adaptive_injection_high_h`` (scale=1).

    Downstream injection stages (M1, M2, M3) read ``ctx.injection_scale``
    and multiply their parameters by it.  When the model is confident
    (low H), injection is suppressed.  When uncertain (high H), injection
    runs at full strength.

    This concentrates quantum influence at semantic "choice points" where
    the model is genuinely uncertain, dramatically improving signal-to-noise
    for consciousness experiments.  Deterministic tokens (articles,
    prepositions) are left alone.

    No-ops when ``config.adaptive_injection`` is False.
    """

    name: str = "adaptive_injection"

    def __call__(self, ctx: SamplingContext) -> None:
        if not ctx.config.adaptive_injection:
            return

        probs = stable_softmax(ctx.row)
        if probs is None:
            ctx.injection_scale = 0.0
            return

        h = shannon_entropy_from_probs(probs)

        # Linear interpolation: low_h -> 0.0, high_h -> 1.0.
        low_h = ctx.config.adaptive_injection_low_h
        high_h = ctx.config.adaptive_injection_high_h

        if high_h <= low_h:
            # Degenerate config -- use binary: 0 below threshold, 1 above.
            ctx.injection_scale = 0.0 if h < low_h else 1.0
            return

        scale = (h - low_h) / (high_h - low_h)
        ctx.injection_scale = max(0.0, min(1.0, scale))
