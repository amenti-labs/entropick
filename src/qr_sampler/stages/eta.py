"""EtaSamplingStage -- entropy-aware probability cutoff."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from qr_sampler.pipeline.registry import StageRegistry
from qr_sampler.stages._utils import shannon_entropy_from_probs, stable_softmax

if TYPE_CHECKING:
    from qr_sampler.pipeline.context import SamplingContext


@StageRegistry.register("eta")
class EtaSamplingStage:
    """Eta sampling: entropy-dependent probability threshold.

    The algorithm:
        1. Compute softmax probabilities and Shannon entropy H.
        2. Convert ``eta_cutoff`` from 1e-4 units: ``eta = eta_cutoff * 1e-4``.
        3. Compute threshold: ``max(eta, sqrt(eta) * exp(-H))``.
        4. Remove tokens with probability below the threshold.

    The entropy-dependent term ``sqrt(eta) * exp(-H)`` makes the
    threshold adaptive: when the model is confident (low H), the
    threshold rises and more tokens are removed.  When uncertain
    (high H), the threshold drops and more candidates survive.

    Always keeps at least one token.

    No-ops when ``config.eta_cutoff <= 0``.
    """

    name: str = "eta"

    def __call__(self, ctx: SamplingContext) -> None:
        if ctx.config.eta_cutoff <= 0.0:
            return

        probs = stable_softmax(ctx.row)
        if probs is None:
            return

        h = shannon_entropy_from_probs(probs)

        eta = ctx.config.eta_cutoff * 1e-4
        threshold = max(eta, math.sqrt(eta) * math.exp(-h))

        below = probs < threshold
        # Always keep at least one token.
        if np.all(below):
            return

        ctx.row = np.where(below, -np.inf, ctx.row)
