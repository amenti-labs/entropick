"""TypicalSamplingStage -- locally typical sampling."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from qr_sampler.pipeline.registry import StageRegistry
from qr_sampler.stages._utils import shannon_entropy_from_probs, stable_softmax

if TYPE_CHECKING:
    from qr_sampler.pipeline.context import SamplingContext


@StageRegistry.register("typical")
class TypicalSamplingStage:
    """Locally typical sampling: keep tokens closest to typical information content.

    The algorithm:
        1. Compute softmax probabilities and Shannon entropy H.
        2. For each token, compute |-log(p_i) - H| (distance from typical).
        3. Sort tokens by distance ascending.
        4. Keep the smallest set whose cumulative probability >= typical_p.
        5. Mask remaining tokens to ``-inf``.

    Typical sampling preferentially selects tokens whose surprisal is
    close to the entropy of the distribution -- these are the "typical"
    tokens that carry neither too much nor too little information.

    No-ops when ``config.typical_p >= 1.0``.
    """

    name: str = "typical"

    def __call__(self, ctx: SamplingContext) -> None:
        if ctx.config.typical_p >= 1.0:
            return

        probs = stable_softmax(ctx.row)
        if probs is None:
            return

        h = shannon_entropy_from_probs(probs)

        # Compute distance from typical information content for each token.
        # For tokens with p=0, information is infinite; they are naturally excluded.
        positive_mask = probs > 0
        distances = np.full(len(probs), np.inf)
        distances[positive_mask] = np.abs(-np.log(probs[positive_mask]) - h)

        # Sort tokens by distance ascending (most typical first).
        sorted_indices = np.argsort(distances)
        sorted_probs = probs[sorted_indices]

        # Keep the smallest set whose cumulative probability >= typical_p.
        cumsum = np.cumsum(sorted_probs)
        # Find the first index where cumsum >= typical_p.
        keep_count = int(np.searchsorted(cumsum, ctx.config.typical_p, side="left")) + 1
        keep_count = max(1, min(keep_count, len(sorted_probs)))

        # Mask all tokens outside the kept set.
        keep_mask = np.zeros(len(ctx.row), dtype=bool)
        keep_mask[sorted_indices[:keep_count]] = True
        ctx.row[~keep_mask] = -np.inf
