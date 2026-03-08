"""TailFreeSamplingStage -- tail-free sampling via second-derivative analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from qr_sampler.pipeline.registry import StageRegistry
from qr_sampler.stages._utils import stable_softmax

if TYPE_CHECKING:
    from qr_sampler.pipeline.context import SamplingContext


@StageRegistry.register("tfs")
class TailFreeSamplingStage:
    """Tail-free sampling: remove low-probability tail using second derivatives.

    The algorithm:
        1. Compute softmax probabilities from logits.
        2. Sort probabilities descending.
        3. Compute the second derivative (differences of differences).
        4. Normalize the absolute second derivatives.
        5. Accumulate cumulative sum; keep tokens where cumsum < z.

    This method identifies the "tail" of the distribution by finding
    where the probability mass drops off sharply (high second derivative),
    then truncates there.

    Always keeps at least one token.

    No-ops when ``config.tfs_z >= 1.0``.
    """

    name: str = "tfs"

    def __call__(self, ctx: SamplingContext) -> None:
        if ctx.config.tfs_z >= 1.0:
            return

        probs = stable_softmax(ctx.row)
        if probs is None:
            return

        # Sort probabilities descending and track original indices.
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]

        # Need at least 3 tokens to compute a second derivative.
        if len(sorted_probs) < 3:
            return

        # First derivative (differences).
        first_deriv = sorted_probs[:-1] - sorted_probs[1:]
        # Second derivative (differences of differences).
        second_deriv = first_deriv[:-1] - first_deriv[1:]

        # Normalize absolute second derivatives.
        abs_second = np.abs(second_deriv)
        total = float(np.sum(abs_second))
        if total == 0.0:
            # Uniform distribution -- no tail to free.
            return

        normalized = abs_second / total

        # Cumulative sum.  Each entry i corresponds to sorted token i.
        cumsum = np.cumsum(normalized)

        # Keep tokens where cumsum < z (plus the first two which have
        # no second derivative entry).  Token i maps to cumsum[i].
        # Tokens 0 and 1 are always kept (cumsum starts at index 0
        # for token 0's second derivative).
        keep_mask = np.zeros(len(sorted_probs), dtype=bool)
        keep_mask[0] = True
        keep_mask[1] = True
        # Keep token i+2 while cumsum[i] < z (tail hasn't started yet).
        # Find how many cumsum entries are below z (contiguous from start).
        below_z = cumsum < ctx.config.tfs_z
        if np.any(below_z):
            # Count leading True values (stop at first False).
            first_false = int(np.argmin(below_z)) if not np.all(below_z) else len(below_z)
            keep_mask[2 : 2 + first_false] = True

        # Always keep at least one token.
        if not np.any(keep_mask):
            keep_mask[0] = True

        # Map back to original indices and mask.
        remove_indices = sorted_indices[~keep_mask]
        ctx.row[remove_indices] = -np.inf
