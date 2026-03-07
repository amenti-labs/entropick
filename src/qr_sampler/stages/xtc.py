"""XTCStage -- Exclude Top Choices using quantum random bits."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from qr_sampler.exceptions import EntropyUnavailableError
from qr_sampler.pipeline.registry import StageRegistry
from qr_sampler.stages._utils import stable_softmax

if TYPE_CHECKING:
    from qr_sampler.pipeline.context import SamplingContext

_logger = logging.getLogger("qr_sampler")


@StageRegistry.register("xtc")
class XTCStage:
    """Exclude Top Choices: probabilistically remove top tokens using quantum bits.

    For each token whose probability exceeds ``xtc_threshold``, a quantum
    random bit decides whether to exclude it (with probability
    ``xtc_probability``).  This forces the model to use less obvious word
    choices, dramatically improving creativity while maintaining coherence.

    The key property for consciousness research: each exclusion is a
    *binary quantum decision*.  PEAR research found the strongest
    mind-matter effects on binary random processes (~10^-4 bits/bit).
    XTC creates exactly this architecture -- a small quantum bias toward
    including or excluding a top token cascades into large semantic shifts.

    Safety: always keeps at least one token (never produces an empty set).
    Operates on logits: sets excluded tokens to ``-inf``.

    No-ops when ``config.xtc_probability <= 0``.
    """

    name: str = "xtc"

    def __call__(self, ctx: SamplingContext) -> None:
        if ctx.config.xtc_probability <= 0.0:
            return

        probs = stable_softmax(ctx.row)
        if probs is None:
            return

        # Find tokens above the exclusion threshold.
        candidates = probs >= ctx.config.xtc_threshold
        n_candidates = int(np.sum(candidates))
        if n_candidates <= 1:
            # Nothing to exclude (need at least 2 candidates to exclude any).
            return

        # Fetch quantum random bytes -- one byte per candidate token.
        t_start = time.perf_counter_ns()
        try:
            raw_bytes = ctx.entropy_source.get_random_bytes(n_candidates)
        except EntropyUnavailableError:
            _logger.warning("XTC: entropy unavailable, skipping exclusion")
            return
        t_end = time.perf_counter_ns()
        ctx.entropy_fetch_ms += (t_end - t_start) / 1_000_000.0

        # Each byte -> uniform in [0, 1) -> exclude if < xtc_probability.
        quantum_u = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float64) / 256.0
        exclude_decisions = quantum_u < ctx.config.xtc_probability

        # Apply exclusions, but always keep at least one token.
        candidate_indices = np.where(candidates)[0]

        # Sort candidates by probability descending -- protect the last survivor.
        candidate_probs = probs[candidate_indices]
        sorted_order = np.argsort(candidate_probs)[::-1]
        sorted_indices = candidate_indices[sorted_order]
        sorted_exclude = exclude_decisions[sorted_order]

        # Count how many we'd exclude.
        n_total_tokens = int(np.sum(np.isfinite(ctx.row)))
        n_excluded = 0

        for i in range(len(sorted_indices)):
            # Don't exclude if it would leave zero tokens.
            remaining = n_total_tokens - n_excluded
            if remaining <= 1:
                break
            if sorted_exclude[i]:
                ctx.row[sorted_indices[i]] = -np.inf
                n_excluded += 1

        if ctx.config.injection_verbose and n_excluded > 0:
            _logger.debug(
                "XTC: excluded %d/%d top tokens (threshold=%.3f, prob=%.3f)",
                n_excluded,
                n_candidates,
                ctx.config.xtc_threshold,
                ctx.config.xtc_probability,
            )
