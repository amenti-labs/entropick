"""MirostatStage -- Mirostat v2 adaptive perplexity-controlled sampling."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from qr_sampler.pipeline.registry import StageRegistry
from qr_sampler.stages._utils import stable_softmax

if TYPE_CHECKING:
    from qr_sampler.pipeline.context import SamplingContext

_logger = logging.getLogger("qr_sampler")

_MU_KEY = "mirostat.mu"


@StageRegistry.register("mirostat")
class MirostatStage:
    """Mirostat v2: adaptive perplexity control via surprise-rate feedback.

    Maintains a running estimate ``mu`` of the acceptable surprise level.
    At each token, keeps only candidates whose information content
    ``-log2(p)`` does not exceed ``mu``, then selects from those candidates
    using the amplified uniform value ``ctx.u``.  After selection, ``mu``
    is updated via a learning-rate feedback loop toward the target ``tau``.

    Writes ``ctx.token_id``, ``ctx.token_rank``, ``ctx.token_prob``,
    and ``ctx.num_candidates``.

    Reads and writes ``ctx.stage_state["mirostat.mu"]``.

    No-ops when ``config.mirostat_mode == 0``.
    """

    name: str = "mirostat"

    def __call__(self, ctx: SamplingContext) -> None:
        if ctx.config.mirostat_mode == 0:
            return
        if ctx.config.mirostat_mode != 2:
            _logger.warning(
                "Mirostat mode %d is not implemented; only mode 2 is supported. "
                "Set mirostat_mode=2 or mirostat_mode=0 to disable.",
                ctx.config.mirostat_mode,
            )
            return

        # --- Compute softmax probabilities ---
        probs = stable_softmax(ctx.row)
        if probs is None:
            return

        # --- Retrieve mu from persistent state ---
        mu = float(ctx.stage_state.get(_MU_KEY, 2.0 * ctx.config.mirostat_tau))
        tau = ctx.config.mirostat_tau
        eta = ctx.config.mirostat_eta

        # --- Sort tokens descending by probability ---
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]

        # --- Filter: keep tokens where -log2(p) <= mu ---
        # Avoid log2(0) by clamping to a tiny value.
        info_content = -np.log2(np.maximum(sorted_probs, 1e-30))
        mask = info_content <= mu

        # Always keep at least one token (the most probable).
        if not np.any(mask):
            mask[0] = True

        candidate_indices = sorted_indices[mask]
        candidate_probs = sorted_probs[mask]

        # --- Renormalize ---
        total = np.sum(candidate_probs)
        if total > 0:
            candidate_probs = candidate_probs / total

        num_candidates = len(candidate_indices)

        # --- CDF selection using ctx.u ---
        cdf = np.cumsum(candidate_probs)
        rank = int(np.searchsorted(cdf, ctx.u, side="left"))
        rank = min(rank, num_candidates - 1)

        selected_vocab_idx = int(candidate_indices[rank])
        selected_prob = float(candidate_probs[rank])

        # --- Update mu: surprise feedback loop ---
        # Use the original (un-normalized) probability for surprise.
        original_prob = float(probs[selected_vocab_idx])
        surprise = -np.log2(max(original_prob, 1e-30))
        error = surprise - tau
        new_mu = max(0.0, mu - eta * error)

        ctx.stage_state[_MU_KEY] = new_mu

        # --- Write selection results ---
        ctx.token_id = selected_vocab_idx
        ctx.token_rank = rank
        ctx.token_prob = selected_prob
        ctx.num_candidates = num_candidates

        if ctx.config.injection_verbose:
            _logger.debug(
                "Mirostat v2: mu=%.3f -> %.3f, surprise=%.3f, tau=%.3f, candidates=%d, rank=%d",
                mu,
                new_mu,
                surprise,
                tau,
                num_candidates,
                rank,
            )
