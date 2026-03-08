"""GumbelSelectionStage -- Gumbel-Max trick for quantum-driven token selection."""

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


@StageRegistry.register("gumbel_selection")
class GumbelSelectionStage:
    """Select a token via the Gumbel-Max trick using quantum random noise.

    Instead of CDF binary search, this stage adds Gumbel noise to
    log-probabilities and takes the argmax.  Each candidate token
    receives an independent quantum-random Gumbel variate, making
    every selection a multi-dimensional quantum decision.

    Algorithm:
        1. Apply top-k filtering (if configured).
        2. Compute log-softmax: ``log(softmax(logits / temperature))``.
        3. Fetch one quantum float per finite candidate.
        4. Convert to Gumbel noise: ``g = -log(-log(u))``.
        5. Select: ``argmax(log_probs + g)``.

    Writes ``ctx.token_id``, ``ctx.token_rank``, ``ctx.token_prob``,
    and ``ctx.num_candidates``.

    When active, this replaces the normal CDF selection (the downstream
    ``SelectionStage`` checks ``ctx.token_id >= 0`` and skips).

    No-ops when ``config.gumbel_selection`` is ``False``.
    """

    name: str = "gumbel_selection"

    def __call__(self, ctx: SamplingContext) -> None:
        if not ctx.config.gumbel_selection:
            return

        temperature = ctx.temperature
        if temperature <= 0:
            # Greedy: pick the argmax directly.
            finite_mask = np.isfinite(ctx.row)
            if not np.any(finite_mask):
                return
            token_id = int(np.argmax(ctx.row))
            ctx.token_id = token_id
            ctx.token_rank = 0
            ctx.token_prob = 1.0
            ctx.num_candidates = 1
            return

        logits = ctx.row.copy()

        # --- Top-k filtering ---
        top_k = ctx.config.top_k
        if top_k > 0 and top_k < len(logits):
            threshold_idx = len(logits) - top_k
            partitioned = np.argpartition(logits, threshold_idx)
            below_k = partitioned[:threshold_idx]
            logits[below_k] = -np.inf

        # --- Identify finite candidates ---
        finite_mask = np.isfinite(logits)
        n_finite = int(np.sum(finite_mask))
        if n_finite == 0:
            return

        # --- Compute log-softmax ---
        probs = stable_softmax(logits)
        if probs is None:
            return

        # Compute probabilities (for ranking and reporting).
        log_probs = np.full_like(logits, -np.inf, dtype=np.float64)
        pos_mask = probs > 0
        log_probs[pos_mask] = np.log(probs[pos_mask])

        # --- Fetch quantum bytes: 8 bytes per candidate for float64 ---
        n_candidates = int(np.sum(pos_mask))
        if n_candidates == 0:
            return

        t_start = time.perf_counter_ns()
        try:
            raw_bytes = ctx.entropy_source.get_random_bytes(n_candidates * 8)
        except EntropyUnavailableError:
            _logger.warning("Gumbel selection: entropy unavailable, skipping")
            return
        t_end = time.perf_counter_ns()
        ctx.entropy_fetch_ms += (t_end - t_start) / 1_000_000.0

        if len(raw_bytes) < n_candidates * 8:
            _logger.warning("Gumbel selection: insufficient entropy bytes, skipping")
            return

        # Convert to uniform floats in (0, 1), clamped away from 0 and 1.
        u_values = np.frombuffer(raw_bytes, dtype=np.uint64).astype(np.float64)
        u_values = u_values / 18446744073709551616.0  # 2**64, maps [0, 2^64-1] -> [0, 1)
        eps = ctx.config.uniform_clamp_epsilon
        u_values = np.clip(u_values, eps, 1.0 - eps)

        # --- Compute Gumbel noise: g = -log(-log(u)) ---
        gumbel_noise = -np.log(-np.log(u_values))

        # --- Add Gumbel noise to log-probs of positive-probability tokens ---
        candidate_indices = np.where(pos_mask)[0]
        perturbed = np.full_like(log_probs, -np.inf)
        perturbed[candidate_indices] = log_probs[candidate_indices] + gumbel_noise

        # --- Select token: argmax ---
        selected_idx = int(np.argmax(perturbed))

        # --- Compute rank and probability ---
        sorted_indices = np.argsort(probs)[::-1]
        rank = int(np.where(sorted_indices == selected_idx)[0][0])

        ctx.token_id = selected_idx
        ctx.token_rank = rank
        ctx.token_prob = float(probs[selected_idx])
        ctx.num_candidates = n_candidates

        if ctx.config.injection_verbose:
            _logger.debug(
                "Gumbel selection: token_id=%d, rank=%d, prob=%.4f, candidates=%d",
                selected_idx,
                rank,
                ctx.token_prob,
                n_candidates,
            )
