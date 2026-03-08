"""Logit perturbation injection method."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from qr_sampler.exceptions import EntropyUnavailableError

if TYPE_CHECKING:
    from qr_sampler.config import QRSamplerConfig
    from qr_sampler.entropy.base import EntropySource

_logger = logging.getLogger("qr_sampler")


class LogitPerturbation:
    """Direct per-logit quantum noise injection.

    Fetches vocab_size * 4 bytes from the entropy source, maps them to
    zero-mean unit-variance float32 noise via the probit transform, and
    adds them to logits before temperature scaling. Every noise dimension
    is independently quantum-derived -- no PRNG expansion.

    Matches the benchmark spec (Entropy Seeding Benchmark §2.2 Method 1):
        logits_out = logits + alpha * normalize(
            entropy_bytes_to_float(source.get_bytes(vocab_size * 4)))
    """

    @staticmethod
    def perturb(
        logits: np.ndarray[Any, np.dtype[np.floating[Any]]],
        entropy_source: EntropySource,
        config: QRSamplerConfig,
        alpha_override: float | None = None,
    ) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
        """Perturb logits with direct per-logit quantum noise.

        Fetches 4 bytes per logit, interprets them as uint32, maps to
        uniform floats in (eps, 1-eps), converts to zero-mean unit-variance
        noise via the probit transform, scales by alpha * sigma, and adds
        to logits.

        Args:
            logits: 1-D float array of raw logit values.
            entropy_source: Source of quantum entropy bytes.
            config: Sampler configuration (uses logit_perturbation_alpha,
                logit_perturbation_sigma, uniform_clamp_epsilon, injection_verbose).
            alpha_override: If provided, use this instead of config.logit_perturbation_alpha.

        Returns:
            Modified logits array (same shape as input). Returns input unchanged
            if alpha == 0 or entropy is unavailable.
        """
        alpha = alpha_override if alpha_override is not None else config.logit_perturbation_alpha
        if alpha == 0.0:
            return logits

        n = len(logits)
        try:
            raw_bytes = entropy_source.get_random_bytes(n * 4)
        except EntropyUnavailableError:
            _logger.warning("LogitPerturbation: entropy unavailable, skipping perturbation")
            return logits

        if len(raw_bytes) < n * 4:
            _logger.warning("LogitPerturbation: insufficient entropy bytes, skipping perturbation")
            return logits

        # Map raw uint32 bytes -> uniform floats in (eps, 1-eps).
        uint32s = np.frombuffer(raw_bytes[: n * 4], dtype=np.uint32)
        eps = config.uniform_clamp_epsilon
        u = uint32s.astype(np.float64) / (2**32)
        u = np.clip(u, eps, 1.0 - eps)

        # Probit (inverse normal CDF) -> zero-mean unit-variance noise.
        noise = _probit(u).astype(np.float32) * config.logit_perturbation_sigma
        result = logits + alpha * noise

        if config.injection_verbose:
            _logger.debug(
                "LogitPerturbation: alpha=%.4f sigma=%.4f n_bytes=%d",
                alpha,
                config.logit_perturbation_sigma,
                len(raw_bytes),
            )

        return result


def _probit(u: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Vectorised probit (inverse normal CDF) via rational approximation.

    Uses the Beasley-Springer-Moro approximation, accurate to ~1e-9.
    Input must be in (0, 1) -- clamp before calling.
    """
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308

    p = np.where(u < 0.5, u, 1.0 - u)
    t = np.sqrt(-2.0 * np.log(p))
    x = t - (c0 + c1 * t + c2 * t**2) / (1.0 + d1 * t + d2 * t**2 + d3 * t**3)
    return np.where(u < 0.5, -x, x)
