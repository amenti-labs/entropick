"""Shared entropy-to-uniform conversion for injection methods."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from qr_sampler.config import QRSamplerConfig

_SQRT2 = math.sqrt(2.0)


def bytes_to_uniform(raw_bytes: bytes, config: QRSamplerConfig) -> float:
    """Convert raw entropy bytes to a uniform float in (eps, 1-eps).

    Uses the same z-score -> normal CDF pipeline as ZScoreMeanAmplifier:
        1. Interpret bytes as uint8 samples
        2. Compute sample mean
        3. Z-score = (mean - population_mean) / SEM
        4. Normal CDF -> uniform in (0, 1)
        5. Clamp to (eps, 1-eps)

    Args:
        raw_bytes: Raw entropy bytes from an entropy source.
        config: Sampler configuration (uses population_mean, population_std,
            uniform_clamp_epsilon).

    Returns:
        Uniform float in (eps, 1-eps).
    """
    samples = np.frombuffer(raw_bytes, dtype=np.uint8)
    n = len(samples)
    sample_mean = float(np.mean(samples))
    sem = config.population_std / math.sqrt(n)
    z_score = (sample_mean - config.population_mean) / sem
    u = 0.5 * (1.0 + math.erf(z_score / _SQRT2))
    eps = config.uniform_clamp_epsilon
    return max(eps, min(1.0 - eps, u))
