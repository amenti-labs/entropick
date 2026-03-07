"""M2: Temperature variance injection method."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qr_sampler.exceptions import EntropyUnavailableError
from qr_sampler.injection._entropy_utils import bytes_to_uniform

if TYPE_CHECKING:
    from qr_sampler.config import QRSamplerConfig
    from qr_sampler.entropy.base import EntropySource

_logger = logging.getLogger("qr_sampler")
_MIN_TEMPERATURE = 0.01


class TempVariance:
    """M2: Quantum temperature modulation.

    Modulates the computed temperature value using quantum entropy,
    introducing stochastic variation in the sharpness of the probability
    distribution.

    Formula: new_temp = temperature * (1 + beta * (u - 0.5))
    where u is the quantum-derived uniform value in (0, 1).
    Result is clamped to [0.01, inf) to prevent degenerate distributions.
    """

    @staticmethod
    def modulate(
        temperature: float,
        entropy_source: EntropySource,
        config: QRSamplerConfig,
        beta_override: float | None = None,
    ) -> float:
        """Modulate temperature with quantum entropy.

        Args:
            temperature: Base temperature value from the temperature strategy.
            entropy_source: Source of quantum entropy bytes.
            config: Sampler configuration (uses temp_variance_beta, sample_count,
                population_mean, population_std, injection_verbose).
            beta_override: If provided, use this instead of config.temp_variance_beta.

        Returns:
            Modulated temperature, clamped to >= 0.01. Returns input unchanged
            if beta == 0 or entropy is unavailable.
        """
        beta = beta_override if beta_override is not None else config.temp_variance_beta
        if beta == 0.0:
            return temperature

        try:
            raw_bytes = entropy_source.get_random_bytes(config.sample_count)
        except EntropyUnavailableError:
            _logger.warning("M2 TempVariance: entropy unavailable, skipping modulation")
            return temperature

        if not raw_bytes:
            _logger.warning("M2 TempVariance: empty entropy payload, skipping modulation")
            return temperature

        u = bytes_to_uniform(raw_bytes, config)

        # Modulate: scale temperature by (1 + beta * (u - 0.5))
        # u in (0,1) -> (u - 0.5) in (-0.5, 0.5) -> modulation in (-beta/2, beta/2)
        modulation = beta * (u - 0.5)
        new_temp = temperature * (1.0 + modulation)
        new_temp = max(_MIN_TEMPERATURE, new_temp)

        if config.injection_verbose:
            _logger.debug(
                "M2 TempVariance: beta=%.4f u=%.6f original=%.4f new=%.4f",
                beta,
                u,
                temperature,
                new_temp,
            )

        return new_temp
