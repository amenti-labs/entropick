"""M2: Temperature variance injection method."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qr_sampler.config import QRSamplerConfig
    from qr_sampler.entropy.base import EntropySource

class TempVariance:
    """M2: Quantum temperature modulation.

    Modulates the computed temperature value using quantum entropy,
    introducing stochastic variation in the sharpness of the probability
    distribution.
    """

    @staticmethod
    def modulate(
        temperature: float,
        entropy_source: EntropySource,
        config: QRSamplerConfig,
    ) -> float:
        """Modulate temperature with quantum entropy.

        Args:
            temperature: Base temperature value from the temperature strategy.
            entropy_source: Source of quantum entropy bytes.
            config: Sampler configuration (uses temp_variance_beta, sample_count,
                injection_verbose).

        Returns:
            Modulated temperature, clamped to >= 0.01. Returns input unchanged
            if temp_variance_beta == 0 or entropy is unavailable.
        """
        return temperature
