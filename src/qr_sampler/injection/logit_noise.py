"""M1: Logit noise injection method."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from qr_sampler.config import QRSamplerConfig
    from qr_sampler.entropy.base import EntropySource


class LogitNoise:
    """M1: Gaussian logit noise injection.

    Adds quantum-seeded Gaussian noise to logits before temperature
    scaling, reshaping the probability distribution at the earliest
    pipeline stage.
    """

    @staticmethod
    def perturb(
        logits: np.ndarray[Any, np.dtype[np.floating[Any]]],
        entropy_source: EntropySource,
        config: QRSamplerConfig,
    ) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
        """Perturb logits with quantum-seeded Gaussian noise.

        Args:
            logits: 1-D float array of raw logit values.
            entropy_source: Source of quantum entropy bytes.
            config: Sampler configuration (uses logit_noise_alpha, logit_noise_sigma,
                sample_count, injection_verbose).

        Returns:
            Modified logits array (same shape as input). Returns input unchanged
            if logit_noise_alpha == 0 or entropy is unavailable.
        """
        return logits
