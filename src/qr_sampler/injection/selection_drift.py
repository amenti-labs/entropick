"""Selection drift injection method."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qr_sampler.exceptions import EntropyUnavailableError
from qr_sampler.injection._entropy_utils import bytes_to_uniform

if TYPE_CHECKING:
    from qr_sampler.config import QRSamplerConfig
    from qr_sampler.entropy.base import EntropySource

_logger = logging.getLogger("qr_sampler")


class SelectionDrift:
    """Quantum selection drift.

    Maintains a per-request drift position that drifts based on quantum
    entropy, replacing the amplified u value with the drift position.
    This creates temporal correlations across tokens within a request.

    The drift position evolves as:
        new_position = (drift_position + step * (qval - 0.5)) % 1.0
    where qval is the quantum-derived uniform value in (0, 1).
    The modulo wraps the position to stay in [0, 1).
    """

    @staticmethod
    def step(
        u: float,
        entropy_source: EntropySource,
        config: QRSamplerConfig,
        drift_position: float,
        step_override: float | None = None,
    ) -> tuple[float, float]:
        """Advance the drift by one step and return the new u value.

        Args:
            u: Current amplified uniform value from the signal amplifier.
            entropy_source: Source of quantum entropy bytes.
            config: Sampler configuration (uses drift_step, sample_count,
                population_mean, population_std, injection_verbose).
            drift_position: Current drift position in [0, 1).
            step_override: If provided, use this instead of config.drift_step.

        Returns:
            Tuple of (new_u, new_drift_position). Both values are the new
            drift position. Returns (u, drift_position) unchanged if
            step == 0 or entropy is unavailable.
        """
        drift_step = step_override if step_override is not None else config.drift_step
        if drift_step == 0.0:
            return (u, drift_position)

        try:
            raw_bytes = entropy_source.get_random_bytes(config.sample_count)
        except EntropyUnavailableError:
            _logger.warning("SelectionDrift: entropy unavailable, skipping step")
            return (u, drift_position)

        if not raw_bytes:
            _logger.warning("SelectionDrift: empty entropy payload, skipping step")
            return (u, drift_position)

        qval = bytes_to_uniform(raw_bytes, config)

        # Update drift position: drift by step * (qval - 0.5)
        # qval in (0,1) -> (qval - 0.5) in (-0.5, 0.5) -> drift in (-step/2, step/2)
        new_position = drift_position + drift_step * (qval - 0.5)

        # Wrap to [0, 1) using modulo -- Python's % handles negatives correctly:
        # e.g., -0.1 % 1.0 == 0.9
        new_position = new_position % 1.0

        if config.injection_verbose:
            _logger.debug(
                "SelectionDrift: step=%.4f qval=%.6f old_pos=%.6f new_pos=%.6f",
                drift_step,
                qval,
                drift_position,
                new_position,
            )

        return (new_position, new_position)
