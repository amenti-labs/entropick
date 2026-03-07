"""M3: Correlated walk injection method."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qr_sampler.exceptions import EntropyUnavailableError
from qr_sampler.injection._entropy_utils import bytes_to_uniform

if TYPE_CHECKING:
    from qr_sampler.config import QRSamplerConfig
    from qr_sampler.entropy.base import EntropySource

_logger = logging.getLogger("qr_sampler")


class CorrelatedWalk:
    """M3: Quantum correlated walk.

    Maintains a per-request walk position that drifts based on quantum
    entropy, replacing the amplified u value with the walk position.
    This creates temporal correlations across tokens within a request.

    The walk position evolves as:
        new_position = (walk_position + step * (qval - 0.5)) % 1.0
    where qval is the quantum-derived uniform value in (0, 1).
    The modulo wraps the position to stay in [0, 1).
    """

    @staticmethod
    def step(
        u: float,
        entropy_source: EntropySource,
        config: QRSamplerConfig,
        walk_position: float,
        step_override: float | None = None,
    ) -> tuple[float, float]:
        """Advance the walk by one step and return the new u value.

        Args:
            u: Current amplified uniform value from the signal amplifier.
            entropy_source: Source of quantum entropy bytes.
            config: Sampler configuration (uses walk_step, sample_count,
                population_mean, population_std, injection_verbose).
            walk_position: Current walk position in [0, 1).
            step_override: If provided, use this instead of config.walk_step.

        Returns:
            Tuple of (new_u, new_walk_position). Both values are the new
            walk position. Returns (u, walk_position) unchanged if
            step == 0 or entropy is unavailable.
        """
        walk_step = step_override if step_override is not None else config.walk_step
        if walk_step == 0.0:
            return (u, walk_position)

        try:
            raw_bytes = entropy_source.get_random_bytes(config.sample_count)
        except EntropyUnavailableError:
            _logger.warning("M3 CorrelatedWalk: entropy unavailable, skipping step")
            return (u, walk_position)

        if not raw_bytes:
            _logger.warning("M3 CorrelatedWalk: empty entropy payload, skipping step")
            return (u, walk_position)

        qval = bytes_to_uniform(raw_bytes, config)

        # Update walk position: drift by step * (qval - 0.5)
        # qval in (0,1) -> (qval - 0.5) in (-0.5, 0.5) -> drift in (-step/2, step/2)
        new_position = walk_position + walk_step * (qval - 0.5)

        # Wrap to [0, 1) using modulo -- Python's % handles negatives correctly:
        # e.g., -0.1 % 1.0 == 0.9
        new_position = new_position % 1.0

        if config.injection_verbose:
            _logger.debug(
                "M3 CorrelatedWalk: step=%.4f qval=%.6f old_pos=%.6f new_pos=%.6f",
                walk_step,
                qval,
                walk_position,
                new_position,
            )

        return (new_position, new_position)
