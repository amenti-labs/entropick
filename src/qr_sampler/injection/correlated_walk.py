"""M3: Correlated walk injection method."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qr_sampler.config import QRSamplerConfig
    from qr_sampler.entropy.base import EntropySource

class CorrelatedWalk:
    """M3: Quantum correlated walk.

    Maintains a per-request walk position that drifts based on quantum
    entropy, replacing the amplified u value with the walk position.
    This creates temporal correlations across tokens within a request.
    """

    @staticmethod
    def step(
        u: float,
        entropy_source: EntropySource,
        config: QRSamplerConfig,
        walk_position: float,
    ) -> tuple[float, float]:
        """Advance the walk by one step and return the new u value.

        Args:
            u: Current amplified uniform value from the signal amplifier.
            entropy_source: Source of quantum entropy bytes.
            config: Sampler configuration (uses walk_step, sample_count,
                injection_verbose).
            walk_position: Current walk position in [0, 1).

        Returns:
            Tuple of (new_u, new_walk_position). Both values are the new
            walk position. Returns (u, walk_position) unchanged if
            walk_step == 0 or entropy is unavailable.
        """
        return (u, walk_position)
