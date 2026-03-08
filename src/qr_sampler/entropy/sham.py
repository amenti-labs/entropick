"""Sham QRNG source for experimental controls.

Uses ``os.urandom()`` but simulates QRNG latency by sleeping for a
configurable duration. Essential for double-blind consciousness experiments
-- the experimenter cannot distinguish sham from real QRNG by observing
timing behaviour.

Configure via ``QR_SHAM_QRNG_LATENCY_MS`` (set at source init, not per-request).
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

from qr_sampler.entropy.base import EntropySource
from qr_sampler.entropy.registry import register_entropy_source
from qr_sampler.exceptions import EntropyUnavailableError

if TYPE_CHECKING:
    from qr_sampler.config import QRSamplerConfig


@register_entropy_source("sham_qrng")
class ShamQrngSource(EntropySource):
    """Sham QRNG for experimental controls.

    Uses ``os.urandom()`` but simulates QRNG latency by sleeping.
    Essential for double-blind consciousness experiments -- the
    experimenter cannot distinguish sham from real QRNG.

    Configure via ``QR_SHAM_QRNG_LATENCY_MS`` (set at source init, not per-request).

    Args:
        config: Sampler configuration with ``sham_qrng_latency_ms``.
    """

    def __init__(self, config: QRSamplerConfig) -> None:
        self._latency_s = config.sham_qrng_latency_ms / 1000.0
        self._closed = False

    @property
    def name(self) -> str:
        """Return ``'sham_qrng'``."""
        return "sham_qrng"

    @property
    def is_available(self) -> bool:
        """Whether the source is open and can provide entropy."""
        return not self._closed

    def get_random_bytes(self, n: int) -> bytes:
        """Return *n* bytes from ``os.urandom()`` after simulating QRNG latency.

        Sleeps for ``sham_qrng_latency_ms`` milliseconds before generating
        bytes, mimicking the network round-trip of a real QRNG server.

        Args:
            n: Number of random bytes to generate.

        Returns:
            Exactly *n* bytes of entropy from ``os.urandom()``.

        Raises:
            EntropyUnavailableError: If the source has been closed.
        """
        if self._closed:
            raise EntropyUnavailableError("ShamQrngSource is closed")
        if self._latency_s > 0:
            time.sleep(self._latency_s)
        return os.urandom(n)

    def close(self) -> None:
        """Mark the source as closed."""
        self._closed = True

    def health_check(self) -> dict[str, Any]:
        """Return health status including simulated latency.

        Returns:
            Dictionary with source name, availability, and configured latency.
        """
        return {
            "source": "sham_qrng",
            "healthy": self.is_available,
            "latency_ms": self._latency_s * 1000,
        }
