"""Tests for ShamQrngSource."""

from __future__ import annotations

import time

import pytest

from qr_sampler.config import QRSamplerConfig
from qr_sampler.entropy.sham import ShamQrngSource
from qr_sampler.exceptions import EntropyUnavailableError


@pytest.fixture()
def config_with_latency() -> QRSamplerConfig:
    """Config with 50ms simulated QRNG latency."""
    return QRSamplerConfig(sham_qrng_latency_ms=50.0, _env_file=None)  # type: ignore[call-arg]


@pytest.fixture()
def config_no_latency() -> QRSamplerConfig:
    """Config with zero latency (no sleep)."""
    return QRSamplerConfig(sham_qrng_latency_ms=0.0, _env_file=None)  # type: ignore[call-arg]


class TestShamQrngSource:
    """Tests for the sham QRNG control source."""

    def test_name(self, config_no_latency: QRSamplerConfig) -> None:
        source = ShamQrngSource(config_no_latency)
        assert source.name == "sham_qrng"

    def test_is_available_when_open(self, config_no_latency: QRSamplerConfig) -> None:
        source = ShamQrngSource(config_no_latency)
        assert source.is_available is True

    def test_is_unavailable_after_close(self, config_no_latency: QRSamplerConfig) -> None:
        source = ShamQrngSource(config_no_latency)
        source.close()
        assert source.is_available is False

    def test_returns_correct_byte_count(self, config_no_latency: QRSamplerConfig) -> None:
        source = ShamQrngSource(config_no_latency)
        for n in (0, 1, 10, 100, 1024, 20480):
            data = source.get_random_bytes(n)
            assert len(data) == n

    def test_returns_bytes_type(self, config_no_latency: QRSamplerConfig) -> None:
        source = ShamQrngSource(config_no_latency)
        data = source.get_random_bytes(16)
        assert isinstance(data, bytes)

    def test_consecutive_calls_differ(self, config_no_latency: QRSamplerConfig) -> None:
        """Two calls with enough bytes should produce different output."""
        source = ShamQrngSource(config_no_latency)
        a = source.get_random_bytes(32)
        b = source.get_random_bytes(32)
        # Statistically near-impossible for 32 random bytes to be equal.
        assert a != b

    def test_closed_source_raises(self, config_no_latency: QRSamplerConfig) -> None:
        """get_random_bytes() should raise after close()."""
        source = ShamQrngSource(config_no_latency)
        source.close()
        with pytest.raises(EntropyUnavailableError, match="closed"):
            source.get_random_bytes(16)

    def test_latency_simulation(self, config_with_latency: QRSamplerConfig) -> None:
        """With 50ms latency, a call should take at least ~50ms."""
        source = ShamQrngSource(config_with_latency)
        t0 = time.perf_counter()
        source.get_random_bytes(16)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        # Allow some tolerance — the sleep should add at least 40ms.
        assert elapsed_ms >= 40.0

    def test_no_latency_is_fast(self, config_no_latency: QRSamplerConfig) -> None:
        """With 0ms latency, calls should be very fast."""
        source = ShamQrngSource(config_no_latency)
        t0 = time.perf_counter()
        source.get_random_bytes(1024)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        # Should complete in well under 10ms without sleep.
        assert elapsed_ms < 10.0

    def test_health_check(self, config_with_latency: QRSamplerConfig) -> None:
        source = ShamQrngSource(config_with_latency)
        health = source.health_check()
        assert health["source"] == "sham_qrng"
        assert health["healthy"] is True
        assert health["latency_ms"] == pytest.approx(50.0)

    def test_health_check_after_close(self, config_no_latency: QRSamplerConfig) -> None:
        source = ShamQrngSource(config_no_latency)
        source.close()
        health = source.health_check()
        assert health["healthy"] is False

    def test_registry_registration(self) -> None:
        """ShamQrngSource should be registered as 'sham_qrng'."""
        from qr_sampler.entropy.registry import EntropySourceRegistry

        klass = EntropySourceRegistry.get("sham_qrng")
        assert klass is ShamQrngSource

    def test_is_subclass_of_abc(self) -> None:
        """ShamQrngSource should be an EntropySource subclass."""
        from qr_sampler.entropy.base import EntropySource

        assert issubclass(ShamQrngSource, EntropySource)

    def test_get_random_float64(self, config_no_latency: QRSamplerConfig) -> None:
        import numpy as np

        source = ShamQrngSource(config_no_latency)
        result = source.get_random_float64((10,))
        assert result.shape == (10,)
        assert result.dtype == np.float64
        assert (result >= 0.0).all()
        assert (result <= 1.0).all()
