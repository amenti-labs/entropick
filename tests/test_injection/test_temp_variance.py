"""Tests for M2: TempVariance injection method."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from qr_sampler.config import QRSamplerConfig
from qr_sampler.entropy.mock import MockUniformSource
from qr_sampler.exceptions import EntropyUnavailableError
from qr_sampler.injection.temp_variance import TempVariance


@pytest.fixture()
def source() -> MockUniformSource:
    """Seeded mock entropy source for reproducible tests."""
    return MockUniformSource(seed=42)


class TestTempVariance:
    """Tests for TempVariance.modulate()."""

    def test_modulate_changes_temperature_when_enabled(
        self,
        source: MockUniformSource,
    ) -> None:
        """With beta=0.3, temperature should be modified."""
        config = QRSamplerConfig(
            _env_file=None,  # type: ignore[call-arg]
            temp_variance_beta=0.3,
        )
        result = TempVariance.modulate(0.7, source, config)
        assert result != pytest.approx(0.7)

    def test_modulate_noop_when_disabled(
        self,
        source: MockUniformSource,
    ) -> None:
        """With beta=0.0, temperature should be returned unchanged."""
        config = QRSamplerConfig(
            _env_file=None,  # type: ignore[call-arg]
            temp_variance_beta=0.0,
        )
        result = TempVariance.modulate(0.7, source, config)
        assert result == pytest.approx(0.7)

    def test_modulate_never_below_minimum(
        self,
        source: MockUniformSource,
    ) -> None:
        """Even with large beta, result must be >= 0.01."""
        config = QRSamplerConfig(
            _env_file=None,  # type: ignore[call-arg]
            temp_variance_beta=2.0,
        )
        # Use a very small starting temperature to stress the clamp.
        result = TempVariance.modulate(0.02, source, config)
        assert result >= 0.01

    def test_modulate_range_bounded_by_beta(self) -> None:
        """Modulation should be bounded by beta/2 of the original temperature."""
        beta = 0.5
        temperature = 1.0
        config = QRSamplerConfig(
            _env_file=None,  # type: ignore[call-arg]
            temp_variance_beta=beta,
        )
        # Run many samples to check the range empirically.
        results = []
        for seed in range(100):
            src = MockUniformSource(seed=seed)
            results.append(TempVariance.modulate(temperature, src, config))

        # modulation = beta * (u - 0.5), u in (0,1)
        # new_temp = temperature * (1 + modulation)
        # Theoretical range: temperature * (1 - beta/2) to temperature * (1 + beta/2)
        min_expected = temperature * (1 - beta / 2)
        max_expected = temperature * (1 + beta / 2)
        for r in results:
            assert r >= min_expected - 0.01, f"{r} < {min_expected}"
            assert r <= max_expected + 0.01, f"{r} > {max_expected}"

    def test_modulate_handles_entropy_unavailable(self) -> None:
        """When entropy source raises, returns temperature unchanged."""
        config = QRSamplerConfig(
            _env_file=None,  # type: ignore[call-arg]
            temp_variance_beta=0.3,
        )
        failing_source = MagicMock()
        failing_source.get_random_bytes.side_effect = EntropyUnavailableError("test")
        result = TempVariance.modulate(0.7, failing_source, config)
        assert result == pytest.approx(0.7)

    def test_modulate_reproducible_with_same_seed(self) -> None:
        """Two calls with identically-seeded sources produce the same result."""
        config = QRSamplerConfig(
            _env_file=None,  # type: ignore[call-arg]
            temp_variance_beta=0.3,
        )
        source_a = MockUniformSource(seed=42)
        source_b = MockUniformSource(seed=42)
        result_a = TempVariance.modulate(0.7, source_a, config)
        result_b = TempVariance.modulate(0.7, source_b, config)
        assert result_a == pytest.approx(result_b)
