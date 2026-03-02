"""Tests for M1: LogitNoise injection method."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from qr_sampler.config import QRSamplerConfig
from qr_sampler.entropy.mock import MockUniformSource
from qr_sampler.exceptions import EntropyUnavailableError
from qr_sampler.injection.logit_noise import LogitNoise


@pytest.fixture()
def source() -> MockUniformSource:
    """Seeded mock entropy source for reproducible tests."""
    return MockUniformSource(seed=42)


@pytest.fixture()
def logits() -> np.ndarray:
    """Sample logits array with clear probability structure."""
    return np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0])


class TestLogitNoise:
    """Tests for LogitNoise.perturb()."""

    def test_perturb_modifies_logits_when_enabled(
        self,
        source: MockUniformSource,
        logits: np.ndarray,
    ) -> None:
        """With alpha=0.05, logits should be modified."""
        config = QRSamplerConfig(
            _env_file=None,  # type: ignore[call-arg]
            logit_noise_alpha=0.05,
            logit_noise_sigma=1.0,
        )
        original = logits.copy()
        result = LogitNoise.perturb(logits, source, config)
        assert not np.array_equal(result, original)

    def test_perturb_noop_when_disabled(
        self,
        source: MockUniformSource,
        logits: np.ndarray,
    ) -> None:
        """With alpha=0.0, logits should be returned unchanged."""
        config = QRSamplerConfig(
            _env_file=None,  # type: ignore[call-arg]
            logit_noise_alpha=0.0,
        )
        original = logits.copy()
        result = LogitNoise.perturb(logits, source, config)
        np.testing.assert_array_equal(result, original)

    def test_perturb_reproducible_with_same_seed(
        self,
        logits: np.ndarray,
    ) -> None:
        """Two calls with identically-seeded sources produce the same result."""
        config = QRSamplerConfig(
            _env_file=None,  # type: ignore[call-arg]
            logit_noise_alpha=0.05,
            logit_noise_sigma=1.0,
        )
        source_a = MockUniformSource(seed=42)
        source_b = MockUniformSource(seed=42)
        result_a = LogitNoise.perturb(logits.copy(), source_a, config)
        result_b = LogitNoise.perturb(logits.copy(), source_b, config)
        np.testing.assert_array_equal(result_a, result_b)

    def test_perturb_scales_with_alpha(
        self,
        logits: np.ndarray,
    ) -> None:
        """Larger alpha should produce larger perturbation magnitude."""
        small_config = QRSamplerConfig(
            _env_file=None,  # type: ignore[call-arg]
            logit_noise_alpha=0.01,
            logit_noise_sigma=1.0,
        )
        large_config = QRSamplerConfig(
            _env_file=None,  # type: ignore[call-arg]
            logit_noise_alpha=1.0,
            logit_noise_sigma=1.0,
        )
        source_small = MockUniformSource(seed=42)
        source_large = MockUniformSource(seed=42)
        result_small = LogitNoise.perturb(logits.copy(), source_small, small_config)
        result_large = LogitNoise.perturb(logits.copy(), source_large, large_config)

        diff_small = np.abs(result_small - logits).max()
        diff_large = np.abs(result_large - logits).max()
        assert diff_large > diff_small

    def test_perturb_handles_entropy_unavailable(
        self,
        logits: np.ndarray,
    ) -> None:
        """When entropy source raises, returns logits unchanged."""
        config = QRSamplerConfig(
            _env_file=None,  # type: ignore[call-arg]
            logit_noise_alpha=0.05,
        )
        failing_source = MagicMock()
        failing_source.get_random_bytes.side_effect = EntropyUnavailableError("test")
        original = logits.copy()
        result = LogitNoise.perturb(logits, failing_source, config)
        np.testing.assert_array_equal(result, original)

    def test_perturb_preserves_shape(
        self,
        source: MockUniformSource,
        logits: np.ndarray,
    ) -> None:
        """Result shape must match input shape."""
        config = QRSamplerConfig(
            _env_file=None,  # type: ignore[call-arg]
            logit_noise_alpha=0.1,
        )
        result = LogitNoise.perturb(logits, source, config)
        assert result.shape == logits.shape
