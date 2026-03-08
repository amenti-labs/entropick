"""Tests for amplifier calibration utilities."""

from __future__ import annotations

from qr_sampler.amplification.calibration import (
    calibrate_population_stats,
    measure_entropy_rate,
)
from qr_sampler.entropy.mock import MockUniformSource
from qr_sampler.entropy.system import SystemEntropySource


class TestCalibratePopulationStats:
    """Tests for calibrate_population_stats()."""

    def test_mock_source_mean_near_127_5(self) -> None:
        """With default MockUniformSource(mean=127.5), measured mean should be close."""
        source = MockUniformSource(mean=127.5, seed=42)
        stats = calibrate_population_stats(source, n_samples=100, bytes_per_sample=1024)
        assert abs(stats["mean"] - 127.5) < 3.0

    def test_mock_source_std_near_expected(self) -> None:
        """With MockUniformSource(std=40), measured std should be close to 40."""
        source = MockUniformSource(mean=127.5, seed=42)
        stats = calibrate_population_stats(source, n_samples=100, bytes_per_sample=1024)
        # MockUniformSource uses _MOCK_BYTE_STD=40.0, but clamping to [0,255]
        # can reduce the effective std slightly.
        assert abs(stats["std"] - 40.0) < 5.0

    def test_system_source_mean_near_127_5(self) -> None:
        """os.urandom() should produce mean near 127.5."""
        source = SystemEntropySource()
        stats = calibrate_population_stats(source, n_samples=50, bytes_per_sample=1024)
        assert abs(stats["mean"] - 127.5) < 3.0

    def test_system_source_std_near_73_6(self) -> None:
        """os.urandom() should produce std near 73.6 (uniform [0,255])."""
        source = SystemEntropySource()
        stats = calibrate_population_stats(source, n_samples=50, bytes_per_sample=1024)
        assert abs(stats["std"] - 73.6) < 3.0

    def test_min_max_range(self) -> None:
        """Min should be >= 0 and max should be <= 255."""
        source = SystemEntropySource()
        stats = calibrate_population_stats(source, n_samples=50, bytes_per_sample=1024)
        assert stats["min"] >= 0.0
        assert stats["max"] <= 255.0

    def test_n_bytes_total(self) -> None:
        """Total byte count should match n_samples * bytes_per_sample."""
        source = MockUniformSource(seed=0)
        stats = calibrate_population_stats(source, n_samples=10, bytes_per_sample=512)
        assert stats["n_bytes_total"] == 10 * 512

    def test_biased_source_shows_shifted_mean(self) -> None:
        """A biased MockUniformSource should show a shifted mean."""
        source = MockUniformSource(mean=140.0, seed=42)
        stats = calibrate_population_stats(source, n_samples=100, bytes_per_sample=1024)
        assert stats["mean"] > 135.0

    def test_returns_dict_with_expected_keys(self) -> None:
        """Result should contain all expected keys."""
        source = MockUniformSource(seed=0)
        stats = calibrate_population_stats(source, n_samples=5, bytes_per_sample=100)
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "n_bytes_total" in stats


class TestMeasureEntropyRate:
    """Tests for measure_entropy_rate()."""

    def test_system_source_high_entropy(self) -> None:
        """os.urandom() should have near-maximum entropy rate."""
        source = SystemEntropySource()
        result = measure_entropy_rate(source, n_bytes=50_000)
        # True random should be ~8.0 bits/byte (incompressible).
        assert result["bits_per_byte"] > 7.5

    def test_system_source_compression_ratio(self) -> None:
        """Random data should not compress well (ratio near or above 1.0)."""
        source = SystemEntropySource()
        result = measure_entropy_rate(source, n_bytes=50_000)
        # zlib adds a small header, so ratio should be >= ~0.99
        assert result["compression_ratio"] >= 0.98

    def test_min_entropy_estimate(self) -> None:
        """Min-entropy for os.urandom() should be near 8.0."""
        source = SystemEntropySource()
        result = measure_entropy_rate(source, n_bytes=50_000)
        # For uniform bytes, -log2(1/256) = 8.0. With finite samples,
        # the most frequent byte will appear slightly more, so min-entropy
        # will be slightly less than 8.0.
        assert result["estimated_min_entropy"] > 7.0

    def test_returns_dict_with_expected_keys(self) -> None:
        """Result should contain all expected keys."""
        source = MockUniformSource(seed=0)
        result = measure_entropy_rate(source, n_bytes=10_000)
        assert "bits_per_byte" in result
        assert "compression_ratio" in result
        assert "estimated_min_entropy" in result

    def test_bits_per_byte_capped_at_8(self) -> None:
        """bits_per_byte should never exceed 8.0."""
        source = SystemEntropySource()
        result = measure_entropy_rate(source, n_bytes=50_000)
        assert result["bits_per_byte"] <= 8.0

    def test_mock_source_still_high_entropy(self) -> None:
        """MockUniformSource with default mean should still be fairly random."""
        source = MockUniformSource(mean=127.5, seed=42)
        result = measure_entropy_rate(source, n_bytes=50_000)
        # Mock source uses normal distribution, not truly uniform, but should
        # still have reasonable entropy.
        assert result["bits_per_byte"] > 6.0
