"""Tests for analysis.statistics -- known-value tests for each statistic."""

from __future__ import annotations

import numpy as np
import pytest

try:
    import scipy  # noqa: F401

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from qr_sampler.analysis.statistics import (
    approximate_entropy,
    autocorrelation_test,
    bayesian_sequential,
    chi_square_rank_test,
    cumulative_deviation,
    entropy_rate,
    hurst_exponent,
    runs_test,
    serial_correlation,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

pytestmark = [
    pytest.mark.skipif(not HAS_SCIPY, reason="scipy required"),
    pytest.mark.filterwarnings("ignore::RuntimeWarning"),
]


@pytest.fixture()
def iid_uniform() -> np.ndarray:
    """1000 IID Uniform(0,1) values -- no structure."""
    rng = np.random.default_rng(42)
    return rng.uniform(0.0, 1.0, size=1000)


@pytest.fixture()
def ar1_process() -> np.ndarray:
    """AR(1) with rho=0.8 -- strong autocorrelation."""
    rng = np.random.default_rng(42)
    n = 1000
    x = np.zeros(n)
    x[0] = rng.normal()
    for i in range(1, n):
        x[i] = 0.8 * x[i - 1] + rng.normal() * 0.2
    # Rescale to (0, 1) via CDF of empirical distribution
    from scipy import stats  # type: ignore[import-untyped]

    return stats.norm.cdf(x)


@pytest.fixture()
def biased_uniform() -> np.ndarray:
    """1000 values with mean shifted to ~0.55."""
    rng = np.random.default_rng(42)
    return np.clip(rng.uniform(0.05, 1.05, size=1000), 0.0, 1.0)


# ---------------------------------------------------------------------------
# autocorrelation_test
# ---------------------------------------------------------------------------


class TestAutocorrelation:
    """Tests for autocorrelation_test."""

    def test_autocorrelation_random_data(self, iid_uniform: np.ndarray) -> None:
        """Random uniform data should have p > 0.05 (no autocorrelation)."""
        result = autocorrelation_test(iid_uniform)
        assert result["p_value"] > 0.05

    def test_autocorrelation_correlated_data(self, ar1_process: np.ndarray) -> None:
        """Correlated data (AR(1)) should have low p-value."""
        result = autocorrelation_test(ar1_process)
        assert result["p_value"] < 0.01

    def test_returns_expected_keys(self, iid_uniform: np.ndarray) -> None:
        """Result dict has all documented keys."""
        result = autocorrelation_test(iid_uniform, max_lag=5)
        assert set(result.keys()) == {"statistic", "p_value", "lags", "autocorrelations"}
        assert len(result["lags"]) == 5
        assert len(result["autocorrelations"]) == 5

    def test_constant_input(self) -> None:
        """Constant input returns p_value = 1.0 (zero variance)."""
        result = autocorrelation_test(np.full(100, 0.5))
        assert result["p_value"] == 1.0


# ---------------------------------------------------------------------------
# runs_test
# ---------------------------------------------------------------------------


class TestRunsTest:
    """Tests for runs_test."""

    def test_runs_test_random(self, iid_uniform: np.ndarray) -> None:
        """Random data passes runs test (p > 0.05)."""
        result = runs_test(iid_uniform)
        assert result["p_value"] > 0.05

    def test_runs_test_alternating(self) -> None:
        """Perfectly alternating data has many runs, detected as non-random."""
        alt = np.array([0.1, 0.9] * 500)
        result = runs_test(alt)
        assert result["p_value"] < 0.01
        assert result["n_runs"] > result["expected_runs"]

    def test_returns_expected_keys(self, iid_uniform: np.ndarray) -> None:
        """Result dict has all documented keys."""
        result = runs_test(iid_uniform)
        assert set(result.keys()) == {"n_runs", "expected_runs", "z_score", "p_value"}

    def test_all_same_value(self) -> None:
        """All identical values returns p_value = 1.0."""
        result = runs_test(np.full(100, 0.5))
        assert result["p_value"] == 1.0


# ---------------------------------------------------------------------------
# serial_correlation
# ---------------------------------------------------------------------------


class TestSerialCorrelation:
    """Tests for serial_correlation."""

    def test_serial_correlation_independent(self, iid_uniform: np.ndarray) -> None:
        """Independent data has low lag-1 correlation (abs < 0.1)."""
        result = serial_correlation(iid_uniform, lag=1)
        assert abs(result["correlation"]) < 0.1

    def test_ar1_positive(self, ar1_process: np.ndarray) -> None:
        """AR(1) with rho=0.8 has strong positive lag-1 correlation."""
        result = serial_correlation(ar1_process, lag=1)
        assert result["correlation"] > 0.3
        assert result["p_value"] < 0.01

    def test_short_array(self) -> None:
        """Array shorter than lag returns zero correlation."""
        result = serial_correlation(np.array([0.5]), lag=2)
        assert result["correlation"] == 0.0
        assert result["p_value"] == 1.0


# ---------------------------------------------------------------------------
# hurst_exponent
# ---------------------------------------------------------------------------


class TestHurstExponent:
    """Tests for hurst_exponent."""

    def test_hurst_exponent_random(self, iid_uniform: np.ndarray) -> None:
        """Random walk gives Hurst exponent ~0.5."""
        result = hurst_exponent(iid_uniform)
        assert 0.3 < result["hurst"] < 0.7

    def test_persistent_series(self) -> None:
        """Cumulative sum (random walk) shows persistence (H > 0.5)."""
        rng = np.random.default_rng(42)
        walk = np.cumsum(rng.normal(0, 1, 2000))
        result = hurst_exponent(walk)
        assert result["hurst"] > 0.5
        assert result["interpretation"] == "persistent"

    def test_hurst_exponent_insufficient_data(self) -> None:
        """Short array returns 0.5 with 'insufficient_data' interpretation."""
        result = hurst_exponent(np.array([0.1, 0.2, 0.3]))
        assert result["hurst"] == 0.5
        assert result["interpretation"] == "insufficient_data"

    def test_returns_expected_keys(self, iid_uniform: np.ndarray) -> None:
        """Result dict has hurst and interpretation."""
        result = hurst_exponent(iid_uniform)
        assert "hurst" in result
        assert "interpretation" in result


# ---------------------------------------------------------------------------
# approximate_entropy
# ---------------------------------------------------------------------------


class TestApproximateEntropy:
    """Tests for approximate_entropy."""

    def test_approximate_entropy_random(self) -> None:
        """Random data has moderate-high ApEn (higher than periodic)."""
        rng = np.random.default_rng(42)
        random_data = rng.uniform(0, 1, 300)
        periodic = np.tile([0.2, 0.8, 0.2, 0.8], 75)

        apen_random = approximate_entropy(random_data)["apen"]
        apen_periodic = approximate_entropy(periodic)["apen"]

        assert apen_random > apen_periodic
        assert apen_random > 0.0

    def test_constant_series(self) -> None:
        """Constant series returns zero ApEn."""
        result = approximate_entropy(np.full(100, 0.5))
        assert result["apen"] == 0.0
        assert result["interpretation"] == "constant_series"

    def test_insufficient_data(self) -> None:
        """Very short series returns insufficient_data."""
        result = approximate_entropy(np.array([0.5]), m=2)
        assert result["interpretation"] == "insufficient_data"


# ---------------------------------------------------------------------------
# cumulative_deviation
# ---------------------------------------------------------------------------


class TestCumulativeDeviation:
    """Tests for cumulative_deviation (GCP metric)."""

    def test_cumulative_deviation_centered(self, iid_uniform: np.ndarray) -> None:
        """Uniform(0,1) data has small z (not significant)."""
        result = cumulative_deviation(iid_uniform)
        assert result["p_value"] > 0.01
        assert abs(result["final_z"]) < 4.0

    def test_biased_detected(self, biased_uniform: np.ndarray) -> None:
        """Biased (mean ~0.55) data shows significant deviation."""
        result = cumulative_deviation(biased_uniform)
        assert result["p_value"] < 0.05
        assert result["final_z"] > 0  # shifted above 0.5

    def test_deviations_length(self, iid_uniform: np.ndarray) -> None:
        """Deviations array has same length as input."""
        result = cumulative_deviation(iid_uniform)
        assert len(result["deviations"]) == len(iid_uniform)

    def test_returns_expected_keys(self, iid_uniform: np.ndarray) -> None:
        """Result dict has all documented keys."""
        result = cumulative_deviation(iid_uniform)
        assert set(result.keys()) == {"deviations", "final_z", "p_value"}


# ---------------------------------------------------------------------------
# chi_square_rank_test
# ---------------------------------------------------------------------------


class TestChiSquareRankTest:
    """Tests for chi_square_rank_test."""

    def test_chi_square_uniform_ranks(self) -> None:
        """Uniform rank distribution passes chi-square test (p > 0.01)."""
        rng = np.random.default_rng(42)
        n_bins = 5
        ranks = rng.integers(0, n_bins, size=1000)
        probs = np.full(n_bins, 1.0 / n_bins)

        result = chi_square_rank_test(ranks, probs)
        assert result["p_value"] > 0.01

    def test_biased_ranks_detected(self) -> None:
        """Ranks concentrated in bin 0 are detected."""
        ranks = np.zeros(1000, dtype=int)
        probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

        result = chi_square_rank_test(ranks, probs)
        assert result["p_value"] < 0.01

    def test_returns_expected_keys(self) -> None:
        """Result dict has chi2, p_value, dof."""
        ranks = np.array([0, 1, 2, 0, 1, 2] * 50)
        probs = np.array([1 / 3, 1 / 3, 1 / 3])
        result = chi_square_rank_test(ranks, probs)
        assert set(result.keys()) == {"chi2", "p_value", "dof"}


# ---------------------------------------------------------------------------
# entropy_rate
# ---------------------------------------------------------------------------


class TestEntropyRate:
    """Tests for entropy_rate."""

    def test_entropy_rate_random_bytes(self) -> None:
        """Random bytes have high bits_per_byte (> 7.0)."""
        rng = np.random.default_rng(42)
        data = bytes(rng.integers(0, 256, size=10000, dtype=np.uint8))
        result = entropy_rate(data)
        assert result["bits_per_byte"] > 7.0

    def test_all_zeros_low_entropy(self) -> None:
        """All-zero bytes have very low bits_per_byte."""
        data = b"\x00" * 10000
        result = entropy_rate(data)
        assert result["bits_per_byte"] < 1.0
        assert result["ratio"] < 0.15

    def test_entropy_rate_empty(self) -> None:
        """Empty bytes returns zeros."""
        result = entropy_rate(b"")
        assert result["bits_per_byte"] == 0.0
        assert result["ratio"] == 0.0

    def test_returns_expected_keys(self) -> None:
        """Result dict has bits_per_byte and ratio."""
        result = entropy_rate(b"\xab" * 100)
        assert set(result.keys()) == {"bits_per_byte", "ratio"}


# ---------------------------------------------------------------------------
# bayesian_sequential
# ---------------------------------------------------------------------------


class TestBayesianSequential:
    """Tests for bayesian_sequential."""

    def test_bayesian_sequential_null(self, iid_uniform: np.ndarray) -> None:
        """Uniform data favors the null hypothesis (BF10 not strongly for effect)."""
        result = bayesian_sequential(iid_uniform)
        assert result["bayes_factor"] < 3.0

    def test_shifted_data_detects_effect(self) -> None:
        """Data with mean well above 0.5 shows evidence for effect."""
        rng = np.random.default_rng(42)
        shifted = rng.normal(0.6, 0.1, size=500)
        result = bayesian_sequential(shifted, prior_r=0.1)
        assert result["bayes_factor"] > 10.0
        assert "effect" in result["interpretation"]

    def test_returns_expected_keys(self, iid_uniform: np.ndarray) -> None:
        """Result dict has bayes_factor and interpretation."""
        result = bayesian_sequential(iid_uniform)
        assert set(result.keys()) == {"bayes_factor", "interpretation"}
