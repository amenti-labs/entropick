"""Tests for analysis.compare -- two-sample comparison with known distributions."""

from __future__ import annotations

import math

import numpy as np
import pytest

try:
    import scipy  # noqa: F401

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from qr_sampler.analysis.compare import (
    ComparisonResult,
    compare_sessions,
    effect_size_report,
    stouffer_z,
)

pytestmark = pytest.mark.skipif(not HAS_SCIPY, reason="scipy required")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def control_uniform() -> np.ndarray:
    """1000 Uniform(0,1) values (control condition)."""
    return np.random.default_rng(42).uniform(0.0, 1.0, size=1000)


@pytest.fixture()
def experimental_shifted() -> np.ndarray:
    """1000 values with mean shifted to ~0.55 (experimental condition)."""
    rng = np.random.default_rng(99)
    return np.clip(rng.uniform(0.05, 1.05, size=1000), 0.0, 1.0)


@pytest.fixture()
def another_uniform() -> np.ndarray:
    """1000 Uniform(0,1) values with different seed."""
    return np.random.default_rng(123).uniform(0.0, 1.0, size=1000)


# ---------------------------------------------------------------------------
# compare_sessions
# ---------------------------------------------------------------------------


class TestCompareSessions:
    """Tests for compare_sessions."""

    def test_compare_identical_distributions(
        self,
        control_uniform: np.ndarray,
        another_uniform: np.ndarray,
    ) -> None:
        """Same distribution has p > 0.05 for all tests."""
        result = compare_sessions(control_uniform, another_uniform)
        assert result.mann_whitney_p > 0.05
        assert result.ks_p_value > 0.05
        assert result.welch_p > 0.05
        assert abs(result.cohens_d) < 0.2

    def test_compare_shifted_distributions(
        self,
        control_uniform: np.ndarray,
        experimental_shifted: np.ndarray,
    ) -> None:
        """Shifted mean detected (low p, positive Cohen's d)."""
        result = compare_sessions(control_uniform, experimental_shifted)
        significant = (
            result.mann_whitney_p < 0.05 or result.ks_p_value < 0.05 or result.welch_p < 0.05
        )
        assert significant
        assert result.cohens_d > 0
        assert result.experimental_mean > result.control_mean

    def test_cohens_d_sign(
        self,
        control_uniform: np.ndarray,
        experimental_shifted: np.ndarray,
    ) -> None:
        """experimental > control gives positive Cohen's d."""
        result = compare_sessions(control_uniform, experimental_shifted)
        assert result.cohens_d > 0

    def test_sample_sizes_correct(
        self,
        control_uniform: np.ndarray,
        experimental_shifted: np.ndarray,
    ) -> None:
        """n_control and n_experimental match input lengths."""
        result = compare_sessions(control_uniform, experimental_shifted)
        assert result.n_control == len(control_uniform)
        assert result.n_experimental == len(experimental_shifted)

    def test_cohens_d_ci_contains_estimate(
        self,
        control_uniform: np.ndarray,
        another_uniform: np.ndarray,
    ) -> None:
        """The 95% CI should contain the point estimate."""
        result = compare_sessions(control_uniform, another_uniform)
        assert result.cohens_d_ci_low <= result.cohens_d <= result.cohens_d_ci_high

    def test_all_fields_finite(
        self,
        control_uniform: np.ndarray,
        another_uniform: np.ndarray,
    ) -> None:
        """All ComparisonResult fields are finite numbers."""
        result = compare_sessions(control_uniform, another_uniform)
        assert math.isfinite(result.mann_whitney_u)
        assert math.isfinite(result.mann_whitney_p)
        assert math.isfinite(result.ks_statistic)
        assert math.isfinite(result.ks_p_value)
        assert math.isfinite(result.welch_t)
        assert math.isfinite(result.welch_p)
        assert math.isfinite(result.cohens_d)


# ---------------------------------------------------------------------------
# stouffer_z
# ---------------------------------------------------------------------------


class TestStoufferZ:
    """Tests for stouffer_z."""

    def test_stouffer_z_null(self) -> None:
        """Small z-scores combine to small combined z (not significant)."""
        result = stouffer_z([0.1, -0.2, 0.15])
        assert abs(result["combined_z"]) < 1.0
        assert result["p_value"] > 0.3

    def test_stouffer_z_significant(self) -> None:
        """Multiple significant z-scores combine to significant result."""
        result = stouffer_z([2.0, 2.0, 2.0])
        assert result["combined_z"] == pytest.approx(6.0 / math.sqrt(3.0))
        assert result["p_value"] < 0.01

    def test_stouffer_z_empty(self) -> None:
        """Empty list returns z=0, p=1."""
        result = stouffer_z([])
        assert result["combined_z"] == 0.0
        assert result["p_value"] == 1.0

    def test_cancelling_z_scores(self) -> None:
        """Opposite z-scores cancel out."""
        result = stouffer_z([3.0, -3.0])
        assert result["combined_z"] == pytest.approx(0.0)
        assert result["p_value"] > 0.9

    def test_returns_expected_keys(self) -> None:
        """Result dict has combined_z and p_value."""
        result = stouffer_z([1.0])
        assert set(result.keys()) == {"combined_z", "p_value"}


# ---------------------------------------------------------------------------
# effect_size_report
# ---------------------------------------------------------------------------


class TestEffectSizeReport:
    """Tests for effect_size_report."""

    def test_effect_size_report_centered(self, control_uniform: np.ndarray) -> None:
        """Uniform data has Cohen's d near 0."""
        result = effect_size_report(control_uniform)
        assert abs(result["cohens_d"]) < 0.2
        assert result["p_value"] > 0.01

    def test_shifted_data_positive_d(self, experimental_shifted: np.ndarray) -> None:
        """Mean-shifted data has positive Cohen's d."""
        result = effect_size_report(experimental_shifted)
        assert result["cohens_d"] > 0

    def test_ci_contains_estimate(self, control_uniform: np.ndarray) -> None:
        """95% CI contains the point estimate."""
        result = effect_size_report(control_uniform)
        assert result["ci_low"] <= result["cohens_d"] <= result["ci_high"]

    def test_returns_expected_keys(self, control_uniform: np.ndarray) -> None:
        """Result dict has all documented keys."""
        result = effect_size_report(control_uniform)
        assert set(result.keys()) == {"cohens_d", "ci_low", "ci_high", "p_value"}


# ---------------------------------------------------------------------------
# ComparisonResult
# ---------------------------------------------------------------------------


class TestComparisonResult:
    """Tests for the ComparisonResult frozen dataclass."""

    def test_comparison_result_frozen(self) -> None:
        """ComparisonResult is immutable."""
        result = ComparisonResult(
            mann_whitney_u=1.0,
            mann_whitney_p=0.5,
            ks_statistic=0.1,
            ks_p_value=0.9,
            welch_t=0.5,
            welch_p=0.6,
            cohens_d=0.1,
            cohens_d_ci_low=-0.1,
            cohens_d_ci_high=0.3,
            control_mean=0.5,
            experimental_mean=0.51,
            n_control=100,
            n_experimental=100,
        )
        with pytest.raises(AttributeError):
            result.cohens_d = 999.0  # type: ignore[misc]
