"""Two-sample comparison tools for qr-sampler experimental sessions.

Provides non-parametric and parametric comparisons of u-value
distributions between control and experimental conditions, plus
meta-analytic combination of z-scores across sessions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


def _require_scipy_stats() -> Any:
    """Import and return ``scipy.stats``, raising if unavailable."""
    try:
        from scipy import stats  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "scipy is required for comparison tests. Install it with: pip install scipy"
        ) from exc
    return stats


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """Result of a two-sample comparison between control and experimental u-values.

    Attributes:
        mann_whitney_u: Mann-Whitney U statistic.
        mann_whitney_p: Two-sided p-value for the Mann-Whitney test.
        ks_statistic: Kolmogorov-Smirnov statistic.
        ks_p_value: Two-sided p-value for the KS test.
        welch_t: Welch's t-statistic.
        welch_p: Two-sided p-value for Welch's t-test.
        cohens_d: Cohen's d effect size (positive = experimental > control).
        cohens_d_ci_low: Lower bound of 95% CI for Cohen's d.
        cohens_d_ci_high: Upper bound of 95% CI for Cohen's d.
        control_mean: Mean of control u-values.
        experimental_mean: Mean of experimental u-values.
        n_control: Number of control observations.
        n_experimental: Number of experimental observations.
    """

    mann_whitney_u: float
    mann_whitney_p: float
    ks_statistic: float
    ks_p_value: float
    welch_t: float
    welch_p: float
    cohens_d: float
    cohens_d_ci_low: float
    cohens_d_ci_high: float
    control_mean: float
    experimental_mean: float
    n_control: int
    n_experimental: int


def _cohens_d_with_ci(
    group1: np.ndarray[Any, np.dtype[np.floating[Any]]],
    group2: np.ndarray[Any, np.dtype[np.floating[Any]]],
) -> tuple[float, float, float]:
    """Compute Cohen's d with 95% confidence interval.

    Uses the pooled standard deviation and Hedges' correction factor
    for the CI width.

    Args:
        group1: Control group values.
        group2: Experimental group values.

    Returns:
        Tuple of (d, ci_low, ci_high).
    """
    n1 = len(group1)
    n2 = len(group2)
    mean1 = float(np.mean(group1))
    mean2 = float(np.mean(group2))
    var1 = float(np.var(group1, ddof=1))
    var2 = float(np.var(group2, ddof=1))

    # Pooled std
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    pooled_std = math.sqrt(pooled_var) if pooled_var > 0 else 1.0

    d = (mean2 - mean1) / pooled_std

    # SE of Cohen's d (Hedges & Olkin, 1985)
    se_d = math.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2.0 * (n1 + n2)))

    ci_low = d - 1.96 * se_d
    ci_high = d + 1.96 * se_d

    return d, ci_low, ci_high


def compare_sessions(
    control_u: np.ndarray[Any, np.dtype[np.floating[Any]]],
    experimental_u: np.ndarray[Any, np.dtype[np.floating[Any]]],
) -> ComparisonResult:
    """Compare u-value distributions between control and experimental sessions.

    Runs three complementary tests:
    - Mann-Whitney U (non-parametric rank test)
    - Kolmogorov-Smirnov (distribution shape)
    - Welch's t-test (parametric mean comparison)

    Plus Cohen's d effect size with 95% CI.

    Args:
        control_u: U-values from the control (no-intention) session.
        experimental_u: U-values from the experimental (intention) session.

    Returns:
        Frozen ComparisonResult dataclass.
    """
    stats = _require_scipy_stats()

    mw_u, mw_p = stats.mannwhitneyu(control_u, experimental_u, alternative="two-sided")
    ks_stat, ks_p = stats.ks_2samp(control_u, experimental_u)
    t_stat, t_p = stats.ttest_ind(control_u, experimental_u, equal_var=False)

    d, ci_low, ci_high = _cohens_d_with_ci(control_u, experimental_u)

    return ComparisonResult(
        mann_whitney_u=float(mw_u),
        mann_whitney_p=float(mw_p),
        ks_statistic=float(ks_stat),
        ks_p_value=float(ks_p),
        welch_t=float(t_stat),
        welch_p=float(t_p),
        cohens_d=d,
        cohens_d_ci_low=ci_low,
        cohens_d_ci_high=ci_high,
        control_mean=float(np.mean(control_u)),
        experimental_mean=float(np.mean(experimental_u)),
        n_control=len(control_u),
        n_experimental=len(experimental_u),
    )


def stouffer_z(z_scores: list[float]) -> dict[str, Any]:
    """Combine z-scores across sessions using Stouffer's method.

    Args:
        z_scores: List of z-scores from individual sessions.

    Returns:
        Dict with ``combined_z`` and ``p_value``.
    """
    stats = _require_scipy_stats()

    k = len(z_scores)
    if k == 0:
        return {"combined_z": 0.0, "p_value": 1.0}

    combined = sum(z_scores) / math.sqrt(k)
    p = float(2.0 * stats.norm.sf(abs(combined)))

    return {"combined_z": combined, "p_value": p}


def effect_size_report(
    u_values: np.ndarray[Any, np.dtype[np.floating[Any]]],
) -> dict[str, Any]:
    """Single-session effect size vs the uniform null (mean = 0.5).

    Computes Cohen's d against the theoretical uniform mean and a
    one-sample t-test p-value.

    Args:
        u_values: Array of uniform values from a single session.

    Returns:
        Dict with ``cohens_d``, ``ci_low``, ``ci_high``, and ``p_value``.
    """
    stats = _require_scipy_stats()

    n = len(u_values)
    mean = float(np.mean(u_values))
    std = float(np.std(u_values, ddof=1))

    d = 0.0 if std == 0.0 else (mean - 0.5) / std

    # SE of d for one-sample
    se_d = math.sqrt(1.0 / n + d**2 / (2.0 * n))
    ci_low = d - 1.96 * se_d
    ci_high = d + 1.96 * se_d

    _, p = stats.ttest_1samp(u_values, 0.5)

    return {
        "cohens_d": d,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": float(p),
    }
