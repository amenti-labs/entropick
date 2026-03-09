"""Statistical test battery for entropick entropy and consciousness research.

All public functions accept numpy arrays and return plain dicts with test
results.  ``scipy`` is imported lazily -- it is only required at call time
(it is a dev dependency).
"""

from __future__ import annotations

import math
import zlib
from typing import Any

import numpy as np

from qr_sampler.analysis._utils import _require_scipy_stats

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def autocorrelation_test(
    u_values: np.ndarray[Any, np.dtype[np.floating[Any]]],
    max_lag: int = 10,
) -> dict[str, Any]:
    """Ljung-Box test for autocorrelation in u-values.

    Args:
        u_values: Array of uniform values in (0, 1).
        max_lag: Maximum lag to test.

    Returns:
        Dict with ``statistic``, ``p_value``, ``lags``, and
        ``autocorrelations`` keys.
    """
    n = len(u_values)
    mean = np.mean(u_values)
    var = np.var(u_values, ddof=0)

    if var == 0.0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "lags": list(range(1, max_lag + 1)),
            "autocorrelations": [0.0] * max_lag,
        }

    autocorrs: list[float] = []
    for lag in range(1, max_lag + 1):
        c = np.sum((u_values[: n - lag] - mean) * (u_values[lag:] - mean)) / (n * var)
        autocorrs.append(float(c))

    # Ljung-Box Q statistic
    q = float(n * (n + 2) * sum(r**2 / (n - k) for k, r in enumerate(autocorrs, start=1)))

    stats = _require_scipy_stats()
    p_value = float(1.0 - stats.chi2.cdf(q, df=max_lag))

    return {
        "statistic": q,
        "p_value": p_value,
        "lags": list(range(1, max_lag + 1)),
        "autocorrelations": autocorrs,
    }


def runs_test(
    u_values: np.ndarray[Any, np.dtype[np.floating[Any]]],
) -> dict[str, Any]:
    """Wald-Wolfowitz runs test for randomness.

    Values are binarised about the median.

    Args:
        u_values: Array of uniform values.

    Returns:
        Dict with ``n_runs``, ``expected_runs``, ``z_score``, ``p_value``.
    """
    stats = _require_scipy_stats()

    median = float(np.median(u_values))
    binary = (u_values >= median).astype(int)

    n1 = int(np.sum(binary == 1))
    n0 = int(np.sum(binary == 0))
    n = n0 + n1

    if n0 == 0 or n1 == 0:
        return {
            "n_runs": 1,
            "expected_runs": 1.0,
            "z_score": 0.0,
            "p_value": 1.0,
        }

    # Count runs
    runs = 1 + int(np.sum(binary[1:] != binary[:-1]))

    expected = 1.0 + (2.0 * n0 * n1) / n
    var_runs = (2.0 * n0 * n1 * (2.0 * n0 * n1 - n)) / (n**2 * (n - 1.0))

    if var_runs <= 0.0:
        return {
            "n_runs": runs,
            "expected_runs": expected,
            "z_score": 0.0,
            "p_value": 1.0,
        }

    z = (runs - expected) / math.sqrt(var_runs)
    p = float(2.0 * stats.norm.sf(abs(z)))

    return {
        "n_runs": runs,
        "expected_runs": expected,
        "z_score": float(z),
        "p_value": p,
    }


def serial_correlation(
    values: np.ndarray[Any, np.dtype[np.floating[Any]]],
    lag: int = 1,
) -> dict[str, Any]:
    """Lag-k Pearson correlation.

    Args:
        values: Numeric array.
        lag: Correlation lag (default 1).

    Returns:
        Dict with ``correlation`` and ``p_value``.
    """
    stats = _require_scipy_stats()

    if len(values) <= lag:
        return {"correlation": 0.0, "p_value": 1.0}

    r, p = stats.pearsonr(values[:-lag], values[lag:])
    return {"correlation": float(r), "p_value": float(p)}


def hurst_exponent(
    values: np.ndarray[Any, np.dtype[np.floating[Any]]],
) -> dict[str, Any]:
    """Rescaled-range (R/S) analysis for the Hurst exponent.

    Args:
        values: Numeric time series.

    Returns:
        Dict with ``hurst`` and ``interpretation``.
    """
    n = len(values)
    if n < 20:
        return {"hurst": 0.5, "interpretation": "insufficient_data"}

    # Use a range of sub-series sizes
    min_size = 10
    sizes: list[int] = []
    rs_values: list[float] = []

    size = min_size
    while size <= n // 2:
        sizes.append(size)
        n_chunks = n // size
        rs_chunk: list[float] = []

        for i in range(n_chunks):
            chunk = values[i * size : (i + 1) * size]
            mean_c = np.mean(chunk)
            deviations = np.cumsum(chunk - mean_c)
            r = float(np.max(deviations) - np.min(deviations))
            s = float(np.std(chunk, ddof=1))
            if s > 0:
                rs_chunk.append(r / s)

        if rs_chunk:
            rs_values.append(float(np.mean(rs_chunk)))

        size = int(size * 1.5)
        if size == sizes[-1]:
            size += 1

    if len(rs_values) < 2:
        return {"hurst": 0.5, "interpretation": "insufficient_data"}

    log_sizes = np.log(np.array(sizes[: len(rs_values)], dtype=float))
    log_rs = np.log(np.array(rs_values, dtype=float))

    # Linear regression for slope
    coeffs = np.polyfit(log_sizes, log_rs, 1)
    h = float(coeffs[0])

    if h > 0.55:
        interpretation = "persistent"
    elif h < 0.45:
        interpretation = "anti-persistent"
    else:
        interpretation = "random"

    return {"hurst": h, "interpretation": interpretation}


def approximate_entropy(
    values: np.ndarray[Any, np.dtype[np.floating[Any]]],
    m: int = 2,
    r: float = 0.2,
) -> dict[str, Any]:
    """Approximate entropy (ApEn) complexity measure.

    Args:
        values: Numeric time series.
        m: Embedding dimension.
        r: Tolerance as fraction of std.

    Returns:
        Dict with ``apen`` and ``interpretation``.
    """
    n = len(values)
    if n < m + 1:
        return {"apen": 0.0, "interpretation": "insufficient_data"}

    tolerance = r * float(np.std(values, ddof=0))
    if tolerance == 0.0:
        return {"apen": 0.0, "interpretation": "constant_series"}

    def _phi(dim: int) -> float:
        templates = np.array([values[i : i + dim] for i in range(n - dim + 1)])
        counts = np.zeros(len(templates))
        for i, t in enumerate(templates):
            dists = np.max(np.abs(templates - t), axis=1)
            counts[i] = np.sum(dists <= tolerance)
        counts /= len(templates)
        return float(np.mean(np.log(counts)))

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)
    apen = phi_m - phi_m1

    if apen > 1.0:
        interpretation = "high_complexity"
    elif apen > 0.5:
        interpretation = "moderate_complexity"
    else:
        interpretation = "low_complexity"

    return {"apen": apen, "interpretation": interpretation}


def cumulative_deviation(
    u_values: np.ndarray[Any, np.dtype[np.floating[Any]]],
) -> dict[str, Any]:
    """Running sum of (u - 0.5) with z-test.

    This is the Global Consciousness Project's primary metric: the
    cumulative deviation of uniform values from 0.5.

    Args:
        u_values: Array of uniform values in (0, 1).

    Returns:
        Dict with ``deviations`` (list), ``final_z``, and ``p_value``.
    """
    stats = _require_scipy_stats()

    n = len(u_values)
    if n == 0:
        return {"deviations": [], "final_z": 0.0, "p_value": 1.0}

    deviations = np.cumsum(u_values - 0.5)

    # Under H0 (uniform), Var(u-0.5) = 1/12, so Var(sum) = n/12
    final_sum = float(deviations[-1])
    z = final_sum / math.sqrt(n / 12.0)
    p = float(2.0 * stats.norm.sf(abs(z)))

    return {
        "deviations": deviations.tolist(),
        "final_z": float(z),
        "p_value": p,
    }


def chi_square_rank_test(
    ranks: np.ndarray[Any, np.dtype[np.integer[Any]]],
    probabilities: np.ndarray[Any, np.dtype[np.floating[Any]]],
) -> dict[str, Any]:
    """Chi-square goodness-of-fit for token rank distribution.

    Compares observed rank frequencies against expected frequencies
    derived from *probabilities*.

    Args:
        ranks: Array of observed token ranks (0-indexed).
        probabilities: Expected probability for each rank bucket.

    Returns:
        Dict with ``chi2``, ``p_value``, and ``dof``.
    """
    stats = _require_scipy_stats()

    n = len(ranks)
    n_bins = len(probabilities)
    expected = probabilities * n

    # Observed counts
    observed = np.zeros(n_bins, dtype=float)
    for rank in ranks:
        if 0 <= rank < n_bins:
            observed[rank] += 1.0

    # Merge bins with expected < 5 into the last bin
    mask = expected >= 5.0
    if not np.all(mask):
        # Pool small bins into an "other" category
        observed_pooled = np.append(observed[mask], np.sum(observed[~mask]))
        expected_pooled = np.append(expected[mask], np.sum(expected[~mask]))
    else:
        observed_pooled = observed
        expected_pooled = expected

    # Filter out zero-expected bins
    nonzero = expected_pooled > 0
    observed_pooled = observed_pooled[nonzero]
    expected_pooled = expected_pooled[nonzero]

    dof = max(len(observed_pooled) - 1, 1)
    chi2_stat, p_val = stats.chisquare(observed_pooled, f_exp=expected_pooled)

    return {
        "chi2": float(chi2_stat),
        "p_value": float(p_val),
        "dof": dof,
    }


def entropy_rate(raw_bytes: bytes) -> dict[str, float]:
    """Compression-based entropy rate estimation.

    Uses zlib compression ratio as a proxy for Shannon entropy rate.

    Args:
        raw_bytes: Raw entropy bytes to analyse.

    Returns:
        Dict with ``bits_per_byte`` and ``ratio``.
    """
    if not raw_bytes:
        return {"bits_per_byte": 0.0, "ratio": 0.0}

    compressed = zlib.compress(raw_bytes, level=9)
    ratio = len(compressed) / len(raw_bytes)

    # Estimate: truly random data is incompressible (ratio >= 1.0),
    # low entropy data compresses well (ratio << 1.0).
    # Map ratio to bits_per_byte: clamp at 8.0 (maximum for bytes).
    bits_per_byte = min(ratio * 8.0, 8.0)

    return {
        "bits_per_byte": bits_per_byte,
        "ratio": ratio,
    }


def bayesian_sequential(
    u_values: np.ndarray[Any, np.dtype[np.floating[Any]]],
    prior_r: float = 0.1,
) -> dict[str, Any]:
    """Bayesian evidence for mean shift from 0.5.

    Uses a Savage-Dickey density ratio approach: compares the likelihood
    of the data under H0 (mean = 0.5) vs H1 (mean ~ Normal(0.5, prior_r)).

    Args:
        u_values: Array of uniform values.
        prior_r: Prior scale for the effect size (std of prior on mean).

    Returns:
        Dict with ``bayes_factor`` and ``interpretation``.
    """
    stats = _require_scipy_stats()

    n = len(u_values)
    observed_mean = float(np.mean(u_values))
    observed_var = float(np.var(u_values, ddof=1)) if n > 1 else 1.0 / 12.0
    # Prior: N(0.5, prior_r^2)
    # Posterior: combine prior with likelihood
    prior_var = prior_r**2
    likelihood_var = observed_var / n

    posterior_var = 1.0 / (1.0 / prior_var + 1.0 / likelihood_var)
    posterior_mean = posterior_var * (0.5 / prior_var + observed_mean / likelihood_var)

    # Savage-Dickey: BF10 = prior(theta=0.5) / posterior(theta=0.5)
    prior_at_null = float(stats.norm.pdf(0.5, loc=0.5, scale=prior_r))
    posterior_at_null = float(
        stats.norm.pdf(0.5, loc=posterior_mean, scale=math.sqrt(posterior_var))
    )

    bf10 = prior_at_null / posterior_at_null if posterior_at_null > 0 else float("inf")

    if bf10 > 10.0:
        interpretation = "strong_evidence_for_effect"
    elif bf10 > 3.0:
        interpretation = "moderate_evidence_for_effect"
    elif bf10 > 1.0:
        interpretation = "anecdotal_evidence_for_effect"
    elif bf10 > 1.0 / 3.0:
        interpretation = "anecdotal_evidence_for_null"
    elif bf10 > 0.1:
        interpretation = "moderate_evidence_for_null"
    else:
        interpretation = "strong_evidence_for_null"

    return {
        "bayes_factor": bf10,
        "interpretation": interpretation,
    }
