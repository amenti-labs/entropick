"""Analysis module for qr-sampler experimental data.

Provides persistence (JSONL save/load), statistical test batteries for
entropy quality and consciousness-influence research, and two-sample
comparison tools for control vs. experimental sessions.
"""

from qr_sampler.analysis.compare import (
    ComparisonResult,
    compare_sessions,
    effect_size_report,
    stouffer_z,
)
from qr_sampler.analysis.persistence import load_records, save_records
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

__all__ = [
    "ComparisonResult",
    "approximate_entropy",
    "autocorrelation_test",
    "bayesian_sequential",
    "chi_square_rank_test",
    "compare_sessions",
    "cumulative_deviation",
    "effect_size_report",
    "entropy_rate",
    "hurst_exponent",
    "load_records",
    "runs_test",
    "save_records",
    "serial_correlation",
    "stouffer_z",
]
