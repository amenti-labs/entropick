"""Shared utilities for analysis modules."""

from __future__ import annotations

from typing import Any


def _require_scipy_stats() -> Any:
    """Import and return ``scipy.stats``, raising if unavailable."""
    try:
        from scipy import stats  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "scipy is required for statistical tests. Install it with: pip install scipy"
        ) from exc
    return stats
