"""Shared utilities for pipeline stages."""

from __future__ import annotations

from typing import Any

import numpy as np

# Type alias for floating-point ndarrays.
FloatArray = np.ndarray[Any, np.dtype[np.floating[Any]]]


def stable_softmax(logits: FloatArray) -> FloatArray | None:
    """Numerically stable softmax via shift-by-max.

    Args:
        logits: 1-D logit array (may contain -inf for masked tokens).

    Returns:
        Probability array of the same shape summing to ~1.0,
        or ``None`` if no finite logits exist or sum is zero.
    """
    finite_mask = np.isfinite(logits)
    if not np.any(finite_mask):
        return None

    max_logit = float(np.max(logits[finite_mask]))
    shifted = logits - max_logit
    exp_shifted = np.exp(shifted)
    total = np.sum(exp_shifted)

    if total == 0.0:
        return None

    result: FloatArray = exp_shifted / total
    return result


def shannon_entropy_from_probs(probs: FloatArray) -> float:
    """Compute Shannon entropy H = -sum(p_i * ln(p_i)) from a probability array.

    Args:
        probs: Probability array (must sum to ~1.0).

    Returns:
        Shannon entropy in nats, guaranteed >= 0.
    """
    mask = probs > 0
    h = -float(np.sum(probs[mask] * np.log(probs[mask])))
    return max(0.0, h)
