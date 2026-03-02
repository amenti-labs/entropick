"""Integration tests for injection methods through QRSamplerLogitsProcessor.

Tests the full pipeline with injection methods (M1: logit noise, M2: temperature
variance, M3: correlated walk) enabled via environment variables and per-request
extra_args. Verifies backward compatibility, combined operation, and state
persistence.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from qr_sampler.processor import QRSamplerLogitsProcessor

# ---------------------------------------------------------------------------
# Mock objects (same pattern as tests/test_processor.py)
# ---------------------------------------------------------------------------


@dataclass
class _MockVllmConfig:
    """Simulates vLLM's VllmConfig with vocab_size access."""

    vocab_size: int = 10


@dataclass
class _MockSamplingParams:
    """Simulates vLLM's SamplingParams."""

    extra_args: dict[str, Any] | None = None


@dataclass
class _MockAddedRequest:
    """Simulates a BatchUpdate added request."""

    req_index: int
    sampling_params: _MockSamplingParams | None = None


@dataclass
class _MockBatchUpdate:
    """Simulates vLLM's BatchUpdate dataclass."""

    removed: list[int] | None = None
    moved: list[Any] | None = None
    added: list[_MockAddedRequest] | None = None

    def __post_init__(self) -> None:
        if self.removed is None:
            self.removed = []
        if self.moved is None:
            self.moved = []
        if self.added is None:
            self.added = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VOCAB_SIZE = 10
_SAMPLE_LOGITS = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]


def _make_processor(**config_overrides: Any) -> QRSamplerLogitsProcessor:
    """Create a processor with mock entropy and optional config overrides.

    Sets environment variables to configure, then instantiates. Cleans up
    environment after construction.
    """
    env_vars = {
        "QR_ENTROPY_SOURCE_TYPE": "mock_uniform",
        "QR_FALLBACK_MODE": "error",
        "QR_LOG_LEVEL": "none",
    }
    for key, value in config_overrides.items():
        env_vars[f"QR_{key.upper()}"] = str(value)

    old_env: dict[str, str | None] = {}
    for key, value in env_vars.items():
        old_env[key] = os.environ.get(key)
        os.environ[key] = value

    try:
        proc = QRSamplerLogitsProcessor(
            vllm_config=_MockVllmConfig(vocab_size=_VOCAB_SIZE),
            device=None,
            is_pin_memory=False,
        )
    finally:
        for key, original in old_env.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original

    return proc


def _assert_onehot(row: np.ndarray) -> None:
    """Assert a logit row is one-hot: exactly one 0.0, rest -inf."""
    n_zero = int(np.sum(row == 0.0))
    n_neginf = int(np.sum(np.isneginf(row)))
    assert n_zero == 1, f"Expected 1 zero, got {n_zero}"
    assert n_neginf == len(row) - 1, f"Expected {len(row) - 1} -inf, got {n_neginf}"


def _register_request(
    proc: QRSamplerLogitsProcessor,
    req_index: int = 0,
    extra_args: dict[str, Any] | None = None,
) -> None:
    """Register a single request with optional extra_args."""
    params = _MockSamplingParams(extra_args=extra_args)
    batch = _MockBatchUpdate(added=[_MockAddedRequest(req_index=req_index, sampling_params=params)])
    proc.update_state(batch)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInjectionIntegration:
    """Integration tests for injection methods through the full pipeline."""

    def test_backward_compat_all_disabled(self) -> None:
        """Default config (all injection disabled) produces standard one-hot output.

        This is the critical backward-compatibility test: when no injection
        fields are set, the processor behaves identically to its pre-injection
        behavior.
        """
        proc = _make_processor()

        # Verify injection is disabled in default config.
        cfg = proc.default_config
        assert cfg.logit_noise_alpha == 0.0
        assert cfg.temp_variance_beta == 0.0
        assert cfg.walk_step == 0.0

        logits = np.array([_SAMPLE_LOGITS])
        result = proc.apply(logits)

        # Standard one-hot output: one 0.0, rest -inf.
        assert result is logits  # In-place modification.
        _assert_onehot(result[0])

    def test_all_methods_combined(self) -> None:
        """All 3 injection methods active simultaneously produce valid output.

        Enables M1 (logit noise), M2 (temp variance), and M3 (correlated walk)
        together. The pipeline must not crash and must still produce a valid
        one-hot logit vector.
        """
        proc = _make_processor(
            logit_noise_alpha=0.05,
            temp_variance_beta=0.2,
            walk_step=0.1,
        )

        # M3 requires per-request state, so register a request.
        _register_request(proc, req_index=0)

        logits = np.array([_SAMPLE_LOGITS])
        result = proc.apply(logits)

        # Must produce valid one-hot output despite all injections active.
        _assert_onehot(result[0])

    def test_m1_solo(self) -> None:
        """M1 logit noise alone produces valid one-hot output.

        Only logit_noise_alpha is non-zero; M2 and M3 remain disabled.
        """
        proc = _make_processor(logit_noise_alpha=0.05)

        # M1 does not require per-request state.
        logits = np.array([_SAMPLE_LOGITS])
        result = proc.apply(logits)

        _assert_onehot(result[0])

    def test_m3_walk_state_persists(self) -> None:
        """M3 correlated walk updates walk_position across apply() calls.

        The walk position must change from its initial value after the first
        apply(), and change again after the second. This proves state
        persistence across engine steps.
        """
        proc = _make_processor(walk_step=0.1)

        # Register request with walk enabled (uses default config from env).
        _register_request(proc, req_index=0)

        # Initial walk position is 0.5 (config default).
        initial_pos = proc._request_states[0].walk_position
        assert initial_pos == pytest.approx(0.5)

        # First apply.
        logits1 = np.array([_SAMPLE_LOGITS])
        proc.apply(logits1)
        pos_after_1 = proc._request_states[0].walk_position

        # Walk position must have changed from initial.
        assert pos_after_1 != pytest.approx(0.5), (
            f"Walk position unchanged after first apply: {pos_after_1}"
        )

        # Second apply (need fresh logits; previous were modified in-place).
        logits2 = np.array([_SAMPLE_LOGITS])
        proc.apply(logits2)
        pos_after_2 = proc._request_states[0].walk_position

        # Walk position must change again.
        assert pos_after_2 != pytest.approx(pos_after_1), (
            f"Walk position unchanged after second apply: {pos_after_2}"
        )

    def test_per_request_override(self) -> None:
        """Per-request extra_args enables injection on a default-disabled processor.

        Creates a processor with all injection disabled, then enables walk_step
        for a single request via extra_args. Verifies the per-request config
        is applied and the walk state updates.
        """
        proc = _make_processor()

        # Default config has injection disabled.
        assert proc.default_config.walk_step == 0.0

        # Enable walk_step for this request only via extra_args.
        _register_request(
            proc,
            req_index=0,
            extra_args={"qr_walk_step": 0.1},
        )

        # Per-request config should have walk_step enabled.
        assert proc._request_states[0].config.walk_step == pytest.approx(0.1)

        logits = np.array([_SAMPLE_LOGITS])
        result = proc.apply(logits)

        # Must produce valid one-hot output.
        _assert_onehot(result[0])

        # Walk position should have changed from initial 0.5.
        assert proc._request_states[0].walk_position != pytest.approx(0.5)
