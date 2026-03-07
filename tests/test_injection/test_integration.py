"""Integration tests for injection methods through QRSamplerLogitsProcessor.

Tests the full pipeline with injection methods (M1: logit noise, M2: temperature
variance, M3: correlated walk) enabled via environment variables and per-request
extra_args. Verifies backward compatibility, combined operation, and state
persistence.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.helpers import (
    SAMPLE_LOGITS,
    assert_onehot,
    make_processor,
    register_request,
)

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
        proc = make_processor()

        # Verify injection is disabled in default config.
        cfg = proc.default_config
        assert cfg.logit_noise_alpha == 0.0
        assert cfg.temp_variance_beta == 0.0
        assert cfg.walk_step == 0.0

        logits = np.array([SAMPLE_LOGITS])
        result = proc.apply(logits)

        # Standard one-hot output: one 0.0, rest -inf.
        assert result is logits  # In-place modification.
        assert_onehot(result[0])

    def test_all_methods_combined(self) -> None:
        """All 3 injection methods active simultaneously produce valid output.

        Enables M1 (logit noise), M2 (temp variance), and M3 (correlated walk)
        together. The pipeline must not crash and must still produce a valid
        one-hot logit vector.
        """
        proc = make_processor(
            logit_noise_alpha=0.05,
            temp_variance_beta=0.2,
            walk_step=0.1,
        )

        # M3 requires per-request state, so register a request.
        register_request(proc, req_index=0)

        logits = np.array([SAMPLE_LOGITS])
        result = proc.apply(logits)

        # Must produce valid one-hot output despite all injections active.
        assert_onehot(result[0])

    def test_m1_solo(self) -> None:
        """M1 logit noise alone produces valid one-hot output.

        Only logit_noise_alpha is non-zero; M2 and M3 remain disabled.
        """
        proc = make_processor(logit_noise_alpha=0.05)

        # M1 does not require per-request state.
        logits = np.array([SAMPLE_LOGITS])
        result = proc.apply(logits)

        assert_onehot(result[0])

    def test_m3_walk_state_persists(self) -> None:
        """M3 correlated walk updates walk_position across apply() calls.

        The walk position must change from its initial value after the first
        apply(), and change again after the second. This proves state
        persistence across engine steps.
        """
        proc = make_processor(walk_step=0.1)

        # Register request with walk enabled (uses default config from env).
        register_request(proc, req_index=0)

        # Initial walk position is 0.5 (config default).
        initial_pos = proc._request_states[0].walk_position
        assert initial_pos == pytest.approx(0.5)

        # First apply.
        logits1 = np.array([SAMPLE_LOGITS])
        proc.apply(logits1)
        pos_after_1 = proc._request_states[0].walk_position

        # Walk position must have changed from initial.
        assert pos_after_1 != pytest.approx(0.5), (
            f"Walk position unchanged after first apply: {pos_after_1}"
        )

        # Second apply (need fresh logits; previous were modified in-place).
        logits2 = np.array([SAMPLE_LOGITS])
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
        proc = make_processor()

        # Default config has injection disabled.
        assert proc.default_config.walk_step == 0.0

        # Enable walk_step for this request only via extra_args.
        register_request(
            proc,
            req_index=0,
            extra_args={"qr_walk_step": 0.1},
        )

        # Per-request config should have walk_step enabled.
        assert proc._request_states[0].config.walk_step == pytest.approx(0.1)

        logits = np.array([SAMPLE_LOGITS])
        result = proc.apply(logits)

        # Must produce valid one-hot output.
        assert_onehot(result[0])

        # Walk position should have changed from initial 0.5.
        assert proc._request_states[0].walk_position != pytest.approx(0.5)

    def test_m3_record_marks_amplifier_stats_unknown(self) -> None:
        """When M3 is active, amplifier z-score diagnostics are marked as unknown."""
        proc = make_processor(
            walk_step=0.1,
            diagnostic_mode=True,
            log_level="none",
        )
        register_request(proc, req_index=0)

        logits = np.array([SAMPLE_LOGITS])
        proc.apply(logits)

        records = proc.sampling_logger.get_diagnostic_data()
        assert len(records) == 1
        assert math.isnan(records[0].sample_mean)
        assert math.isnan(records[0].z_score)
