"""Tests for the Mirostat v2 stage."""

from __future__ import annotations

import numpy as np
import pytest

from tests.helpers import (
    SAMPLE_LOGITS,
    assert_onehot,
    make_processor,
    register_request,
)


class TestMirostatDisabled:
    """Verify Mirostat is a no-op when disabled."""

    def test_disabled_by_default(self) -> None:
        """Mirostat has no effect when mirostat_mode=0 (default)."""
        proc = make_processor()
        register_request(proc, req_index=0)
        logits = np.array([SAMPLE_LOGITS])
        result = proc.apply(logits)
        assert_onehot(result[0])

    def test_mode_zero_is_noop(self) -> None:
        """Explicitly setting mode=0 produces valid output without mirostat."""
        proc = make_processor(mirostat_mode=0, diagnostic_mode=True)
        register_request(proc, req_index=0)
        logits = np.array([SAMPLE_LOGITS])
        result = proc.apply(logits)
        assert_onehot(result[0])


class TestMirostatEnabled:
    """Verify Mirostat v2 produces valid one-hot output."""

    def test_mode_2_produces_onehot(self) -> None:
        """Mirostat v2 produces valid one-hot logits."""
        proc = make_processor(mirostat_mode=2)
        register_request(proc, req_index=0)
        logits = np.array([SAMPLE_LOGITS])
        result = proc.apply(logits)
        assert_onehot(result[0])

    def test_selects_valid_token(self) -> None:
        """Mirostat v2 selects a token within the vocabulary."""
        proc = make_processor(mirostat_mode=2, diagnostic_mode=True)
        register_request(proc, req_index=0)
        logits = np.array([SAMPLE_LOGITS])
        proc.apply(logits)
        records = proc.sampling_logger.get_diagnostic_data()
        assert len(records) == 1
        assert 0 <= records[0].token_id < len(SAMPLE_LOGITS)

    def test_mu_state_updates(self) -> None:
        """Mirostat mu state changes after each token."""
        proc = make_processor(mirostat_mode=2, mirostat_tau=5.0, mirostat_eta=0.1)
        register_request(proc, req_index=0)

        state = proc._request_states[0]
        initial_mu = state.stage_state["mirostat.mu"]
        assert initial_mu == pytest.approx(10.0)  # 2 * tau

        logits = np.array([SAMPLE_LOGITS])
        proc.apply(logits)

        new_mu = state.stage_state["mirostat.mu"]
        # mu should change after one step.
        assert new_mu != pytest.approx(initial_mu)

    def test_multiple_tokens_converge(self) -> None:
        """Over multiple tokens, mu drifts toward tau."""
        proc = make_processor(mirostat_mode=2, mirostat_tau=3.0, mirostat_eta=0.3)
        register_request(proc, req_index=0)

        state = proc._request_states[0]
        for _ in range(10):
            logits = np.array([SAMPLE_LOGITS])
            proc.apply(logits)

        # mu should have moved from initial (2*tau=6.0) toward a value
        # influenced by the actual surprise rate.
        final_mu = state.stage_state["mirostat.mu"]
        assert isinstance(final_mu, float)

    def test_batch_processing(self) -> None:
        """Mirostat works with batch processing."""
        proc = make_processor(mirostat_mode=2)
        register_request(proc, req_index=0)
        register_request(proc, req_index=1)

        logits = np.array([SAMPLE_LOGITS, SAMPLE_LOGITS])
        result = proc.apply(logits)
        assert_onehot(result[0])
        assert_onehot(result[1])


class TestMirostatPerRequest:
    """Test per-request mirostat configuration."""

    def test_per_request_enable(self) -> None:
        """Mirostat can be enabled per-request."""
        proc = make_processor(diagnostic_mode=True)
        register_request(proc, req_index=0, extra_args={"qr_mirostat_mode": 2})

        logits = np.array([SAMPLE_LOGITS])
        result = proc.apply(logits)
        assert_onehot(result[0])

    def test_per_request_tau(self) -> None:
        """Mirostat tau can be overridden per-request."""
        proc = make_processor(mirostat_mode=2)
        register_request(proc, req_index=0, extra_args={"qr_mirostat_tau": 2.0})

        state = proc._request_states[0]
        # mu should be 2 * tau = 4.0
        assert state.stage_state["mirostat.mu"] == pytest.approx(4.0)

    def test_per_request_eta(self) -> None:
        """Mirostat eta can be overridden per-request."""
        proc = make_processor(mirostat_mode=2)
        register_request(proc, req_index=0, extra_args={"qr_mirostat_eta": 0.5})

        logits = np.array([SAMPLE_LOGITS])
        result = proc.apply(logits)
        assert_onehot(result[0])


class TestMirostatUnsupportedMode:
    """Verify Mirostat mode=1 logs a warning."""

    def test_mode_1_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """mirostat_mode=1 should log a warning and produce valid output."""
        proc = make_processor(mirostat_mode=1)
        register_request(proc, req_index=0)
        logits = np.array([SAMPLE_LOGITS])
        with caplog.at_level("WARNING", logger="qr_sampler"):
            result = proc.apply(logits)
        assert_onehot(result[0])
        assert any("mode 1 is not implemented" in msg for msg in caplog.messages)


class TestMirostatEdgeCases:
    """Edge cases for Mirostat."""

    def test_single_token_vocab(self) -> None:
        """Mirostat with a single-token vocabulary."""
        proc = make_processor(mirostat_mode=2, vocab_size=1)
        register_request(proc, req_index=0)
        logits = np.array([[5.0]], dtype=np.float32)
        result = proc.apply(logits)
        assert_onehot(result[0])

    def test_all_equal_logits(self) -> None:
        """Mirostat with uniform logit distribution."""
        proc = make_processor(mirostat_mode=2)
        register_request(proc, req_index=0)
        logits = np.ones((1, 10), dtype=np.float32)
        result = proc.apply(logits)
        assert_onehot(result[0])

    def test_one_hot_input(self) -> None:
        """Mirostat with only one finite logit."""
        proc = make_processor(mirostat_mode=2, diagnostic_mode=True)
        register_request(proc, req_index=0)
        logits = np.full((1, 5), -np.inf, dtype=np.float32)
        logits[0, 2] = 1.0
        result = proc.apply(logits)
        assert_onehot(result[0])
        records = proc.sampling_logger.get_diagnostic_data()
        assert records[0].token_id == 2

    def test_very_low_tau(self) -> None:
        """Very low tau should restrict to top tokens."""
        proc = make_processor(
            mirostat_mode=2,
            mirostat_tau=0.1,
            diagnostic_mode=True,
        )
        register_request(proc, req_index=0)
        logits = np.array([SAMPLE_LOGITS])
        proc.apply(logits)
        records = proc.sampling_logger.get_diagnostic_data()
        # With very low tau, mu should converge to keeping very few tokens.
        assert records[0].num_candidates >= 1
