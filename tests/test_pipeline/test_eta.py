"""Tests for EtaSamplingStage -- entropy-aware probability cutoff."""

from __future__ import annotations

import numpy as np

from tests.helpers import (
    SAMPLE_LOGITS,
    assert_onehot,
    make_processor,
    register_request,
)


class TestEtaSamplingStage:
    """Test the Eta Sampling stage."""

    def test_disabled_by_default(self) -> None:
        """Eta sampling has no effect when eta_cutoff=0 (default)."""
        proc = make_processor()
        register_request(proc, req_index=0)
        logits = np.array([SAMPLE_LOGITS])
        result = proc.apply(logits)
        assert_onehot(result[0])

    def test_filters_tokens(self) -> None:
        """With a moderate eta_cutoff, low-probability tokens are removed."""
        proc = make_processor(eta_cutoff=100.0, diagnostic_mode=True)
        register_request(proc, req_index=0)

        logits = np.array([SAMPLE_LOGITS])
        result = proc.apply(logits)
        assert_onehot(result[0])

        records = proc.sampling_logger.get_diagnostic_data()
        assert len(records) == 1
        # eta = 100.0 * 1e-4 = 0.01; threshold filters some low-prob tokens.
        assert records[0].num_candidates < 10

    def test_keeps_at_least_one_token(self) -> None:
        """Even with a very high cutoff, at least one token survives."""
        proc = make_processor(eta_cutoff=10000.0)
        register_request(proc, req_index=0)

        logits = np.array([SAMPLE_LOGITS])
        result = proc.apply(logits)
        assert_onehot(result[0])

    def test_per_request_override(self) -> None:
        """eta_cutoff can be overridden per-request."""
        proc = make_processor(diagnostic_mode=True)
        register_request(proc, req_index=0, extra_args={"qr_eta_cutoff": 100.0})

        logits = np.array([SAMPLE_LOGITS])
        proc.apply(logits)

        records = proc.sampling_logger.get_diagnostic_data()
        assert len(records) == 1
        assert records[0].num_candidates < 10

    def test_single_token_vocab(self) -> None:
        """Eta sampling with a single-token vocabulary."""
        proc = make_processor(eta_cutoff=100.0, vocab_size=1)
        register_request(proc, req_index=0)
        logits = np.array([[5.0]], dtype=np.float32)
        result = proc.apply(logits)
        assert_onehot(result[0])

    def test_uniform_logits(self) -> None:
        """When all logits are equal, all have the same probability."""
        proc = make_processor(eta_cutoff=100.0, diagnostic_mode=True)
        register_request(proc, req_index=0)
        logits = np.ones((1, 10), dtype=np.float32)
        result = proc.apply(logits)
        assert_onehot(result[0])

        records = proc.sampling_logger.get_diagnostic_data()
        # Uniform 10 tokens: p=0.1 each, H=ln(10)~2.3.
        # eta=0.01, sqrt(0.01)*exp(-2.3)~0.01, threshold~max(0.01, 0.01)=0.01.
        # All p=0.1 > 0.01 -> all survive.
        assert records[0].num_candidates == 10

    def test_high_cutoff_filters_aggressively(self) -> None:
        """A very high eta_cutoff filters most tokens."""
        proc = make_processor(eta_cutoff=5000.0, diagnostic_mode=True)
        register_request(proc, req_index=0)

        logits = np.array([SAMPLE_LOGITS])
        proc.apply(logits)

        records = proc.sampling_logger.get_diagnostic_data()
        # eta = 0.5, which is very high; most tokens filtered.
        assert records[0].num_candidates <= 3

    def test_low_cutoff_keeps_most(self) -> None:
        """A very low eta_cutoff keeps nearly all tokens."""
        proc = make_processor(eta_cutoff=1.0, diagnostic_mode=True)
        register_request(proc, req_index=0)

        logits = np.array([SAMPLE_LOGITS])
        proc.apply(logits)

        records = proc.sampling_logger.get_diagnostic_data()
        # eta = 0.0001, very small threshold -> most tokens survive.
        # (Temperature scaling may reduce some low-prob tokens further.)
        assert records[0].num_candidates >= 5
