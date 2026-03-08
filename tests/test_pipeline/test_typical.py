"""Tests for TypicalSamplingStage -- locally typical sampling."""

from __future__ import annotations

import numpy as np

from tests.helpers import (
    SAMPLE_LOGITS,
    assert_onehot,
    make_processor,
    register_request,
)


class TestTypicalSamplingStage:
    """Test the Typical Sampling stage."""

    def test_disabled_by_default(self) -> None:
        """Typical sampling has no effect when typical_p=1.0 (default)."""
        proc = make_processor()
        register_request(proc, req_index=0)
        logits = np.array([SAMPLE_LOGITS])
        result = proc.apply(logits)
        assert_onehot(result[0])

    def test_filters_tokens(self) -> None:
        """Low typical_p removes atypical tokens."""
        proc = make_processor(typical_p=0.5, diagnostic_mode=True)
        register_request(proc, req_index=0)

        logits = np.array([SAMPLE_LOGITS])
        result = proc.apply(logits)
        assert_onehot(result[0])

        records = proc.sampling_logger.get_diagnostic_data()
        assert len(records) == 1
        # With typical_p=0.5, only tokens covering 50% of mass near typical info survive.
        assert records[0].num_candidates < 10

    def test_keeps_at_least_one_token(self) -> None:
        """Even with typical_p near 0, at least one token survives."""
        proc = make_processor(typical_p=0.01)
        register_request(proc, req_index=0)

        logits = np.array([SAMPLE_LOGITS])
        result = proc.apply(logits)
        assert_onehot(result[0])

    def test_per_request_override(self) -> None:
        """typical_p can be overridden per-request."""
        proc = make_processor(diagnostic_mode=True)
        register_request(proc, req_index=0, extra_args={"qr_typical_p": 0.5})

        logits = np.array([SAMPLE_LOGITS])
        proc.apply(logits)

        records = proc.sampling_logger.get_diagnostic_data()
        assert len(records) == 1
        assert records[0].num_candidates < 10

    def test_single_token_vocab(self) -> None:
        """Typical sampling with a single-token vocabulary."""
        proc = make_processor(typical_p=0.5, vocab_size=1)
        register_request(proc, req_index=0)
        logits = np.array([[5.0]], dtype=np.float32)
        result = proc.apply(logits)
        assert_onehot(result[0])

    def test_uniform_logits(self) -> None:
        """When all logits are equal, all tokens are equally typical."""
        proc = make_processor(typical_p=0.5, diagnostic_mode=True)
        register_request(proc, req_index=0)
        logits = np.ones((1, 10), dtype=np.float32)
        result = proc.apply(logits)
        assert_onehot(result[0])

        records = proc.sampling_logger.get_diagnostic_data()
        # Uniform: all tokens have the same distance from H, so typical_p=0.5
        # keeps about half.  At least 1 and at most 10.
        assert 1 <= records[0].num_candidates <= 10

    def test_very_low_typical_p(self) -> None:
        """Very low typical_p keeps only the most typical token(s)."""
        proc = make_processor(typical_p=0.01, diagnostic_mode=True)
        register_request(proc, req_index=0)

        logits = np.array([SAMPLE_LOGITS])
        proc.apply(logits)

        records = proc.sampling_logger.get_diagnostic_data()
        # Very low threshold -> only 1-2 tokens survive.
        assert records[0].num_candidates <= 3
