"""Tests for TopNSigmaStage -- logit-space sigma filtering."""

from __future__ import annotations

import numpy as np

from tests.helpers import (
    SAMPLE_LOGITS,
    assert_onehot,
    make_processor,
    register_request,
)


class TestTopNSigmaStage:
    """Test the Top-N-Sigma filtering stage."""

    def test_disabled_by_default(self) -> None:
        """Top-N-Sigma has no effect when top_n_sigma=0 (default)."""
        proc = make_processor()
        register_request(proc, req_index=0)
        logits = np.array([SAMPLE_LOGITS])
        result = proc.apply(logits)
        assert_onehot(result[0])

    def test_filters_tokens(self) -> None:
        """Tokens far below max are excluded with a tight sigma."""
        proc = make_processor(top_n_sigma=1.0, diagnostic_mode=True)
        register_request(proc, req_index=0)

        logits = np.array([SAMPLE_LOGITS])
        result = proc.apply(logits)
        assert_onehot(result[0])

        records = proc.sampling_logger.get_diagnostic_data()
        assert len(records) == 1
        # With sigma=1.0, tokens more than 1 std below max are removed.
        # The sample logits span a wide range, so some must be filtered.
        assert records[0].num_candidates < 10

    def test_keeps_at_least_one_token(self) -> None:
        """Even with a very tight sigma, at least one token survives."""
        proc = make_processor(top_n_sigma=0.001)
        register_request(proc, req_index=0)

        logits = np.array([SAMPLE_LOGITS])
        result = proc.apply(logits)
        assert_onehot(result[0])

    def test_per_request_override(self) -> None:
        """top_n_sigma can be overridden per-request."""
        proc = make_processor(diagnostic_mode=True)
        register_request(proc, req_index=0, extra_args={"qr_top_n_sigma": 1.0})

        logits = np.array([SAMPLE_LOGITS])
        proc.apply(logits)

        records = proc.sampling_logger.get_diagnostic_data()
        assert len(records) == 1
        assert records[0].num_candidates < 10

    def test_single_token_vocab(self) -> None:
        """Top-N-Sigma with a single-token vocabulary."""
        proc = make_processor(top_n_sigma=1.0, vocab_size=1)
        register_request(proc, req_index=0)
        logits = np.array([[5.0]], dtype=np.float32)
        result = proc.apply(logits)
        assert_onehot(result[0])

    def test_uniform_logits(self) -> None:
        """When all logits are equal, std=0 so threshold=max and all survive."""
        proc = make_processor(top_n_sigma=1.0, diagnostic_mode=True)
        register_request(proc, req_index=0)
        logits = np.ones((1, 10), dtype=np.float32)
        result = proc.apply(logits)
        assert_onehot(result[0])

        records = proc.sampling_logger.get_diagnostic_data()
        # All tokens equal -> std=0 -> threshold=max -> none below -> all survive.
        assert records[0].num_candidates == 10

    def test_large_sigma_keeps_all(self) -> None:
        """A very large n keeps all tokens."""
        proc = make_processor(top_n_sigma=100.0, diagnostic_mode=True)
        register_request(proc, req_index=0)
        logits = np.array([SAMPLE_LOGITS])
        proc.apply(logits)

        records = proc.sampling_logger.get_diagnostic_data()
        assert records[0].num_candidates == 10
