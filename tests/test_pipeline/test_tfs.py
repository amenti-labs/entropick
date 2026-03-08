"""Tests for TailFreeSamplingStage -- tail-free sampling."""

from __future__ import annotations

import numpy as np

from tests.helpers import (
    SAMPLE_LOGITS,
    assert_onehot,
    make_processor,
    register_request,
)


class TestTailFreeSamplingStage:
    """Test the Tail-Free Sampling stage."""

    def test_disabled_by_default(self) -> None:
        """TFS has no effect when tfs_z=1.0 (default)."""
        proc = make_processor()
        register_request(proc, req_index=0)
        logits = np.array([SAMPLE_LOGITS])
        result = proc.apply(logits)
        assert_onehot(result[0])

    def test_filters_tokens(self) -> None:
        """Low tfs_z removes tail tokens."""
        proc = make_processor(tfs_z=0.5, diagnostic_mode=True)
        register_request(proc, req_index=0)

        logits = np.array([SAMPLE_LOGITS])
        result = proc.apply(logits)
        assert_onehot(result[0])

        records = proc.sampling_logger.get_diagnostic_data()
        assert len(records) == 1
        # With z=0.5, tail tokens are removed.
        assert records[0].num_candidates < 10

    def test_keeps_at_least_one_token(self) -> None:
        """Even with tfs_z=0, at least one token survives."""
        proc = make_processor(tfs_z=0.0)
        register_request(proc, req_index=0)

        logits = np.array([SAMPLE_LOGITS])
        result = proc.apply(logits)
        assert_onehot(result[0])

    def test_per_request_override(self) -> None:
        """tfs_z can be overridden per-request."""
        proc = make_processor(diagnostic_mode=True)
        register_request(proc, req_index=0, extra_args={"qr_tfs_z": 0.5})

        logits = np.array([SAMPLE_LOGITS])
        proc.apply(logits)

        records = proc.sampling_logger.get_diagnostic_data()
        assert len(records) == 1
        assert records[0].num_candidates < 10

    def test_single_token_vocab(self) -> None:
        """TFS with a single-token vocabulary."""
        proc = make_processor(tfs_z=0.5, vocab_size=1)
        register_request(proc, req_index=0)
        logits = np.array([[5.0]], dtype=np.float32)
        result = proc.apply(logits)
        assert_onehot(result[0])

    def test_uniform_logits(self) -> None:
        """When all logits are equal, second derivatives are zero -- no filtering."""
        proc = make_processor(tfs_z=0.5, diagnostic_mode=True)
        register_request(proc, req_index=0)
        logits = np.ones((1, 10), dtype=np.float32)
        result = proc.apply(logits)
        assert_onehot(result[0])

        records = proc.sampling_logger.get_diagnostic_data()
        # Uniform distribution: all second derivatives are 0, so no filtering.
        assert records[0].num_candidates == 10

    def test_two_token_vocab(self) -> None:
        """TFS with two tokens (not enough for second derivative)."""
        proc = make_processor(tfs_z=0.5, vocab_size=2)
        register_request(proc, req_index=0)
        logits = np.array([[3.0, 1.0]], dtype=np.float32)
        result = proc.apply(logits)
        assert_onehot(result[0])

    def test_very_low_z_keeps_top_tokens(self) -> None:
        """Very low z keeps only the most important tokens."""
        proc = make_processor(tfs_z=0.1, diagnostic_mode=True)
        register_request(proc, req_index=0)

        logits = np.array([SAMPLE_LOGITS])
        proc.apply(logits)

        records = proc.sampling_logger.get_diagnostic_data()
        # Very low z -> aggressive tail removal.
        assert records[0].num_candidates <= 5
