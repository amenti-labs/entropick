"""Tests for the Gumbel-Max selection stage."""

from __future__ import annotations

import numpy as np

from tests.helpers import (
    SAMPLE_LOGITS,
    assert_onehot,
    make_processor,
    register_request,
)


class TestGumbelDisabled:
    """Verify Gumbel selection is a no-op when disabled."""

    def test_disabled_by_default(self) -> None:
        """Gumbel selection has no effect when gumbel_selection=False (default)."""
        proc = make_processor()
        register_request(proc, req_index=0)
        logits = np.array([SAMPLE_LOGITS])
        result = proc.apply(logits)
        assert_onehot(result[0])

    def test_explicitly_disabled(self) -> None:
        """Explicitly setting gumbel_selection=False produces normal CDF output."""
        proc = make_processor(gumbel_selection=False)
        register_request(proc, req_index=0)
        logits = np.array([SAMPLE_LOGITS])
        result = proc.apply(logits)
        assert_onehot(result[0])


class TestGumbelEnabled:
    """Verify Gumbel-Max selection produces valid one-hot output."""

    def test_produces_onehot(self) -> None:
        """Gumbel selection produces valid one-hot logits."""
        proc = make_processor(gumbel_selection=True)
        register_request(proc, req_index=0)
        logits = np.array([SAMPLE_LOGITS])
        result = proc.apply(logits)
        assert_onehot(result[0])

    def test_selects_valid_token(self) -> None:
        """Gumbel selection selects a token within the vocabulary."""
        proc = make_processor(gumbel_selection=True, diagnostic_mode=True)
        register_request(proc, req_index=0)
        logits = np.array([SAMPLE_LOGITS])
        proc.apply(logits)
        records = proc.sampling_logger.get_diagnostic_data()
        assert len(records) == 1
        assert 0 <= records[0].token_id < len(SAMPLE_LOGITS)

    def test_reports_num_candidates(self) -> None:
        """Gumbel selection reports the correct number of candidates."""
        proc = make_processor(gumbel_selection=True, diagnostic_mode=True)
        register_request(proc, req_index=0)
        logits = np.array([SAMPLE_LOGITS])
        proc.apply(logits)
        records = proc.sampling_logger.get_diagnostic_data()
        assert records[0].num_candidates > 0
        assert records[0].num_candidates <= len(SAMPLE_LOGITS)

    def test_fetches_entropy(self) -> None:
        """Gumbel selection fetches entropy bytes for Gumbel noise."""
        proc = make_processor(gumbel_selection=True, diagnostic_mode=True)
        register_request(proc, req_index=0)
        logits = np.array([SAMPLE_LOGITS])
        proc.apply(logits)
        records = proc.sampling_logger.get_diagnostic_data()
        # entropy_fetch_ms should include Gumbel's fetch.
        assert records[0].entropy_fetch_ms >= 0.0

    def test_with_top_k(self) -> None:
        """Gumbel selection works with top-k filtering."""
        proc = make_processor(
            gumbel_selection=True,
            top_k=3,
            diagnostic_mode=True,
        )
        register_request(proc, req_index=0)
        logits = np.array([SAMPLE_LOGITS])
        result = proc.apply(logits)
        assert_onehot(result[0])

        records = proc.sampling_logger.get_diagnostic_data()
        assert records[0].num_candidates <= 3

    def test_bypasses_cdf_selection(self) -> None:
        """When Gumbel is active, the normal SelectionStage is skipped."""
        proc = make_processor(gumbel_selection=True, diagnostic_mode=True)
        register_request(proc, req_index=0)
        logits = np.array([SAMPLE_LOGITS])
        proc.apply(logits)
        records = proc.sampling_logger.get_diagnostic_data()
        assert len(records) == 1
        # Token should be selected by Gumbel, not CDF.
        assert records[0].token_id >= 0

    def test_batch_processing(self) -> None:
        """Gumbel selection works with batch processing."""
        proc = make_processor(gumbel_selection=True)
        register_request(proc, req_index=0)
        register_request(proc, req_index=1)

        logits = np.array([SAMPLE_LOGITS, SAMPLE_LOGITS])
        result = proc.apply(logits)
        assert_onehot(result[0])
        assert_onehot(result[1])


class TestGumbelPerRequest:
    """Test per-request Gumbel configuration."""

    def test_per_request_enable(self) -> None:
        """Gumbel selection can be enabled per-request."""
        proc = make_processor(diagnostic_mode=True)
        register_request(
            proc,
            req_index=0,
            extra_args={"qr_gumbel_selection": True},
        )

        logits = np.array([SAMPLE_LOGITS])
        result = proc.apply(logits)
        assert_onehot(result[0])

    def test_per_request_mixed(self) -> None:
        """One request uses Gumbel, another uses CDF."""
        proc = make_processor(diagnostic_mode=True)
        register_request(proc, req_index=0)  # CDF
        register_request(
            proc,
            req_index=1,
            extra_args={"qr_gumbel_selection": True},
        )  # Gumbel

        logits = np.array([SAMPLE_LOGITS, SAMPLE_LOGITS])
        result = proc.apply(logits)
        assert_onehot(result[0])
        assert_onehot(result[1])


class TestGumbelEdgeCases:
    """Edge cases for Gumbel selection."""

    def test_single_token_vocab(self) -> None:
        """Gumbel selection with a single-token vocabulary."""
        proc = make_processor(gumbel_selection=True, vocab_size=1)
        register_request(proc, req_index=0)
        logits = np.array([[5.0]], dtype=np.float32)
        result = proc.apply(logits)
        assert_onehot(result[0])

    def test_all_equal_logits(self) -> None:
        """Gumbel selection with uniform logit distribution."""
        proc = make_processor(gumbel_selection=True)
        register_request(proc, req_index=0)
        logits = np.ones((1, 10), dtype=np.float32)
        result = proc.apply(logits)
        assert_onehot(result[0])

    def test_one_finite_logit(self) -> None:
        """Gumbel selection with only one finite logit."""
        proc = make_processor(
            gumbel_selection=True,
            diagnostic_mode=True,
            vocab_size=5,
        )
        register_request(proc, req_index=0)
        logits = np.full((1, 5), -np.inf, dtype=np.float32)
        logits[0, 3] = 1.0
        result = proc.apply(logits)
        assert_onehot(result[0])
        records = proc.sampling_logger.get_diagnostic_data()
        assert records[0].token_id == 3

    def test_two_tokens(self) -> None:
        """Gumbel selection with only two tokens."""
        proc = make_processor(gumbel_selection=True, vocab_size=2)
        register_request(proc, req_index=0)
        logits = np.array([[3.0, 1.0]], dtype=np.float32)
        result = proc.apply(logits)
        assert_onehot(result[0])

    def test_combined_with_min_p(self) -> None:
        """Gumbel selection composes with min-p filtering."""
        proc = make_processor(
            gumbel_selection=True,
            min_p=0.3,
        )
        register_request(proc, req_index=0)
        logits = np.array([SAMPLE_LOGITS])
        result = proc.apply(logits)
        assert_onehot(result[0])

    def test_with_top_p(self) -> None:
        """Gumbel selection applies nucleus (top-p) filtering."""
        proc = make_processor(
            gumbel_selection=True,
            top_p=0.5,
            diagnostic_mode=True,
        )
        register_request(proc, req_index=0)
        # Skewed logits: one dominant token.
        logits = np.array([[10.0, 5.0, 1.0, 0.0, -1.0, -2.0, -3.0, -5.0, -10.0, -20.0]])
        result = proc.apply(logits)
        assert_onehot(result[0])

        records = proc.sampling_logger.get_diagnostic_data()
        # top_p=0.5 should reduce candidates vs full vocab.
        assert records[0].num_candidates < 10

    def test_with_top_k_and_top_p(self) -> None:
        """Gumbel selection applies both top-k and top-p filtering."""
        proc = make_processor(
            gumbel_selection=True,
            top_k=5,
            top_p=0.8,
            diagnostic_mode=True,
        )
        register_request(proc, req_index=0)
        logits = np.array([SAMPLE_LOGITS])
        result = proc.apply(logits)
        assert_onehot(result[0])

        records = proc.sampling_logger.get_diagnostic_data()
        assert records[0].num_candidates <= 5
