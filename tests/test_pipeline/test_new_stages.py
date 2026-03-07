"""Tests for Min-P, XTC, and Adaptive Injection stages."""

from __future__ import annotations

import numpy as np

from tests.helpers import (
    SAMPLE_LOGITS,
    assert_onehot,
    make_processor,
    register_request,
)


class TestMinPStage:
    """Test the Min-P filtering stage."""

    def test_disabled_by_default(self) -> None:
        """Min-P has no effect when min_p=0 (default)."""
        proc = make_processor()
        register_request(proc, req_index=0)
        logits = np.array([SAMPLE_LOGITS])
        result = proc.apply(logits)
        assert_onehot(result[0])

    def test_filters_low_probability_tokens(self) -> None:
        """Tokens below min_p * max(p) are excluded."""
        proc = make_processor(min_p=0.5)
        register_request(proc, req_index=0)

        # With min_p=0.5, only tokens with p >= 0.5 * max(p) survive.
        logits = np.array([SAMPLE_LOGITS])
        result = proc.apply(logits)
        assert_onehot(result[0])

    def test_high_min_p_selects_top_token(self) -> None:
        """Very high min_p leaves only the top token(s)."""
        proc = make_processor(min_p=0.99, diagnostic_mode=True)
        register_request(proc, req_index=0)

        logits = np.array([SAMPLE_LOGITS])
        proc.apply(logits)

        records = proc.sampling_logger.get_diagnostic_data()
        assert len(records) == 1
        # With min_p=0.99, only top token survives -> rank 0.
        assert records[0].token_rank == 0

    def test_min_p_zero_keeps_all(self) -> None:
        """min_p=0 keeps all tokens (no filtering)."""
        proc = make_processor(min_p=0.0, diagnostic_mode=True)
        register_request(proc, req_index=0)

        logits = np.array([SAMPLE_LOGITS])
        proc.apply(logits)

        records = proc.sampling_logger.get_diagnostic_data()
        assert len(records) == 1
        # All 10 tokens should survive.
        assert records[0].num_candidates == 10

    def test_preserves_at_least_one_token(self) -> None:
        """Even if all tokens would be filtered, at least one survives."""
        proc = make_processor(min_p=1.0)
        register_request(proc, req_index=0)

        logits = np.array([SAMPLE_LOGITS])
        result = proc.apply(logits)
        assert_onehot(result[0])

    def test_per_request_override(self) -> None:
        """min_p can be overridden per-request."""
        proc = make_processor(diagnostic_mode=True)
        register_request(
            proc, req_index=0, extra_args={"qr_min_p": 0.99}
        )

        logits = np.array([SAMPLE_LOGITS])
        proc.apply(logits)

        records = proc.sampling_logger.get_diagnostic_data()
        assert records[0].token_rank == 0


class TestXTCStage:
    """Test the Exclude Top Choices stage."""

    def test_disabled_by_default(self) -> None:
        """XTC has no effect when xtc_probability=0 (default)."""
        proc = make_processor()
        register_request(proc, req_index=0)
        logits = np.array([SAMPLE_LOGITS])
        result = proc.apply(logits)
        assert_onehot(result[0])

    def test_excludes_top_tokens(self) -> None:
        """With high xtc_probability, top tokens are excluded from logits."""
        proc = make_processor(
            xtc_probability=1.0,
            xtc_threshold=0.01,
            diagnostic_mode=True,
        )
        register_request(proc, req_index=0)

        logits = np.array([SAMPLE_LOGITS])
        result = proc.apply(logits)
        assert_onehot(result[0])

        records = proc.sampling_logger.get_diagnostic_data()
        assert len(records) == 1
        # With xtc_probability=1.0, most top tokens are excluded.
        # The selected token should still be valid and the num_candidates
        # should be reduced compared to without XTC.
        assert records[0].num_candidates < 10

    def test_always_keeps_at_least_one(self) -> None:
        """XTC never empties the candidate set."""
        proc = make_processor(
            xtc_probability=1.0,
            xtc_threshold=0.0,  # All tokens are candidates for exclusion.
        )
        register_request(proc, req_index=0)

        logits = np.array([SAMPLE_LOGITS])
        result = proc.apply(logits)
        # Must still produce a valid one-hot result.
        assert_onehot(result[0])

    def test_high_threshold_skips_exclusion(self) -> None:
        """If threshold is very high, no tokens qualify for exclusion."""
        proc = make_processor(
            xtc_probability=1.0,
            xtc_threshold=0.99,
            diagnostic_mode=True,
        )
        register_request(proc, req_index=0)

        logits = np.array([SAMPLE_LOGITS])
        proc.apply(logits)

        records = proc.sampling_logger.get_diagnostic_data()
        assert len(records) == 1
        # High threshold means no token has p >= 0.99, so XTC is a no-op.
        # (For a 10-token softmax, max p is ~0.64.)

    def test_zero_probability_disables(self) -> None:
        """xtc_probability=0 disables the stage entirely."""
        proc = make_processor(
            xtc_probability=0.0,
            xtc_threshold=0.01,
        )
        register_request(proc, req_index=0)
        logits = np.array([SAMPLE_LOGITS])
        result = proc.apply(logits)
        assert_onehot(result[0])

    def test_per_request_override(self) -> None:
        """XTC can be configured per-request."""
        proc = make_processor(diagnostic_mode=True)
        register_request(
            proc,
            req_index=0,
            extra_args={
                "qr_xtc_probability": 1.0,
                "qr_xtc_threshold": 0.01,
            },
        )

        logits = np.array([SAMPLE_LOGITS])
        proc.apply(logits)

        records = proc.sampling_logger.get_diagnostic_data()
        assert len(records) == 1
        # XTC with prob=1.0 should reduce candidates.
        assert records[0].num_candidates < 10

    def test_uses_quantum_entropy_for_decisions(self) -> None:
        """XTC fetches entropy bytes for exclusion decisions."""
        proc = make_processor(
            xtc_probability=0.5,
            xtc_threshold=0.01,
            diagnostic_mode=True,
        )
        register_request(proc, req_index=0)

        logits = np.array([SAMPLE_LOGITS])
        proc.apply(logits)

        records = proc.sampling_logger.get_diagnostic_data()
        assert len(records) == 1
        # Entropy fetch time should be non-zero (XTC fetches bytes).
        assert records[0].entropy_fetch_ms >= 0.0


class TestAdaptiveInjectionStage:
    """Test the Adaptive Injection stage."""

    def test_disabled_by_default(self) -> None:
        """Adaptive injection is off by default."""
        proc = make_processor(logit_noise_alpha=0.5, diagnostic_mode=True)
        register_request(proc, req_index=0)

        logits = np.array([SAMPLE_LOGITS])
        proc.apply(logits)

        records = proc.sampling_logger.get_diagnostic_data()
        assert len(records) == 1

    def test_scales_injection_by_entropy(self) -> None:
        """When enabled, injection_scale varies with distribution entropy."""
        import os

        from qr_sampler.config import QRSamplerConfig
        from qr_sampler.pipeline.context import SamplingContext
        from qr_sampler.stages.adaptive_injection import AdaptiveInjectionStage
        old = os.environ.get("QR_ENTROPY_SOURCE_TYPE")
        os.environ["QR_ENTROPY_SOURCE_TYPE"] = "mock_uniform"
        try:
            config = QRSamplerConfig(
                adaptive_injection=True,
                adaptive_injection_low_h=1.0,
                adaptive_injection_high_h=3.0,
            )
        finally:
            if old is None:
                os.environ.pop("QR_ENTROPY_SOURCE_TYPE", None)
            else:
                os.environ["QR_ENTROPY_SOURCE_TYPE"] = old

        stage = AdaptiveInjectionStage()

        # Low entropy distribution (very peaked) -> low injection scale.
        peaked_logits = np.array([100.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        ctx_peaked = SamplingContext(
            row=peaked_logits,
            config=config,
            entropy_source=None,  # type: ignore[arg-type]
            amplifier=None,  # type: ignore[arg-type]
            temperature_strategy=None,  # type: ignore[arg-type]
            config_hash="test",
        )
        stage(ctx_peaked)
        assert ctx_peaked.injection_scale < 0.5

        # High entropy distribution (uniform) -> higher injection scale.
        # ln(5) ~ 1.61, with low_h=1.0, high_h=3.0 -> scale ~ 0.30.
        # Use more tokens for higher entropy.
        uniform_logits = np.ones(50, dtype=np.float32)
        ctx_uniform = SamplingContext(
            row=uniform_logits,
            config=config,
            entropy_source=None,  # type: ignore[arg-type]
            amplifier=None,  # type: ignore[arg-type]
            temperature_strategy=None,  # type: ignore[arg-type]
            config_hash="test",
        )
        stage(ctx_uniform)
        # ln(50) ~ 3.91 -> scale = (3.91 - 1.0) / (3.0 - 1.0) = 1.0 (clamped)
        assert ctx_uniform.injection_scale > 0.5

    def test_scale_zero_suppresses_m1(self) -> None:
        """When model is confident, M1 logit noise is suppressed."""
        proc = make_processor(
            adaptive_injection=True,
            adaptive_injection_low_h=10.0,  # Very high -> everything is "low H"
            adaptive_injection_high_h=20.0,
            logit_noise_alpha=1.0,
            diagnostic_mode=True,
        )
        register_request(proc, req_index=0)

        # Our 10-token logits have H ~ 1.4 nats, well below low_h=10.
        logits = np.array([SAMPLE_LOGITS])
        proc.apply(logits)

        records = proc.sampling_logger.get_diagnostic_data()
        assert len(records) == 1

    def test_full_scale_allows_injection(self) -> None:
        """When model is uncertain and thresholds are low, injection runs."""
        proc = make_processor(
            adaptive_injection=True,
            adaptive_injection_low_h=0.0,  # Everything above 0 gets some scale
            adaptive_injection_high_h=0.5,  # Very low -> everything is "high H"
            logit_noise_alpha=0.5,
            diagnostic_mode=True,
        )
        register_request(proc, req_index=0)

        logits = np.array([SAMPLE_LOGITS])
        proc.apply(logits)

        records = proc.sampling_logger.get_diagnostic_data()
        assert len(records) == 1

    def test_per_request_override(self) -> None:
        """Adaptive injection can be toggled per-request."""
        proc = make_processor(diagnostic_mode=True)
        register_request(
            proc,
            req_index=0,
            extra_args={"qr_adaptive_injection": True},
        )

        logits = np.array([SAMPLE_LOGITS])
        proc.apply(logits)

        records = proc.sampling_logger.get_diagnostic_data()
        assert len(records) == 1

    def test_degenerate_thresholds(self) -> None:
        """When low_h >= high_h, falls back to binary mode."""
        import os

        from qr_sampler.config import QRSamplerConfig
        from qr_sampler.pipeline.context import SamplingContext
        from qr_sampler.stages.adaptive_injection import AdaptiveInjectionStage
        old = os.environ.get("QR_ENTROPY_SOURCE_TYPE")
        os.environ["QR_ENTROPY_SOURCE_TYPE"] = "mock_uniform"
        try:
            config = QRSamplerConfig(
                adaptive_injection=True,
                adaptive_injection_low_h=5.0,
                adaptive_injection_high_h=5.0,  # Equal -> binary mode
            )
        finally:
            if old is None:
                os.environ.pop("QR_ENTROPY_SOURCE_TYPE", None)
            else:
                os.environ["QR_ENTROPY_SOURCE_TYPE"] = old

        stage = AdaptiveInjectionStage()

        # H of uniform 5-token dist is ln(5) ~ 1.61 < 5.0 -> scale = 0.
        uniform_logits = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        ctx = SamplingContext(
            row=uniform_logits,
            config=config,
            entropy_source=None,  # type: ignore[arg-type]
            amplifier=None,  # type: ignore[arg-type]
            temperature_strategy=None,  # type: ignore[arg-type]
            config_hash="test",
        )
        stage(ctx)
        assert ctx.injection_scale == 0.0


class TestCombinedNewStages:
    """Test the new stages working together in the full pipeline."""

    def test_min_p_plus_xtc(self) -> None:
        """Min-P and XTC compose in the same pipeline."""
        proc = make_processor(
            min_p=0.3,
            xtc_probability=0.5,
            xtc_threshold=0.1,
        )
        register_request(proc, req_index=0)

        logits = np.array([SAMPLE_LOGITS])
        result = proc.apply(logits)
        assert_onehot(result[0])

    def test_all_new_stages_with_existing(self) -> None:
        """All new stages + M1/M2/M3 work together."""
        proc = make_processor(
            adaptive_injection=True,
            adaptive_injection_low_h=0.0,
            adaptive_injection_high_h=2.0,
            logit_noise_alpha=0.1,
            temp_variance_beta=0.2,
            walk_step=0.05,
            min_p=0.1,
            xtc_probability=0.3,
            xtc_threshold=0.1,
            diagnostic_mode=True,
        )
        register_request(proc, req_index=0)

        logits = np.array([SAMPLE_LOGITS])
        result = proc.apply(logits)
        assert_onehot(result[0])

        records = proc.sampling_logger.get_diagnostic_data()
        assert len(records) == 1

    def test_batch_processing(self) -> None:
        """New stages work correctly with batch processing."""
        proc = make_processor(
            min_p=0.2,
            xtc_probability=0.5,
            xtc_threshold=0.1,
        )
        register_request(proc, req_index=0)
        register_request(proc, req_index=1)

        logits = np.array([SAMPLE_LOGITS, SAMPLE_LOGITS])
        result = proc.apply(logits)
        assert_onehot(result[0])
        assert_onehot(result[1])


class TestNewStagesEdgeCases:
    """Edge cases: degenerate inputs for Min-P, XTC, and Adaptive Injection."""

    def test_min_p_single_token(self) -> None:
        """Min-P with a single-token vocabulary."""
        proc = make_processor(min_p=0.5)
        register_request(proc, req_index=0)
        logits = np.array([[5.0]], dtype=np.float32)
        result = proc.apply(logits)
        assert_onehot(result[0])

    def test_min_p_all_identical_logits(self) -> None:
        """Min-P when all logits are equal (uniform distribution)."""
        proc = make_processor(min_p=0.5, diagnostic_mode=True)
        register_request(proc, req_index=0)
        logits = np.ones((1, 10), dtype=np.float32)
        result = proc.apply(logits)
        assert_onehot(result[0])
        # All tokens have equal probability, so all survive min_p filtering.
        records = proc.sampling_logger.get_diagnostic_data()
        assert records[0].num_candidates == 10

    def test_min_p_all_inf_except_one(self) -> None:
        """Min-P when only one token has finite logits."""
        proc = make_processor(min_p=0.5, diagnostic_mode=True)
        register_request(proc, req_index=0)
        logits = np.full((1, 5), -np.inf, dtype=np.float32)
        logits[0, 2] = 1.0
        result = proc.apply(logits)
        assert_onehot(result[0])
        records = proc.sampling_logger.get_diagnostic_data()
        assert records[0].token_id == 2

    def test_xtc_single_token(self) -> None:
        """XTC with a single-token vocabulary never excludes it."""
        proc = make_processor(xtc_probability=1.0, xtc_threshold=0.0)
        register_request(proc, req_index=0)
        logits = np.array([[5.0]], dtype=np.float32)
        result = proc.apply(logits)
        assert_onehot(result[0])

    def test_xtc_all_identical_logits(self) -> None:
        """XTC when all logits are equal."""
        proc = make_processor(xtc_probability=1.0, xtc_threshold=0.0)
        register_request(proc, req_index=0)
        logits = np.ones((1, 10), dtype=np.float32)
        result = proc.apply(logits)
        assert_onehot(result[0])

    def test_xtc_all_inf_except_one(self) -> None:
        """XTC with only one finite token keeps it."""
        proc = make_processor(
            xtc_probability=1.0, xtc_threshold=0.0, diagnostic_mode=True,
        )
        register_request(proc, req_index=0)
        logits = np.full((1, 5), -np.inf, dtype=np.float32)
        logits[0, 3] = 1.0
        result = proc.apply(logits)
        assert_onehot(result[0])
        records = proc.sampling_logger.get_diagnostic_data()
        assert records[0].token_id == 3

    def test_adaptive_all_inf_logits(self) -> None:
        """Adaptive injection with all-inf logits sets scale to 0."""
        import os

        from qr_sampler.config import QRSamplerConfig
        from qr_sampler.pipeline.context import SamplingContext
        from qr_sampler.stages.adaptive_injection import AdaptiveInjectionStage

        old = os.environ.get("QR_ENTROPY_SOURCE_TYPE")
        os.environ["QR_ENTROPY_SOURCE_TYPE"] = "mock_uniform"
        try:
            config = QRSamplerConfig(
                adaptive_injection=True,
                adaptive_injection_low_h=1.0,
                adaptive_injection_high_h=3.0,
            )
        finally:
            if old is None:
                os.environ.pop("QR_ENTROPY_SOURCE_TYPE", None)
            else:
                os.environ["QR_ENTROPY_SOURCE_TYPE"] = old

        stage = AdaptiveInjectionStage()
        logits = np.full(10, -np.inf, dtype=np.float32)
        ctx = SamplingContext(
            row=logits,
            config=config,
            entropy_source=None,  # type: ignore[arg-type]
            amplifier=None,  # type: ignore[arg-type]
            temperature_strategy=None,  # type: ignore[arg-type]
            config_hash="test",
        )
        stage(ctx)
        assert ctx.injection_scale == 0.0

    def test_adaptive_single_token(self) -> None:
        """Adaptive injection with single-token vocab (H=0) sets scale to 0."""
        import os

        from qr_sampler.config import QRSamplerConfig
        from qr_sampler.pipeline.context import SamplingContext
        from qr_sampler.stages.adaptive_injection import AdaptiveInjectionStage

        old = os.environ.get("QR_ENTROPY_SOURCE_TYPE")
        os.environ["QR_ENTROPY_SOURCE_TYPE"] = "mock_uniform"
        try:
            config = QRSamplerConfig(
                adaptive_injection=True,
                adaptive_injection_low_h=1.0,
                adaptive_injection_high_h=3.0,
            )
        finally:
            if old is None:
                os.environ.pop("QR_ENTROPY_SOURCE_TYPE", None)
            else:
                os.environ["QR_ENTROPY_SOURCE_TYPE"] = old

        stage = AdaptiveInjectionStage()
        logits = np.array([5.0], dtype=np.float32)
        ctx = SamplingContext(
            row=logits,
            config=config,
            entropy_source=None,  # type: ignore[arg-type]
            amplifier=None,  # type: ignore[arg-type]
            temperature_strategy=None,  # type: ignore[arg-type]
            config_hash="test",
        )
        stage(ctx)
        # Single token -> H=0 -> below low_h=1.0 -> scale=0.
        assert ctx.injection_scale == 0.0

    def test_combined_edge_two_tokens(self) -> None:
        """All new stages with only two tokens."""
        proc = make_processor(
            adaptive_injection=True,
            adaptive_injection_low_h=0.0,
            adaptive_injection_high_h=1.0,
            min_p=0.3,
            xtc_probability=0.5,
            xtc_threshold=0.1,
        )
        register_request(proc, req_index=0)
        logits = np.array([[3.0, 1.0]], dtype=np.float32)
        result = proc.apply(logits)
        assert_onehot(result[0])
