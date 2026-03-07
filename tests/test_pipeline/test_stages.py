"""Tests for the pipeline stage architecture.

Verifies the PipelineStage protocol, StageRegistry, SamplingContext,
custom pipeline construction, and stage composition.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from qr_sampler.pipeline.registry import StageRegistry
from qr_sampler.pipeline.stage import PipelineStage

if TYPE_CHECKING:
    from qr_sampler.pipeline.context import SamplingContext
from qr_sampler.stages import (
    AdaptiveInjectionStage,
    CorrelatedWalkStage,
    EntropyFetchStage,
    LogitNoiseStage,
    MinPStage,
    SelectionStage,
    TemperatureStage,
    TempVarianceStage,
    XTCStage,
    build_default_pipeline,
)
from tests.helpers import (
    SAMPLE_LOGITS,
    assert_onehot,
    make_processor,
    register_request,
)


class TestPipelineStageProtocol:
    """Test that all built-in stages satisfy the PipelineStage protocol."""

    @pytest.mark.parametrize(
        "stage_cls",
        [
            AdaptiveInjectionStage,
            LogitNoiseStage,
            TemperatureStage,
            TempVarianceStage,
            MinPStage,
            XTCStage,
            EntropyFetchStage,
            CorrelatedWalkStage,
            SelectionStage,
        ],
    )
    def test_stage_is_protocol_compliant(self, stage_cls: type) -> None:
        """Each built-in stage satisfies the PipelineStage protocol."""
        stage = stage_cls()
        assert isinstance(stage, PipelineStage)
        assert hasattr(stage, "name")
        assert isinstance(stage.name, str)
        assert callable(stage)

    def test_custom_stage_protocol_compliance(self) -> None:
        """A plain class with name + __call__ satisfies PipelineStage."""

        class MyStage:
            name: str = "my_custom_stage"

            def __call__(self, ctx: SamplingContext) -> None:
                ctx.temperature *= 2.0

        stage = MyStage()
        assert isinstance(stage, PipelineStage)


class TestStageRegistry:
    """Test the StageRegistry discovery and lookup."""

    def test_builtin_stages_registered(self) -> None:
        """All 9 built-in stages are registered."""
        registered = StageRegistry.list_registered()
        expected = {
            "adaptive_injection",
            "logit_noise",
            "temperature",
            "temp_variance",
            "min_p",
            "xtc",
            "entropy_fetch",
            "correlated_walk",
            "selection",
        }
        assert expected.issubset(set(registered))

    def test_get_known_stage(self) -> None:
        """Can look up a registered stage by name."""
        cls = StageRegistry.get("selection")
        assert cls is SelectionStage

    def test_get_unknown_stage_raises(self) -> None:
        """Looking up an unknown stage raises KeyError."""
        with pytest.raises(KeyError, match="nonexistent"):
            StageRegistry.get("nonexistent")


class TestBuildDefaultPipeline:
    """Test the default pipeline builder."""

    def test_returns_9_stages(self) -> None:
        """Default pipeline has 9 stages."""
        pipeline = build_default_pipeline()
        assert len(pipeline) == 9

    def test_returns_fresh_list(self) -> None:
        """Each call returns a new list (safe to mutate)."""
        p1 = build_default_pipeline()
        p2 = build_default_pipeline()
        assert p1 is not p2

    def test_correct_order(self) -> None:
        """Stages are in the correct pipeline order."""
        pipeline = build_default_pipeline()
        names = [s.name for s in pipeline]
        assert names == [
            "adaptive_injection",
            "logit_noise",
            "temperature",
            "temp_variance",
            "min_p",
            "xtc",
            "entropy_fetch",
            "correlated_walk",
            "selection",
        ]


class TestCustomPipeline:
    """Test processor with custom pipeline configurations."""

    def test_custom_pipeline_subset(self) -> None:
        """Processor works with a subset of stages (skip injection methods)."""
        minimal_pipeline: list[Any] = [
            TemperatureStage(),
            EntropyFetchStage(),
            SelectionStage(),
        ]
        proc = make_processor()
        proc._pipeline = minimal_pipeline

        logits = np.array([SAMPLE_LOGITS])
        result = proc.apply(logits)
        assert_onehot(result[0])

    def test_custom_stage_in_pipeline(self) -> None:
        """A custom stage can be inserted into the pipeline."""

        class DoubleTemperatureStage:
            name: str = "double_temp"

            def __call__(self, ctx: SamplingContext) -> None:
                ctx.temperature *= 2.0

        pipeline: list[Any] = [
            TemperatureStage(),
            DoubleTemperatureStage(),
            EntropyFetchStage(),
            SelectionStage(),
        ]
        proc = make_processor(diagnostic_mode=True)
        proc._pipeline = pipeline

        logits = np.array([SAMPLE_LOGITS])
        result = proc.apply(logits)
        assert_onehot(result[0])

        # Verify temperature was doubled.
        records = proc.sampling_logger.get_diagnostic_data()
        assert len(records) == 1
        # Default fixed temp is 0.7, doubled = 1.4
        assert records[0].temperature_used == pytest.approx(1.4)

    def test_pipeline_property(self) -> None:
        """Processor exposes its pipeline via property."""
        proc = make_processor()
        assert len(proc.pipeline) == 9
        assert all(isinstance(s, PipelineStage) for s in proc.pipeline)

    def test_processor_accepts_pipeline_arg(self) -> None:
        """Processor __init__ accepts a custom pipeline."""
        import os

        minimal: list[Any] = [
            TemperatureStage(),
            EntropyFetchStage(),
            SelectionStage(),
        ]

        old_env: dict[str, str | None] = {}
        env_vars = {
            "QR_ENTROPY_SOURCE_TYPE": "mock_uniform",
            "QR_FALLBACK_MODE": "error",
            "QR_LOG_LEVEL": "none",
        }
        for key, value in env_vars.items():
            old_env[key] = os.environ.get(key)
            os.environ[key] = value
        try:
            from qr_sampler.processor import QRSamplerLogitsProcessor

            proc = QRSamplerLogitsProcessor(
                vllm_config=None,
                pipeline=minimal,
            )
            assert len(proc.pipeline) == 3
            assert proc.pipeline[0].name == "temperature"
        finally:
            for key, original in old_env.items():
                if original is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original


class TestStageState:
    """Test persistent stage state via stage_state dict."""

    def test_correlated_walk_uses_stage_state(self) -> None:
        """CorrelatedWalk reads/writes stage_state instead of walk_position."""
        proc = make_processor(walk_step=0.1)
        register_request(proc, req_index=0)

        # Verify initial walk position is in stage_state.
        state = proc._request_states[0]
        assert "correlated_walk.position" in state.stage_state
        assert state.stage_state["correlated_walk.position"] == pytest.approx(0.5)

        # Apply and verify it changed.
        logits = np.array([SAMPLE_LOGITS])
        proc.apply(logits)

        new_pos = state.stage_state["correlated_walk.position"]
        assert new_pos != pytest.approx(0.5)

    def test_walk_position_property_compat(self) -> None:
        """_RequestState.walk_position property reads from stage_state."""
        proc = make_processor(walk_step=0.1)
        register_request(proc, req_index=0)

        state = proc._request_states[0]
        assert state.walk_position == pytest.approx(0.5)

        state.walk_position = 0.75
        assert state.stage_state["correlated_walk.position"] == pytest.approx(0.75)

    def test_m3_marks_diagnostics_nan(self) -> None:
        """When M3 is active, amplifier diagnostics are NaN."""
        proc = make_processor(walk_step=0.1, diagnostic_mode=True)
        register_request(proc, req_index=0)

        logits = np.array([SAMPLE_LOGITS])
        proc.apply(logits)

        records = proc.sampling_logger.get_diagnostic_data()
        assert len(records) == 1
        assert math.isnan(records[0].sample_mean)
        assert math.isnan(records[0].z_score)
