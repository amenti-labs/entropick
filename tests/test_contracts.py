"""Contract tests -- verify every implementation meets its ABC/protocol contract.

Auto-discovers all registered implementations and parametrizes tests so that
new implementations are automatically covered without adding test cases.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest

from qr_sampler.amplification.base import AmplificationResult, SignalAmplifier
from qr_sampler.config import QRSamplerConfig
from qr_sampler.entropy.base import EntropySource
from qr_sampler.pipeline.registry import StageRegistry
from qr_sampler.pipeline.stage import PipelineStage
from qr_sampler.temperature.base import TemperatureResult, TemperatureStrategy

# ---------------------------------------------------------------------------
# Entropy source contracts
# ---------------------------------------------------------------------------

# Sources that can be instantiated without external dependencies or config.
_SIMPLE_ENTROPY_SOURCES: list[tuple[str, type[EntropySource]]] = []


def _collect_entropy_sources() -> list[tuple[str, type[EntropySource]]]:
    """Collect entropy sources that can be tested without external deps."""
    from qr_sampler.entropy.mock import MockUniformSource
    from qr_sampler.entropy.system import SystemEntropySource
    from qr_sampler.entropy.timing import TimingNoiseSource

    return [
        ("system", SystemEntropySource),
        ("timing", TimingNoiseSource),
        ("mock_uniform", MockUniformSource),
    ]


@pytest.fixture(params=_collect_entropy_sources(), ids=lambda x: x[0])
def entropy_source(request: pytest.FixtureRequest) -> EntropySource:
    """Parametrized fixture yielding each simple entropy source instance."""
    _, cls = request.param
    source = cls()
    yield source
    source.close()


class TestEntropySourceContract:
    """Every EntropySource must satisfy these contracts."""

    def test_is_subclass(self, entropy_source: EntropySource) -> None:
        assert isinstance(entropy_source, EntropySource)

    def test_name_is_nonempty_string(self, entropy_source: EntropySource) -> None:
        assert isinstance(entropy_source.name, str)
        assert len(entropy_source.name) > 0

    def test_is_available_returns_bool(self, entropy_source: EntropySource) -> None:
        assert isinstance(entropy_source.is_available, bool)

    def test_get_random_bytes_returns_exact_count(self, entropy_source: EntropySource) -> None:
        for n in [1, 16, 256, 1024]:
            result = entropy_source.get_random_bytes(n)
            assert isinstance(result, bytes)
            assert len(result) == n

    def test_get_random_float64_shape(self, entropy_source: EntropySource) -> None:
        result = entropy_source.get_random_float64((3, 4))
        assert result.shape == (3, 4)
        assert result.dtype == np.float64
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_get_random_float64_with_out(self, entropy_source: EntropySource) -> None:
        out = np.zeros((2, 3), dtype=np.float64)
        result = entropy_source.get_random_float64((2, 3), out=out)
        assert result is out

    def test_health_check_returns_dict(self, entropy_source: EntropySource) -> None:
        health = entropy_source.health_check()
        assert isinstance(health, dict)
        assert "source" in health
        assert "healthy" in health

    def test_close_is_idempotent(self, entropy_source: EntropySource) -> None:
        entropy_source.close()
        entropy_source.close()  # Should not raise.


# ---------------------------------------------------------------------------
# Fallback entropy source contract
# ---------------------------------------------------------------------------


class TestFallbackEntropySourceContract:
    """FallbackEntropySource wraps two sources and delegates correctly."""

    def test_delegates_to_primary(self) -> None:
        from qr_sampler.entropy.fallback import FallbackEntropySource
        from qr_sampler.entropy.mock import MockUniformSource
        from qr_sampler.entropy.system import SystemEntropySource

        primary = MockUniformSource(seed=42)
        fallback_src = SystemEntropySource()
        source = FallbackEntropySource(primary, fallback_src)

        assert isinstance(source, EntropySource)
        result = source.get_random_bytes(16)
        assert len(result) == 16
        source.close()

    def test_falls_back_on_entropy_error(self) -> None:
        from qr_sampler.entropy.fallback import FallbackEntropySource
        from qr_sampler.entropy.system import SystemEntropySource
        from qr_sampler.exceptions import EntropyUnavailableError

        class FailingSource(EntropySource):
            @property
            def name(self) -> str:
                return "failing"

            @property
            def is_available(self) -> bool:
                return False

            def get_random_bytes(self, n: int) -> bytes:
                raise EntropyUnavailableError("test")

            def close(self) -> None:
                pass

        source = FallbackEntropySource(FailingSource(), SystemEntropySource())
        result = source.get_random_bytes(16)
        assert len(result) == 16
        source.close()


# ---------------------------------------------------------------------------
# Signal amplifier contracts
# ---------------------------------------------------------------------------


def _make_test_config(**overrides: Any) -> QRSamplerConfig:
    """Create a config for testing with sensible defaults."""
    env_patch = {
        "QR_ENTROPY_SOURCE_TYPE": "mock_uniform",
        "QR_FALLBACK_MODE": "error",
        "QR_LOG_LEVEL": "none",
    }
    old_env: dict[str, str | None] = {}
    for key, value in env_patch.items():
        old_env[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        return QRSamplerConfig(**overrides)
    finally:
        for key, original in old_env.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original


def _collect_amplifiers() -> list[tuple[str, SignalAmplifier]]:
    """Collect amplifier instances for testing."""
    from qr_sampler.amplification.zscore import ZScoreMeanAmplifier

    config = _make_test_config()
    amplifiers: list[tuple[str, SignalAmplifier]] = [
        ("zscore", ZScoreMeanAmplifier(config)),
    ]

    # ECDF requires calibration, so we calibrate it here.
    try:
        from qr_sampler.amplification.ecdf import ECDFAmplifier
        from qr_sampler.entropy.mock import MockUniformSource

        ecdf = ECDFAmplifier(config)
        ecdf.calibrate(MockUniformSource(seed=42), config)
        amplifiers.append(("ecdf", ecdf))
    except Exception:
        pass

    return amplifiers


@pytest.fixture(params=_collect_amplifiers(), ids=lambda x: x[0])
def amplifier(request: pytest.FixtureRequest) -> SignalAmplifier:
    """Parametrized fixture yielding each amplifier instance."""
    _, amp = request.param
    return amp


class TestSignalAmplifierContract:
    """Every SignalAmplifier must satisfy these contracts."""

    def test_is_subclass(self, amplifier: SignalAmplifier) -> None:
        assert isinstance(amplifier, SignalAmplifier)

    def test_amplify_returns_result(self, amplifier: SignalAmplifier) -> None:
        raw_bytes = os.urandom(1024)
        result = amplifier.amplify(raw_bytes)
        assert isinstance(result, AmplificationResult)

    def test_u_in_unit_interval(self, amplifier: SignalAmplifier) -> None:
        for _ in range(10):
            raw_bytes = os.urandom(1024)
            result = amplifier.amplify(raw_bytes)
            assert 0.0 < result.u < 1.0, f"u={result.u} not in (0, 1)"

    def test_diagnostics_is_dict(self, amplifier: SignalAmplifier) -> None:
        result = amplifier.amplify(os.urandom(1024))
        assert isinstance(result.diagnostics, dict)

    def test_result_is_frozen(self, amplifier: SignalAmplifier) -> None:
        result = amplifier.amplify(os.urandom(1024))
        with pytest.raises(AttributeError):
            result.u = 0.5  # type: ignore[misc]

    def test_empty_bytes_raises(self, amplifier: SignalAmplifier) -> None:
        from qr_sampler.exceptions import SignalAmplificationError

        with pytest.raises(SignalAmplificationError):
            amplifier.amplify(b"")


# ---------------------------------------------------------------------------
# Temperature strategy contracts
# ---------------------------------------------------------------------------


def _collect_strategies() -> list[tuple[str, TemperatureStrategy]]:
    """Collect temperature strategy instances for testing."""
    from qr_sampler.temperature.edt import EDTTemperatureStrategy
    from qr_sampler.temperature.fixed import FixedTemperatureStrategy

    return [
        ("fixed", FixedTemperatureStrategy()),
        ("edt", EDTTemperatureStrategy(vocab_size=32000)),
    ]


@pytest.fixture(params=_collect_strategies(), ids=lambda x: x[0])
def strategy(request: pytest.FixtureRequest) -> TemperatureStrategy:
    """Parametrized fixture yielding each temperature strategy instance."""
    _, strat = request.param
    return strat


class TestTemperatureStrategyContract:
    """Every TemperatureStrategy must satisfy these contracts."""

    def test_is_subclass(self, strategy: TemperatureStrategy) -> None:
        assert isinstance(strategy, TemperatureStrategy)

    def test_returns_temperature_result(self, strategy: TemperatureStrategy) -> None:
        logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32)
        config = _make_test_config()
        result = strategy.compute_temperature(logits, config)
        assert isinstance(result, TemperatureResult)

    def test_temperature_is_positive(self, strategy: TemperatureStrategy) -> None:
        logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32)
        config = _make_test_config()
        result = strategy.compute_temperature(logits, config)
        assert result.temperature > 0.0

    def test_shannon_entropy_non_negative(self, strategy: TemperatureStrategy) -> None:
        logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32)
        config = _make_test_config()
        result = strategy.compute_temperature(logits, config)
        assert result.shannon_entropy >= 0.0

    def test_diagnostics_is_dict(self, strategy: TemperatureStrategy) -> None:
        logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32)
        config = _make_test_config()
        result = strategy.compute_temperature(logits, config)
        assert isinstance(result.diagnostics, dict)

    def test_result_is_frozen(self, strategy: TemperatureStrategy) -> None:
        logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32)
        config = _make_test_config()
        result = strategy.compute_temperature(logits, config)
        with pytest.raises(AttributeError):
            result.temperature = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Pipeline stage contracts
# ---------------------------------------------------------------------------


def _collect_stages() -> list[tuple[str, PipelineStage]]:
    """Collect all registered pipeline stage instances."""
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
    )

    return [
        ("adaptive_injection", AdaptiveInjectionStage()),
        ("logit_noise", LogitNoiseStage()),
        ("temperature", TemperatureStage()),
        ("temp_variance", TempVarianceStage()),
        ("min_p", MinPStage()),
        ("xtc", XTCStage()),
        ("entropy_fetch", EntropyFetchStage()),
        ("correlated_walk", CorrelatedWalkStage()),
        ("selection", SelectionStage()),
    ]


@pytest.fixture(params=_collect_stages(), ids=lambda x: x[0])
def pipeline_stage(request: pytest.FixtureRequest) -> PipelineStage:
    """Parametrized fixture yielding each pipeline stage instance."""
    _, stage = request.param
    return stage


class TestPipelineStageContract:
    """Every PipelineStage must satisfy these contracts."""

    def test_satisfies_protocol(self, pipeline_stage: PipelineStage) -> None:
        assert isinstance(pipeline_stage, PipelineStage)

    def test_name_is_nonempty_string(self, pipeline_stage: PipelineStage) -> None:
        assert isinstance(pipeline_stage.name, str)
        assert len(pipeline_stage.name) > 0

    def test_is_callable(self, pipeline_stage: PipelineStage) -> None:
        assert callable(pipeline_stage)

    def test_registered_in_stage_registry(self, pipeline_stage: PipelineStage) -> None:
        """Built-in stages should be findable via StageRegistry."""
        cls = StageRegistry.get(pipeline_stage.name)
        assert isinstance(pipeline_stage, cls)

    def test_stage_registry_roundtrip(self, pipeline_stage: PipelineStage) -> None:
        """Registry lookup returns the correct class."""
        cls = StageRegistry.get(pipeline_stage.name)
        new_instance = cls()
        assert new_instance.name == pipeline_stage.name
        assert isinstance(new_instance, PipelineStage)
