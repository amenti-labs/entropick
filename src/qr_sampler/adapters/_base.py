"""Shared component construction and base class for framework adapters.

All adapters need the same set of components: config, entropy source,
amplifier, temperature strategy, pipeline, and logger. This module
provides a builder that constructs them from a ``QRSamplerConfig``,
plus a base class that eliminates boilerplate across adapters.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import TYPE_CHECKING, Any

from qr_sampler.amplification.registry import AmplifierRegistry
from qr_sampler.config import QRSamplerConfig
from qr_sampler.entropy.fallback import FallbackEntropySource
from qr_sampler.entropy.registry import EntropySourceRegistry
from qr_sampler.logging.logger import SamplingLogger
from qr_sampler.logging.types import TokenSamplingRecord
from qr_sampler.pipeline.context import SamplingContext
from qr_sampler.stages import build_default_pipeline
from qr_sampler.temperature.registry import TemperatureStrategyRegistry

if TYPE_CHECKING:
    from qr_sampler.entropy.base import EntropySource
    from qr_sampler.pipeline.stage import PipelineStage

logger = logging.getLogger("qr_sampler")

# Default vocabulary size when no model context provides one.
_DEFAULT_VOCAB_SIZE = 32000


def _config_hash(config: QRSamplerConfig) -> str:
    """Compute a short hash of the config for logging.

    Args:
        config: The sampler configuration to hash.

    Returns:
        First 16 hex characters of the SHA-256 digest of the config dump.
    """
    raw = config.model_dump_json().encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _accepts_config(cls: type) -> bool:
    """Check if a class constructor accepts a QRSamplerConfig as first arg.

    Args:
        cls: The class to inspect.

    Returns:
        True if the constructor expects a config argument.
    """
    import inspect

    try:
        sig = inspect.signature(cls)
    except (ValueError, TypeError):
        return False

    params = list(sig.parameters.values())
    for param in params:
        annotation = param.annotation
        if annotation is inspect.Parameter.empty:
            if param.name == "config":
                return True
        elif annotation is QRSamplerConfig or (
            isinstance(annotation, str) and "QRSamplerConfig" in annotation
        ):
            return True
        # Only check the first non-self parameter.
        break
    return False


def _build_entropy_source(config: QRSamplerConfig) -> EntropySource:
    """Build the entropy source from config, wrapping with fallback if needed.

    Args:
        config: Sampler configuration specifying source type and fallback mode.

    Returns:
        An EntropySource, potentially wrapped in FallbackEntropySource.
    """
    source_cls = EntropySourceRegistry.get(config.entropy_source_type)

    if _accepts_config(source_cls):
        primary: EntropySource = source_cls(config)  # type: ignore[call-arg]
    else:
        primary = source_cls()

    if config.fallback_mode == "error":
        return primary

    if config.fallback_mode == "system":
        from qr_sampler.entropy.system import SystemEntropySource

        fallback: EntropySource = SystemEntropySource()
    elif config.fallback_mode == "mock_uniform":
        from qr_sampler.entropy.mock import MockUniformSource

        fallback = MockUniformSource()
    else:
        logger.warning(
            "Unknown fallback_mode %r, using system fallback",
            config.fallback_mode,
        )
        from qr_sampler.entropy.system import SystemEntropySource

        fallback = SystemEntropySource()

    return FallbackEntropySource(primary, fallback)


class AdapterComponents:
    """Shared components constructed from a QRSamplerConfig.

    Adapters should instantiate this once and reuse it for all calls.
    This avoids duplicating the component construction logic across
    adapters.

    Attributes:
        config: The resolved sampler configuration.
        entropy_source: The entropy source (possibly wrapped in fallback).
        amplifier: The signal amplifier.
        temperature_strategy: The temperature strategy.
        pipeline: The ordered list of pipeline stages.
        sampling_logger: The diagnostic logger.
        config_hash: Short hash of the config for logging.
        vocab_size: Vocabulary size (used for temperature strategies that need it).
    """

    __slots__ = (
        "amplifier",
        "config",
        "config_hash",
        "entropy_source",
        "pipeline",
        "sampling_logger",
        "temperature_strategy",
        "vocab_size",
    )

    def __init__(
        self,
        config: QRSamplerConfig | None = None,
        vocab_size: int = _DEFAULT_VOCAB_SIZE,
        pipeline: list[PipelineStage] | None = None,
        **overrides: Any,
    ) -> None:
        """Build all components from config.

        Args:
            config: Base configuration. If ``None``, loads from environment.
            vocab_size: Model vocabulary size (for EDT temperature strategy).
            pipeline: Custom pipeline stages. Uses default if ``None``.
            **overrides: Fields to override on the config (e.g., ``top_k=100``).
        """
        if config is not None and overrides:
            merged = config.model_dump()
            merged.update(overrides)
            self.config = QRSamplerConfig.model_validate(merged)
        elif config is not None:
            self.config = config
        elif overrides:
            self.config = QRSamplerConfig.model_validate(overrides)
        else:
            self.config = QRSamplerConfig()

        self.vocab_size = vocab_size
        self.entropy_source = _build_entropy_source(self.config)
        self.amplifier = AmplifierRegistry.build(self.config)
        if hasattr(self.amplifier, "calibrate"):
            self.amplifier.calibrate(self.entropy_source, self.config)
        self.temperature_strategy = TemperatureStrategyRegistry.build(self.config, self.vocab_size)
        self.sampling_logger = SamplingLogger(self.config)
        self.config_hash = _config_hash(self.config)
        self.pipeline = pipeline if pipeline is not None else build_default_pipeline()

    def close(self) -> None:
        """Release all resources held by the components."""
        self.entropy_source.close()


def _init_stage_state(config: QRSamplerConfig) -> dict[str, Any]:
    """Build the initial stage_state dict for a new adapter session.

    Args:
        config: Resolved configuration providing initial values.

    Returns:
        Stage state dict with default values for all stateful stages.
    """
    return {
        "selection_drift.position": config.drift_initial_position,
        "mirostat.mu": 2.0 * config.mirostat_tau,
        "token_history": [],
    }


def _run_pipeline_and_log(
    ctx: SamplingContext,
    components: AdapterComponents,
    t_start_ns: int,
) -> None:
    """Run pipeline stages, append token history, and log the sampling record.

    This is the shared core logic used by all framework adapters.

    Args:
        ctx: The sampling context (modified in-place by stages).
        components: Adapter components providing pipeline and logger.
        t_start_ns: Timestamp from ``time.perf_counter_ns()`` at row start.
    """
    for stage in components.pipeline:
        stage(ctx)

    if ctx.token_id >= 0:
        ctx.stage_state.setdefault("token_history", []).append(ctx.token_id)

    t_end_ns = time.perf_counter_ns()
    total_sampling_ms = (t_end_ns - t_start_ns) / 1_000_000.0

    record = TokenSamplingRecord(
        timestamp_ns=t_start_ns,
        entropy_fetch_ms=ctx.entropy_fetch_ms,
        total_sampling_ms=total_sampling_ms,
        entropy_source_used=ctx.entropy_source_name,
        entropy_is_fallback=ctx.entropy_is_fallback,
        sample_mean=ctx.sample_mean,
        z_score=ctx.z_score,
        u_value=ctx.u,
        temperature_strategy=components.config.temperature_strategy,
        shannon_entropy=ctx.shannon_entropy,
        temperature_used=ctx.temperature,
        token_id=ctx.token_id,
        token_rank=ctx.token_rank,
        token_prob=ctx.token_prob,
        num_candidates=ctx.num_candidates,
        config_hash=components.config_hash,
        injection_alpha=ctx.effective_alpha,
        injection_beta=ctx.effective_beta,
        injection_step=ctx.effective_step,
        injection_scale=ctx.injection_scale,
    )
    components.sampling_logger.log_token(record)


class _AdapterBase:
    """Shared base for all framework adapters.

    Provides lazy initialization, stage state management, config/logger
    properties, and resource cleanup. Subclasses only need to implement
    ``__call__`` for their framework-specific I/O protocol.
    """

    def __init__(
        self,
        config: QRSamplerConfig | None = None,
        vocab_size: int = 0,
        pipeline: list[PipelineStage] | None = None,
        **overrides: Any,
    ) -> None:
        self._requested_vocab_size = vocab_size
        self._components: AdapterComponents | None = None
        self._config = config
        self._pipeline = pipeline
        self._overrides = overrides
        self._stage_state: dict[str, Any] = {}

    def _ensure_initialized(self, vocab_size: int) -> AdapterComponents:
        """Lazily initialize components on first call when vocab_size is known.

        Args:
            vocab_size: Vocabulary size inferred from input tensor/list.

        Returns:
            The initialized AdapterComponents.
        """
        if self._components is not None:
            return self._components

        effective_vocab = self._requested_vocab_size or vocab_size
        self._components = AdapterComponents(
            config=self._config,
            vocab_size=effective_vocab,
            pipeline=self._pipeline,
            **self._overrides,
        )

        self._stage_state = _init_stage_state(self._components.config)

        logger.info(
            "%s initialized: vocab_size=%d, entropy_source=%s, pipeline=[%s]",
            type(self).__name__,
            effective_vocab,
            self._components.entropy_source.name,
            ", ".join(s.name for s in self._components.pipeline),
        )
        return self._components

    def _build_context(self, row_np: Any, components: AdapterComponents) -> SamplingContext:
        """Build a SamplingContext from a numpy row.

        Args:
            row_np: 1-D numpy array of logits.
            components: The initialized adapter components.

        Returns:
            A new SamplingContext ready for the pipeline.
        """
        return SamplingContext(
            row=row_np,
            config=components.config,
            entropy_source=components.entropy_source,
            amplifier=components.amplifier,
            temperature_strategy=components.temperature_strategy,
            config_hash=components.config_hash,
            stage_state=self._stage_state,
        )

    @property
    def config(self) -> QRSamplerConfig | None:
        """The active configuration, or None if not yet initialized."""
        if self._components is not None:
            return self._components.config
        return self._config

    @property
    def sampling_logger(self) -> Any:
        """The diagnostic logger, or None if not yet initialized."""
        if self._components is not None:
            return self._components.sampling_logger
        return None

    def close(self) -> None:
        """Release all resources held by the adapter."""
        if self._components is not None:
            self._components.close()
