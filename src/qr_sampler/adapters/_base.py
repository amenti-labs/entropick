"""Shared component construction for framework adapters.

All adapters need the same set of components: config, entropy source,
amplifier, temperature strategy, pipeline, and logger. This module
provides a builder that constructs them from a ``QRSamplerConfig``.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING, Any

from qr_sampler.amplification.registry import AmplifierRegistry
from qr_sampler.config import QRSamplerConfig
from qr_sampler.entropy.fallback import FallbackEntropySource
from qr_sampler.entropy.registry import EntropySourceRegistry
from qr_sampler.logging.logger import SamplingLogger
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
