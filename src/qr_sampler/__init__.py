"""entropick: Plug any physical randomness source into LLM token sampling.

Replaces standard pseudorandom token sampling with external-entropy-driven
selection. Supports quantum random number generators, OpenEntropy hardware
noise, processor timing jitter, and any user-supplied entropy source via gRPC.
"""

from __future__ import annotations

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("entropick")
except PackageNotFoundError:
    __version__ = "0.0.0"

from qr_sampler.config import QRSamplerConfig, resolve_config, validate_extra_args
from qr_sampler.exceptions import (
    ConfigValidationError,
    EntropyUnavailableError,
    QRSamplerError,
    SignalAmplificationError,
    TokenSelectionError,
)
from qr_sampler.pipeline.context import SamplingContext
from qr_sampler.pipeline.registry import StageRegistry
from qr_sampler.pipeline.stage import PipelineStage
from qr_sampler.processor import QRSamplerLogitsProcessor
from qr_sampler.stages import build_default_pipeline

__all__ = [
    "ConfigValidationError",
    "EntropyUnavailableError",
    "PipelineStage",
    "QRSamplerConfig",
    "QRSamplerError",
    "QRSamplerLogitsProcessor",
    "SamplingContext",
    "SignalAmplificationError",
    "StageRegistry",
    "TokenSelectionError",
    "__version__",
    "build_default_pipeline",
    "resolve_config",
    "validate_extra_args",
]
