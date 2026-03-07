"""Built-in pipeline stages for qr-sampler.

Provides the default sampling pipeline and all built-in stage classes.
Each stage is registered via ``@StageRegistry.register()`` and discoverable
via the ``qr_sampler.pipeline_stages`` entry-point group.

Default pipeline order:
    1. ``adaptive_injection`` — scale injection intensity by model uncertainty
    2. ``logit_noise``        — M1: per-logit quantum noise (before temperature)
    3. ``temperature``        — compute temperature via strategy
    4. ``temp_variance``      — M2: quantum temperature modulation
    5. ``min_p``              — dynamic probability floor filtering
    6. ``xtc``                — exclude top choices using quantum bits
    7. ``entropy_fetch``      — JIT entropy fetch + signal amplification
    8. ``correlated_walk``    — M3: per-request correlated walk
    9. ``selection``          — CDF-based token selection
"""

from qr_sampler.pipeline.stage import PipelineStage
from qr_sampler.stages.adaptive_injection import AdaptiveInjectionStage
from qr_sampler.stages.correlated_walk import CorrelatedWalkStage
from qr_sampler.stages.entropy_fetch import EntropyFetchStage
from qr_sampler.stages.logit_noise import LogitNoiseStage
from qr_sampler.stages.min_p import MinPStage
from qr_sampler.stages.selection import SelectionStage
from qr_sampler.stages.temp_variance import TempVarianceStage
from qr_sampler.stages.temperature import TemperatureStage
from qr_sampler.stages.xtc import XTCStage


def build_default_pipeline() -> list[PipelineStage]:
    """Build the default sampling pipeline with all built-in stages.

    Returns a fresh list each call so callers can safely mutate it
    (e.g., insert or remove stages for experiments).
    """
    return [
        AdaptiveInjectionStage(),
        LogitNoiseStage(),
        TemperatureStage(),
        TempVarianceStage(),
        MinPStage(),
        XTCStage(),
        EntropyFetchStage(),
        CorrelatedWalkStage(),
        SelectionStage(),
    ]


__all__ = [
    "AdaptiveInjectionStage",
    "CorrelatedWalkStage",
    "EntropyFetchStage",
    "LogitNoiseStage",
    "MinPStage",
    "SelectionStage",
    "TempVarianceStage",
    "TemperatureStage",
    "XTCStage",
    "build_default_pipeline",
]
