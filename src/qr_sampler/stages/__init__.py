"""Built-in pipeline stages for qr-sampler.

Provides the default sampling pipeline and all built-in stage classes.
Each stage is registered via ``@StageRegistry.register()`` and discoverable
via the ``qr_sampler.pipeline_stages`` entry-point group.

Default pipeline order:
     1. ``adaptive_injection`` — scale injection intensity by model uncertainty
     2. ``logit_perturbation``        — per-logit quantum noise (before temperature)
     3. ``dry``                — DRY n-gram repetition penalty
     4. ``top_n_sigma``        — logit-space sigma filtering (pre-softmax)
     5. ``temperature``        — compute temperature via strategy
     6. ``temp_modulation``     — quantum temperature modulation
     7. ``min_p``              — dynamic probability floor filtering
     8. ``tfs``                — tail-free sampling via second derivatives
     9. ``typical``            — locally typical sampling
    10. ``eta``                — entropy-aware probability cutoff
    11. ``xtc``                — exclude top choices using quantum bits
    12. ``entropy_fetch``      — JIT entropy fetch + signal amplification
    13. ``selection_drift``    — per-request selection drift
    14. ``mirostat``           — Mirostat v2 adaptive perplexity control
    15. ``gumbel_selection``   — Gumbel-Max quantum selection
    16. ``selection``          — CDF-based token selection (skipped if mirostat/gumbel active)
"""

from qr_sampler.pipeline.stage import PipelineStage
from qr_sampler.stages.adaptive_injection import AdaptiveInjectionStage
from qr_sampler.stages.dry import DRYPenaltyStage
from qr_sampler.stages.entropy_fetch import EntropyFetchStage
from qr_sampler.stages.eta import EtaSamplingStage
from qr_sampler.stages.gumbel_selection import GumbelSelectionStage
from qr_sampler.stages.logit_perturbation import LogitPerturbationStage
from qr_sampler.stages.min_p import MinPStage
from qr_sampler.stages.mirostat import MirostatStage
from qr_sampler.stages.selection import SelectionStage
from qr_sampler.stages.selection_drift import SelectionDriftStage
from qr_sampler.stages.temp_modulation import TemperatureModulationStage
from qr_sampler.stages.temperature import TemperatureStage
from qr_sampler.stages.tfs import TailFreeSamplingStage
from qr_sampler.stages.top_n_sigma import TopNSigmaStage
from qr_sampler.stages.typical import TypicalSamplingStage
from qr_sampler.stages.xtc import XTCStage


def build_default_pipeline() -> list[PipelineStage]:
    """Build the default sampling pipeline with all built-in stages.

    Returns a fresh list each call so callers can safely mutate it
    (e.g., insert or remove stages for experiments).
    """
    return [
        AdaptiveInjectionStage(),
        LogitPerturbationStage(),
        DRYPenaltyStage(),
        TopNSigmaStage(),
        TemperatureStage(),
        TemperatureModulationStage(),
        MinPStage(),
        TailFreeSamplingStage(),
        TypicalSamplingStage(),
        EtaSamplingStage(),
        XTCStage(),
        EntropyFetchStage(),
        SelectionDriftStage(),
        MirostatStage(),
        GumbelSelectionStage(),
        SelectionStage(),
    ]


__all__ = [
    "AdaptiveInjectionStage",
    "DRYPenaltyStage",
    "EntropyFetchStage",
    "EtaSamplingStage",
    "GumbelSelectionStage",
    "LogitPerturbationStage",
    "MinPStage",
    "MirostatStage",
    "SelectionDriftStage",
    "SelectionStage",
    "TailFreeSamplingStage",
    "TemperatureModulationStage",
    "TemperatureStage",
    "TopNSigmaStage",
    "TypicalSamplingStage",
    "XTCStage",
    "build_default_pipeline",
]
