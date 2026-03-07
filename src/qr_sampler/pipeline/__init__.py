"""Pipeline architecture for composable token sampling.

Provides the ``PipelineStage`` protocol, ``SamplingContext`` state bag,
``StageRegistry`` for entry-point discovery, and ``run_pipeline`` executor.
"""

from qr_sampler.pipeline.context import SamplingContext
from qr_sampler.pipeline.registry import StageRegistry
from qr_sampler.pipeline.stage import PipelineStage

__all__ = ["PipelineStage", "SamplingContext", "StageRegistry"]
