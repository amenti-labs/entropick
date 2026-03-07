"""TemperatureStage — compute temperature via the configured strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qr_sampler.pipeline.registry import StageRegistry

if TYPE_CHECKING:
    from qr_sampler.pipeline.context import SamplingContext


@StageRegistry.register("temperature")
class TemperatureStage:
    """Compute temperature using the per-request temperature strategy.

    Writes ``ctx.temperature`` and ``ctx.shannon_entropy``.
    Always runs (temperature is required for selection).
    """

    name: str = "temperature"

    def __call__(self, ctx: SamplingContext) -> None:
        result = ctx.temperature_strategy.compute_temperature(ctx.row, ctx.config)
        ctx.temperature = result.temperature
        ctx.shannon_entropy = result.shannon_entropy
