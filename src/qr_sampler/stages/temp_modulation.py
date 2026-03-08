"""TemperatureModulationStage — quantum temperature modulation."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from qr_sampler.injection.temp_modulation import TemperatureModulation
from qr_sampler.pipeline.registry import StageRegistry

if TYPE_CHECKING:
    from qr_sampler.pipeline.context import SamplingContext


@StageRegistry.register("temp_modulation")
class TemperatureModulationStage:
    """Modulate temperature using quantum entropy.

    Delegates to ``TemperatureModulation.modulate()`` which fetches entropy bytes,
    converts to a uniform value, and scales temperature.
    No-ops when ``config.temp_modulation_beta == 0``.

    Respects ``ctx.injection_scale``: effective beta is
    ``config.temp_modulation_beta * injection_scale``.
    """

    name: str = "temp_modulation"

    def __call__(self, ctx: SamplingContext) -> None:
        effective_beta = ctx.config.temp_modulation_beta * ctx.injection_scale
        ctx.effective_beta = effective_beta
        if effective_beta <= 0.0:
            return
        t_start = time.perf_counter_ns()
        ctx.temperature = TemperatureModulation.modulate(
            ctx.temperature,
            ctx.entropy_source,
            ctx.config,
            beta_override=effective_beta,
        )
        t_end = time.perf_counter_ns()
        ctx.entropy_fetch_ms += (t_end - t_start) / 1_000_000.0
