"""CorrelatedWalkStage — M3: per-request correlated walk."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from qr_sampler.injection.correlated_walk import CorrelatedWalk
from qr_sampler.pipeline.registry import StageRegistry

if TYPE_CHECKING:
    from qr_sampler.pipeline.context import SamplingContext

_WALK_POSITION_KEY = "correlated_walk.position"


@StageRegistry.register("correlated_walk")
class CorrelatedWalkStage:
    """M3: Correlated walk that drifts the selection point across tokens.

    Reads and writes ``ctx.stage_state["correlated_walk.position"]``.
    Replaces ``ctx.u`` with the walk position and marks amplifier
    diagnostics as unknown (NaN) since u no longer comes from the amplifier.
    No-ops when ``config.walk_step == 0`` or no persistent state is available.

    Respects ``ctx.injection_scale``: effective step is
    ``config.walk_step * injection_scale``.
    """

    name: str = "correlated_walk"

    def __call__(self, ctx: SamplingContext) -> None:
        effective_step = ctx.config.walk_step * ctx.injection_scale
        if effective_step <= 0.0:
            return
        if _WALK_POSITION_KEY not in ctx.stage_state:
            return

        walk_position = ctx.stage_state[_WALK_POSITION_KEY]

        t_start = time.perf_counter_ns()
        ctx.u, new_position = CorrelatedWalk.step(
            ctx.u, ctx.entropy_source, ctx.config, walk_position,
            step_override=effective_step,
        )
        t_end = time.perf_counter_ns()
        ctx.entropy_fetch_ms += (t_end - t_start) / 1_000_000.0

        ctx.stage_state[_WALK_POSITION_KEY] = new_position

        # Mark amplifier diagnostics as unknown since u was replaced.
        ctx.sample_mean = float("nan")
        ctx.z_score = float("nan")
