"""SelectionDriftStage — per-request selection drift."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from qr_sampler.injection.selection_drift import SelectionDrift
from qr_sampler.pipeline.registry import StageRegistry

if TYPE_CHECKING:
    from qr_sampler.pipeline.context import SamplingContext

_DRIFT_POSITION_KEY = "selection_drift.position"


@StageRegistry.register("selection_drift")
class SelectionDriftStage:
    """Selection drift that drifts the selection point across tokens.

    Reads and writes ``ctx.stage_state["selection_drift.position"]``.
    Replaces ``ctx.u`` with the drift position and marks amplifier
    diagnostics as unknown (NaN) since u no longer comes from the amplifier.
    No-ops when ``config.drift_step == 0`` or no persistent state is available.

    Respects ``ctx.injection_scale``: effective step is
    ``config.drift_step * injection_scale``.
    """

    name: str = "selection_drift"

    def __call__(self, ctx: SamplingContext) -> None:
        effective_step = ctx.config.drift_step * ctx.injection_scale
        ctx.effective_step = effective_step
        if effective_step <= 0.0:
            return
        if _DRIFT_POSITION_KEY not in ctx.stage_state:
            return

        drift_position = ctx.stage_state[_DRIFT_POSITION_KEY]

        t_start = time.perf_counter_ns()
        ctx.u, new_position = SelectionDrift.step(
            ctx.u,
            ctx.entropy_source,
            ctx.config,
            drift_position,
            step_override=effective_step,
        )
        t_end = time.perf_counter_ns()
        ctx.entropy_fetch_ms += (t_end - t_start) / 1_000_000.0

        ctx.stage_state[_DRIFT_POSITION_KEY] = new_position

        # Mark amplifier diagnostics as unknown since u was replaced.
        ctx.sample_mean = float("nan")
        ctx.z_score = float("nan")
