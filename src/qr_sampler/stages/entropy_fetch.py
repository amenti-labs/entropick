"""EntropyFetchStage — JIT entropy fetch and signal amplification."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from qr_sampler.entropy.fallback import FallbackEntropySource
from qr_sampler.pipeline.registry import StageRegistry

if TYPE_CHECKING:
    from qr_sampler.pipeline.context import SamplingContext


@StageRegistry.register("entropy_fetch")
class EntropyFetchStage:
    """Fetch entropy bytes just-in-time and amplify to a uniform float.

    Writes ``ctx.u``, ``ctx.sample_mean``, ``ctx.z_score``,
    ``ctx.entropy_source_name``, ``ctx.entropy_is_fallback``,
    and accumulates ``ctx.entropy_fetch_ms``.
    """

    name: str = "entropy_fetch"

    def __call__(self, ctx: SamplingContext) -> None:
        t_start = time.perf_counter_ns()

        ctx.entropy_source_name = ctx.entropy_source.name
        ctx.entropy_is_fallback = False

        raw_bytes = ctx.entropy_source.get_random_bytes(ctx.config.sample_count)

        # Detect if fallback was used.
        if isinstance(ctx.entropy_source, FallbackEntropySource):
            ctx.entropy_source_name = ctx.entropy_source.last_source_used
            ctx.entropy_is_fallback = (
                ctx.entropy_source.last_source_used
                != ctx.entropy_source._primary.name
            )

        t_end = time.perf_counter_ns()
        ctx.entropy_fetch_ms += (t_end - t_start) / 1_000_000.0

        # Amplify to uniform float.
        amp_result = ctx.amplifier.amplify(raw_bytes)
        ctx.u = amp_result.u
        ctx.sample_mean = amp_result.diagnostics.get("sample_mean", 0.0)
        ctx.z_score = amp_result.diagnostics.get("z_score", 0.0)
