"""LogitNoiseStage — M1: per-logit quantum noise injection."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from qr_sampler.injection.logit_noise import LogitNoise
from qr_sampler.pipeline.registry import StageRegistry

if TYPE_CHECKING:
    from qr_sampler.pipeline.context import SamplingContext


@StageRegistry.register("logit_noise")
class LogitNoiseStage:
    """M1: Direct per-logit quantum noise before temperature scaling.

    Delegates to ``LogitNoise.perturb()`` which fetches ``vocab_size * 4``
    bytes from the entropy source and applies probit-transformed noise.
    No-ops when ``config.logit_noise_alpha == 0``.

    Respects ``ctx.injection_scale``: effective alpha is
    ``config.logit_noise_alpha * injection_scale``.
    """

    name: str = "logit_noise"

    def __call__(self, ctx: SamplingContext) -> None:
        effective_alpha = ctx.config.logit_noise_alpha * ctx.injection_scale
        if effective_alpha <= 0.0:
            return
        t_start = time.perf_counter_ns()
        ctx.row = LogitNoise.perturb(
            ctx.row, ctx.entropy_source, ctx.config, alpha_override=effective_alpha
        )
        t_end = time.perf_counter_ns()
        ctx.entropy_fetch_ms += (t_end - t_start) / 1_000_000.0
