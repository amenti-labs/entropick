"""LogitPerturbationStage — per-logit quantum noise injection."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from qr_sampler.injection.logit_perturbation import LogitPerturbation
from qr_sampler.pipeline.registry import StageRegistry

if TYPE_CHECKING:
    from qr_sampler.pipeline.context import SamplingContext


@StageRegistry.register("logit_perturbation")
class LogitPerturbationStage:
    """Direct per-logit quantum noise before temperature scaling.

    Delegates to ``LogitPerturbation.perturb()`` which fetches ``vocab_size * 4``
    bytes from the entropy source and applies probit-transformed noise.
    No-ops when ``config.logit_perturbation_alpha == 0``.

    Respects ``ctx.injection_scale``: effective alpha is
    ``config.logit_perturbation_alpha * injection_scale``.
    """

    name: str = "logit_perturbation"

    def __call__(self, ctx: SamplingContext) -> None:
        effective_alpha = ctx.config.logit_perturbation_alpha * ctx.injection_scale
        ctx.effective_alpha = effective_alpha
        if effective_alpha <= 0.0:
            return
        t_start = time.perf_counter_ns()
        ctx.row = LogitPerturbation.perturb(
            ctx.row, ctx.entropy_source, ctx.config, alpha_override=effective_alpha
        )
        t_end = time.perf_counter_ns()
        ctx.entropy_fetch_ms += (t_end - t_start) / 1_000_000.0
