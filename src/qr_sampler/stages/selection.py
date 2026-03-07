"""SelectionStage — CDF-based token selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qr_sampler.pipeline.registry import StageRegistry
from qr_sampler.selection.selector import TokenSelector

if TYPE_CHECKING:
    from qr_sampler.pipeline.context import SamplingContext


@StageRegistry.register("selection")
class SelectionStage:
    """Select a token via CDF lookup using the amplified uniform value.

    Writes ``ctx.token_id``, ``ctx.token_rank``, ``ctx.token_prob``,
    and ``ctx.num_candidates``.
    """

    name: str = "selection"

    def __init__(self) -> None:
        self._selector = TokenSelector()

    def __call__(self, ctx: SamplingContext) -> None:
        selection = self._selector.select(
            ctx.row,
            ctx.temperature,
            ctx.config.top_k,
            ctx.config.top_p,
            ctx.u,
        )
        ctx.token_id = selection.token_id
        ctx.token_rank = selection.token_rank
        ctx.token_prob = selection.token_prob
        ctx.num_candidates = selection.num_candidates
