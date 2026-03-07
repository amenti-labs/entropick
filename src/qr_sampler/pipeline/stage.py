"""Pipeline stage protocol — the uniform interface for all sampling steps."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from qr_sampler.pipeline.context import SamplingContext


@runtime_checkable
class PipelineStage(Protocol):
    """A single step in the token sampling pipeline.

    Stages are stateless with respect to per-request data — all mutable
    state flows through ``SamplingContext``. Stages may hold internal
    caches or tool instances (e.g., a ``TokenSelector``) as long as
    they carry no per-request state.

    To create a new stage:
        1. Implement this protocol (``name`` attribute + ``__call__``).
        2. Register via ``@StageRegistry.register("my_stage")``.
        3. Add an entry point in ``pyproject.toml`` under
           ``[project.entry-points."qr_sampler.pipeline_stages"]``.
    """

    name: str
    """Unique identifier for this stage."""

    def __call__(self, ctx: SamplingContext) -> None:
        """Execute this stage, reading from and writing to ``ctx``.

        Args:
            ctx: Mutable sampling context for one token.
        """
        ...
