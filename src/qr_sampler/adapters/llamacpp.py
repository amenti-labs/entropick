"""llama-cpp-python adapter for qr-sampler.

Provides a callable compatible with llama-cpp-python's ``LogitsProcessorList``
that replaces standard token sampling with external-entropy-driven selection
via the qr-sampler pipeline.

Usage::

    from llama_cpp import Llama, LogitsProcessorList
    from qr_sampler.adapters.llamacpp import QRSamplerCallback

    llm = Llama(model_path="model.gguf")
    callback = QRSamplerCallback()
    output = llm.create_completion(
        "Once upon a time",
        logits_processor=LogitsProcessorList([callback]),
    )

llama-cpp-python passes flat Python lists to logits processors:
``(input_ids: list[int], scores: list[float]) -> list[float]``.
The adapter converts scores to numpy, runs the full pipeline
(entropy fetch, amplification, temperature, selection), forces
one-hot logits, and converts back to a list.

No lazy import of llama_cpp is needed since the adapter only works
with plain lists and numpy arrays.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from qr_sampler.adapters._base import AdapterComponents
from qr_sampler.logging.types import TokenSamplingRecord
from qr_sampler.pipeline.context import SamplingContext

if TYPE_CHECKING:
    from qr_sampler.config import QRSamplerConfig
    from qr_sampler.pipeline.stage import PipelineStage

logger = logging.getLogger("qr_sampler")


class QRSamplerCallback:
    """llama-cpp-python logits processor using qr-sampler pipeline.

    This class implements the llama-cpp-python ``LogitsProcessor`` protocol:
    ``__call__(input_ids: list[int], scores: list[float]) -> list[float]``.

    Each call processes a single generation step. The callback maintains
    a selection drift position across steps (per-instance state).

    Args:
        config: Base configuration. If ``None``, loads from environment.
        vocab_size: Model vocabulary size. Inferred from first call if 0.
        pipeline: Custom pipeline stages. Uses default if ``None``.
        **overrides: Config field overrides (e.g., ``top_k=100``).

    Example::

        from qr_sampler.adapters.llamacpp import QRSamplerCallback
        from qr_sampler.config import QRSamplerConfig

        config = QRSamplerConfig(entropy_source_type="system", top_k=50)
        callback = QRSamplerCallback(config=config)
        output = llm.create_completion(
            "Once upon a time",
            logits_processor=LogitsProcessorList([callback]),
        )
    """

    def __init__(
        self,
        config: QRSamplerConfig | None = None,
        vocab_size: int = 0,
        pipeline: list[PipelineStage] | None = None,
        **overrides: Any,
    ) -> None:
        """Initialize the callback and all subsystems.

        Args:
            config: Base configuration. If ``None``, loads from environment.
            vocab_size: Model vocabulary size. If 0, inferred from first call.
            pipeline: Custom pipeline stages. Uses default if ``None``.
            **overrides: Config field overrides (e.g., ``top_k=100``).
        """
        self._requested_vocab_size = vocab_size
        self._components: AdapterComponents | None = None
        self._config = config
        self._pipeline = pipeline
        self._overrides = overrides
        self._stage_state: dict[str, Any] = {}

    def _ensure_initialized(self, vocab_size: int) -> AdapterComponents:
        """Lazily initialize components on first call when vocab_size is known.

        Args:
            vocab_size: Vocabulary size inferred from scores list length.

        Returns:
            The initialized AdapterComponents.
        """
        if self._components is not None:
            return self._components

        effective_vocab = self._requested_vocab_size or vocab_size
        self._components = AdapterComponents(
            config=self._config,
            vocab_size=effective_vocab,
            pipeline=self._pipeline,
            **self._overrides,
        )

        # Initialize persistent stage state.
        cfg = self._components.config
        self._stage_state = {
            "selection_drift.position": cfg.drift_initial_position,
            "mirostat.mu": 2.0 * cfg.mirostat_tau,
            "token_history": [],
        }

        logger.info(
            "QRSamplerCallback initialized: vocab_size=%d, entropy_source=%s, pipeline=[%s]",
            effective_vocab,
            self._components.entropy_source.name,
            ", ".join(s.name for s in self._components.pipeline),
        )
        return self._components

    def __call__(self, input_ids: list[int], scores: list[float]) -> list[float]:
        """Process logits for one generation step.

        Implements the llama-cpp-python ``LogitsProcessor`` protocol.
        Converts the flat scores list to numpy, runs the qr-sampler
        pipeline, forces one-hot logits, and returns a new list.

        Args:
            input_ids: Token IDs generated so far. Not used by this processor.
            scores: Logit scores for the current generation step (flat list).

        Returns:
            Modified scores list with one-hot logits (``-inf`` everywhere
            except ``0.0`` at the selected token index).
        """
        vocab_size = len(scores)
        components = self._ensure_initialized(vocab_size)

        t_start_ns = time.perf_counter_ns()

        # Convert flat list to numpy array.
        row_np = np.array(scores, dtype=np.float64)

        # Build context.
        ctx = SamplingContext(
            row=row_np,
            config=components.config,
            entropy_source=components.entropy_source,
            amplifier=components.amplifier,
            temperature_strategy=components.temperature_strategy,
            config_hash=components.config_hash,
            stage_state=self._stage_state,
        )

        # Run pipeline.
        for stage in components.pipeline:
            stage(ctx)

        # Append selected token to history (used by DRY penalty).
        if ctx.token_id >= 0:
            ctx.stage_state.setdefault("token_history", []).append(ctx.token_id)

        # Persist stage state.
        self._stage_state = ctx.stage_state

        # Build one-hot output list.
        if ctx.token_id < 0:
            logger.error("Pipeline produced no token selection; returning original scores")
            return scores
        result = [float("-inf")] * vocab_size
        result[ctx.token_id] = 0.0

        # Log sampling record.
        t_end_ns = time.perf_counter_ns()
        total_sampling_ms = (t_end_ns - t_start_ns) / 1_000_000.0

        record = TokenSamplingRecord(
            timestamp_ns=t_start_ns,
            entropy_fetch_ms=ctx.entropy_fetch_ms,
            total_sampling_ms=total_sampling_ms,
            entropy_source_used=ctx.entropy_source_name,
            entropy_is_fallback=ctx.entropy_is_fallback,
            sample_mean=ctx.sample_mean,
            z_score=ctx.z_score,
            u_value=ctx.u,
            temperature_strategy=components.config.temperature_strategy,
            shannon_entropy=ctx.shannon_entropy,
            temperature_used=ctx.temperature,
            token_id=ctx.token_id,
            token_rank=ctx.token_rank,
            token_prob=ctx.token_prob,
            num_candidates=ctx.num_candidates,
            config_hash=components.config_hash,
            injection_alpha=ctx.effective_alpha,
            injection_beta=ctx.effective_beta,
            injection_step=ctx.effective_step,
            injection_scale=ctx.injection_scale,
        )
        components.sampling_logger.log_token(record)

        return result

    @property
    def config(self) -> QRSamplerConfig | None:
        """The active configuration, or None if not yet initialized."""
        if self._components is not None:
            return self._components.config
        return self._config

    @property
    def sampling_logger(self) -> Any:
        """The diagnostic logger, or None if not yet initialized."""
        if self._components is not None:
            return self._components.sampling_logger
        return None

    def close(self) -> None:
        """Release all resources held by the callback."""
        if self._components is not None:
            self._components.close()
