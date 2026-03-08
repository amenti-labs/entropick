"""SGLang adapter for qr-sampler.

Provides a custom logit processor compatible with SGLang's runtime
that replaces standard token sampling with external-entropy-driven
selection via the qr-sampler pipeline.

Usage::

    from qr_sampler.adapters.sglang import QRSamplerCustomLogitProcessor

    processor = QRSamplerCustomLogitProcessor()
    # Pass to SGLang runtime via custom_logit_processor parameter.

SGLang's custom logit processor protocol accepts a torch tensor of logits
and returns a modified torch tensor:
``__call__(self, logits: torch.Tensor) -> torch.Tensor``.
The adapter converts logits to numpy, runs the full pipeline
(entropy fetch, amplification, temperature, selection), forces
one-hot logits, and writes back into the original tensor.

Torch is imported conditionally at call time and the adapter raises
``ImportError`` with a clear message if torch is not installed.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from qr_sampler.adapters._base import AdapterComponents
from qr_sampler.logging.types import TokenSamplingRecord
from qr_sampler.pipeline.context import SamplingContext

if TYPE_CHECKING:
    from qr_sampler.config import QRSamplerConfig
    from qr_sampler.pipeline.stage import PipelineStage

logger = logging.getLogger("qr_sampler")


class QRSamplerCustomLogitProcessor:
    """SGLang custom logit processor using qr-sampler pipeline.

    This class implements the SGLang ``CustomLogitProcessor`` protocol:
    ``__call__(self, logits: torch.Tensor) -> torch.Tensor``.

    Each call processes a single generation step. The processor maintains
    a selection drift position across steps (per-instance state).

    Args:
        config: Base configuration. If ``None``, loads from environment.
        vocab_size: Model vocabulary size. Inferred from first call if 0.
        pipeline: Custom pipeline stages. Uses default if ``None``.
        **overrides: Config field overrides (e.g., ``top_k=100``).

    Example::

        from qr_sampler.adapters.sglang import QRSamplerCustomLogitProcessor
        from qr_sampler.config import QRSamplerConfig

        config = QRSamplerConfig(entropy_source_type="system", top_k=50)
        processor = QRSamplerCustomLogitProcessor(config=config)
    """

    def __init__(
        self,
        config: QRSamplerConfig | None = None,
        vocab_size: int = 0,
        pipeline: list[PipelineStage] | None = None,
        **overrides: Any,
    ) -> None:
        """Initialize the adapter and all subsystems.

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
            vocab_size: Vocabulary size inferred from logits tensor shape.

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
            "QRSamplerCustomLogitProcessor initialized: vocab_size=%d, "
            "entropy_source=%s, pipeline=[%s]",
            effective_vocab,
            self._components.entropy_source.name,
            ", ".join(s.name for s in self._components.pipeline),
        )
        return self._components

    def __call__(self, logits: Any) -> Any:
        """Process logits for one generation step.

        Implements the SGLang ``CustomLogitProcessor`` protocol.
        Processes each row in the batch independently through the
        qr-sampler pipeline and forces one-hot logits.

        Args:
            logits: ``torch.Tensor`` of logit scores. May be 1-D
                (single request) or 2-D ``(batch_size, vocab_size)``.

        Returns:
            Modified logits tensor with one-hot logits (in-place).

        Raises:
            ImportError: If torch is not installed.
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "QRSamplerCustomLogitProcessor requires PyTorch. Install it with: pip install torch"
            ) from exc

        # Infer vocab size from logits shape.
        vocab_size = logits.shape[-1]
        components = self._ensure_initialized(vocab_size)

        is_batched = logits.dim() >= 2
        num_rows = logits.shape[0] if is_batched else 1

        for i in range(num_rows):
            row_tensor = logits[i] if is_batched else logits
            self._process_row(row_tensor, components, torch)

        return logits

    def _process_row(
        self,
        row_tensor: Any,
        components: AdapterComponents,
        torch_module: Any,
    ) -> None:
        """Process a single row through the pipeline and force one-hot.

        Args:
            row_tensor: 1-D tensor of logits for one request.
            components: The initialized adapter components.
            torch_module: The torch module (passed to avoid re-import).
        """
        t_start_ns = time.perf_counter_ns()

        # Convert to numpy (zero-copy if CPU).
        if row_tensor.is_cuda:
            row_np = row_tensor.detach().cpu().numpy()
        else:
            row_np = row_tensor.detach().numpy()

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

        # Force one-hot on the original tensor.
        if ctx.token_id < 0:
            logger.error("Pipeline produced no token selection; skipping one-hot forcing")
            return
        row_tensor.fill_(float("-inf"))
        row_tensor[ctx.token_id] = 0.0

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
        """Release all resources held by the processor."""
        if self._components is not None:
            self._components.close()
