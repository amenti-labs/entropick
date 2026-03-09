"""SGLang adapter for entropick.

Provides a custom logit processor compatible with SGLang's runtime
that replaces standard token sampling with external-entropy-driven
selection via the entropick pipeline.

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
from typing import Any

from qr_sampler.adapters._base import _AdapterBase, _run_pipeline_and_log

logger = logging.getLogger("qr_sampler")


class QRSamplerCustomLogitProcessor(_AdapterBase):
    """SGLang custom logit processor using entropick pipeline.

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

    def __call__(self, logits: Any) -> Any:
        """Process logits for one generation step.

        Implements the SGLang ``CustomLogitProcessor`` protocol.
        Processes each row in the batch independently through the
        entropick pipeline and forces one-hot logits.

        Args:
            logits: ``torch.Tensor`` of logit scores. May be 1-D
                (single request) or 2-D ``(batch_size, vocab_size)``.

        Returns:
            Modified logits tensor with one-hot logits (in-place).

        Raises:
            ImportError: If torch is not installed.
        """
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "QRSamplerCustomLogitProcessor requires PyTorch. Install it with: pip install torch"
            ) from exc

        vocab_size = logits.shape[-1]
        components = self._ensure_initialized(vocab_size)

        is_batched = logits.dim() >= 2
        num_rows = logits.shape[0] if is_batched else 1

        for i in range(num_rows):
            row_tensor = logits[i] if is_batched else logits
            self._process_row(row_tensor, components)

        return logits

    def _process_row(self, row_tensor: Any, components: Any) -> None:
        """Process a single row through the pipeline and force one-hot."""
        t_start_ns = time.perf_counter_ns()

        # Convert to numpy (zero-copy if CPU).
        if row_tensor.is_cuda:
            row_np = row_tensor.detach().cpu().numpy()
        else:
            row_np = row_tensor.detach().numpy()

        ctx = self._build_context(row_np, components)
        _run_pipeline_and_log(ctx, components, t_start_ns)
        self._stage_state = ctx.stage_state

        # Force one-hot on the original tensor.
        if ctx.token_id < 0:
            logger.error("Pipeline produced no token selection; skipping one-hot forcing")
            return
        row_tensor.fill_(float("-inf"))
        row_tensor[ctx.token_id] = 0.0
