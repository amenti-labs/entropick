"""Hugging Face Transformers adapter for entropick.

Provides a ``LogitsProcessor`` compatible with ``model.generate()``
that replaces standard token sampling with external-entropy-driven
selection via the entropick pipeline.

Usage::

    from qr_sampler.adapters.transformers import QRSamplerLogitsProcessorHF

    processor = QRSamplerLogitsProcessorHF()
    outputs = model.generate(input_ids, logits_processor=[processor])

The adapter handles torch<->numpy conversion, runs the full pipeline
(entropy fetch, amplification, temperature, selection), and forces
one-hot logits so the downstream sampler picks the selected token.

Torch is imported conditionally and the adapter degrades gracefully
if torch is not installed (raises ``ImportError`` at call time with
a clear message).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from qr_sampler.adapters._base import _AdapterBase, _run_pipeline_and_log

logger = logging.getLogger("qr_sampler")


class QRSamplerLogitsProcessorHF(_AdapterBase):
    """Hugging Face Transformers LogitsProcessor using entropick pipeline.

    This class implements the Transformers ``LogitsProcessor`` protocol:
    ``__call__(input_ids: torch.LongTensor, scores: torch.FloatTensor)
    -> torch.FloatTensor``.

    Each call processes a single generation step. The processor maintains
    a selection drift position across steps (per-instance state).

    Args:
        config: Base configuration. If ``None``, loads from environment.
        vocab_size: Model vocabulary size. Inferred from first call if 0.
        pipeline: Custom pipeline stages. Uses default if ``None``.
        **overrides: Config field overrides (e.g., ``top_k=100``).

    Example::

        from qr_sampler.adapters.transformers import QRSamplerLogitsProcessorHF
        from qr_sampler.config import QRSamplerConfig

        config = QRSamplerConfig(entropy_source_type="system", top_k=50)
        processor = QRSamplerLogitsProcessorHF(config=config)
        outputs = model.generate(
            input_ids,
            logits_processor=[processor],
            do_sample=True,
        )
    """

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        """Process logits for one generation step.

        Implements the Transformers ``LogitsProcessor`` protocol.
        Processes each row in the batch independently through the
        entropick pipeline and forces one-hot logits.

        Args:
            input_ids: ``torch.LongTensor`` of shape ``(batch_size, seq_len)``.
                Token IDs generated so far. Not used by this processor.
            scores: ``torch.FloatTensor`` of shape ``(batch_size, vocab_size)``.
                Logit scores for the current generation step.

        Returns:
            Modified scores tensor with one-hot logits (in-place).

        Raises:
            ImportError: If torch is not installed.
        """
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "QRSamplerLogitsProcessorHF requires PyTorch. Install it with: pip install torch"
            ) from exc

        vocab_size = scores.shape[-1]
        components = self._ensure_initialized(vocab_size)

        is_batched = scores.dim() >= 2
        num_rows = scores.shape[0] if is_batched else 1

        for i in range(num_rows):
            row_tensor = scores[i] if is_batched else scores
            self._process_row(row_tensor, components)

        return scores

    def _process_row(self, row_tensor: Any, components: Any) -> None:
        """Process a single row through the pipeline and force one-hot."""
        t_start_ns = time.perf_counter_ns()

        # .cpu() moves GPU tensors (CUDA/MPS) to host memory; no-op on CPU.
        row_np = row_tensor.detach().cpu().numpy()

        ctx = self._build_context(row_np, components)
        _run_pipeline_and_log(ctx, components, t_start_ns)
        self._stage_state = ctx.stage_state

        # Force one-hot on the original tensor.
        if ctx.token_id < 0:
            logger.error("Pipeline produced no token selection; skipping one-hot forcing")
            return
        row_tensor.fill_(float("-inf"))
        row_tensor[ctx.token_id] = 0.0
