"""llama-cpp-python adapter for entropick.

Provides a callable compatible with llama-cpp-python's ``LogitsProcessorList``
that replaces standard token sampling with external-entropy-driven selection
via the entropick pipeline.

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

import numpy as np

from qr_sampler.adapters._base import _AdapterBase, _run_pipeline_and_log

logger = logging.getLogger("qr_sampler")


class QRSamplerCallback(_AdapterBase):
    """llama-cpp-python logits processor using entropick pipeline.

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

    def __call__(self, input_ids: list[int], scores: list[float]) -> list[float]:
        """Process logits for one generation step.

        Implements the llama-cpp-python ``LogitsProcessor`` protocol.
        Converts the flat scores list to numpy, runs the entropick
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
        row_np = np.array(scores, dtype=np.float64)

        ctx = self._build_context(row_np, components)
        _run_pipeline_and_log(ctx, components, t_start_ns)
        self._stage_state = ctx.stage_state

        # Build one-hot output list.
        if ctx.token_id < 0:
            logger.error("Pipeline produced no token selection; returning original scores")
            return scores
        result: list[float] = [float("-inf")] * vocab_size
        result[ctx.token_id] = 0.0
        return result
