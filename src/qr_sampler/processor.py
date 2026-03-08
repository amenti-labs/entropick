"""vLLM V1 LogitsProcessor -- the integration layer for qr-sampler.

Thin adapter that builds a ``SamplingContext`` for each batch row and
runs it through the configurable pipeline of ``PipelineStage`` instances.

Registered via entry point::

    [project.entry-points."vllm.logits_processors"]
    qr_sampler = "qr_sampler.processor:QRSamplerLogitsProcessor"

The processor applies globally to all requests in a vLLM instance. Deploy
separate instances for different sampling strategies.
"""

from __future__ import annotations

import importlib
import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from qr_sampler.adapters._base import _build_entropy_source, _config_hash
from qr_sampler.amplification.registry import AmplifierRegistry
from qr_sampler.config import QRSamplerConfig, resolve_config, validate_extra_args
from qr_sampler.logging.logger import SamplingLogger
from qr_sampler.logging.types import TokenSamplingRecord
from qr_sampler.pipeline.context import SamplingContext
from qr_sampler.stages import build_default_pipeline
from qr_sampler.temperature.registry import TemperatureStrategyRegistry

if TYPE_CHECKING:
    from qr_sampler.amplification.base import SignalAmplifier
    from qr_sampler.entropy.base import EntropySource
    from qr_sampler.pipeline.stage import PipelineStage
    from qr_sampler.temperature.base import FloatArray, TemperatureStrategy

logger = logging.getLogger("qr_sampler")

# Default vocabulary size when vllm_config does not provide one (testing).
_DEFAULT_VOCAB_SIZE = 32000


class _RequestState:
    """Per-request state tracked across engine steps.

    Attributes:
        config: Resolved per-request configuration.
        amplifier: Signal amplifier for this request.
        strategy: Temperature strategy for this request.
        config_hash_str: Short hash for logging.
        stage_state: Persistent state dict for pipeline stages.
    """

    __slots__ = ("amplifier", "config", "config_hash_str", "stage_state", "strategy")

    def __init__(
        self,
        config: QRSamplerConfig,
        amplifier: SignalAmplifier,
        strategy: TemperatureStrategy,
        config_hash_str: str,
        stage_state: dict[str, Any],
    ) -> None:
        self.config = config
        self.amplifier = amplifier
        self.strategy = strategy
        self.config_hash_str = config_hash_str
        self.stage_state = stage_state

    @property
    def drift_position(self) -> float:
        """Convenience accessor for selection drift position."""
        return float(self.stage_state.get("selection_drift.position", 0.5))

    @drift_position.setter
    def drift_position(self, value: float) -> None:
        self.stage_state["selection_drift.position"] = value


class QRSamplerLogitsProcessor:
    """vLLM V1 LogitsProcessor that replaces token sampling with
    external-entropy-driven selection.

    The processor builds a ``SamplingContext`` for each batch row and
    runs it through a configurable pipeline of ``PipelineStage`` instances.
    The selected token is forced via a one-hot logit vector.

    Constructor signature matches vLLM V1's ``LogitsProcessor`` ABC::

        __init__(self, vllm_config, device, is_pin_memory)
    """

    def __init__(
        self,
        vllm_config: Any = None,
        device: Any = None,
        is_pin_memory: bool = False,
        pipeline: list[PipelineStage] | None = None,
    ) -> None:
        """Initialize the processor and all subsystems.

        Args:
            vllm_config: vLLM's ``VllmConfig`` object (provides vocab_size).
                ``None`` in test environments -- uses ``_DEFAULT_VOCAB_SIZE``.
            device: ``torch.device`` for tensor operations. ``None`` in tests.
            is_pin_memory: Whether to use pinned CPU memory for transfers.
            pipeline: Custom pipeline stages. If ``None``, uses the default
                pipeline (all built-in stages).
        """
        # --- Extract vocab_size ---
        self._vocab_size = self._extract_vocab_size(vllm_config)
        self._device = device
        self._is_pin_memory = is_pin_memory

        # --- Load default configuration ---
        self._default_config = QRSamplerConfig()

        # --- Build shared components ---
        self._entropy_source = _build_entropy_source(self._default_config)
        self._default_amplifier = AmplifierRegistry.build(self._default_config)
        if hasattr(self._default_amplifier, "calibrate"):
            self._default_amplifier.calibrate(self._entropy_source, self._default_config)
        self._default_strategy = TemperatureStrategyRegistry.build(
            self._default_config, self._vocab_size
        )
        self._logger = SamplingLogger(self._default_config)

        # --- Pre-compute default state ---
        self._default_config_hash = _config_hash(self._default_config)

        # --- Pre-allocate tensors ---
        self._onehot_template = self._create_onehot_template()
        self._cpu_buffer = self._create_cpu_buffer()

        # --- Pipeline ---
        self._pipeline: list[PipelineStage] = (
            pipeline if pipeline is not None else build_default_pipeline()
        )

        # --- Per-request state ---
        self._request_states: dict[int, _RequestState] = {}

        logger.debug(
            "QRSamplerLogitsProcessor initialized: vocab_size=%d, "
            "entropy_source=%s, pipeline=%d stages",
            self._vocab_size,
            self._entropy_source.name,
            len(self._pipeline),
        )

    @staticmethod
    def _extract_vocab_size(vllm_config: Any) -> int:
        """Extract vocabulary size from vLLM config, with fallback.

        Args:
            vllm_config: vLLM config object, or ``None`` for tests.

        Returns:
            Vocabulary size as integer.
        """
        if vllm_config is None:
            return _DEFAULT_VOCAB_SIZE

        try:
            return int(vllm_config.model_config.hf_text_config.vocab_size)
        except AttributeError:
            pass

        try:
            return int(vllm_config.vocab_size)
        except AttributeError:
            pass

        logger.warning(
            "Could not extract vocab_size from vllm_config, using default %d",
            _DEFAULT_VOCAB_SIZE,
        )
        return _DEFAULT_VOCAB_SIZE

    def _create_onehot_template(self) -> Any:
        """Create the one-hot template tensor filled with -inf."""
        try:
            torch = importlib.import_module("torch")
            return torch.full(
                (self._vocab_size,),
                float("-inf"),
                device=self._device,
                dtype=torch.float32,
            )
        except (ImportError, OSError):
            return np.full(self._vocab_size, float("-inf"), dtype=np.float32)

    def _create_cpu_buffer(self) -> Any:
        """Create a pinned-memory CPU buffer for transfers."""
        if not self._is_pin_memory:
            return None
        try:
            torch = importlib.import_module("torch")
            return torch.empty(self._vocab_size, dtype=torch.float32, pin_memory=True)
        except (ImportError, OSError):
            return None

    def is_argmax_invariant(self) -> bool:
        """Return ``False`` -- this processor fundamentally changes token selection."""
        return False

    @classmethod
    def validate_params(cls, params: Any) -> None:
        """Validate ``qr_*`` keys in ``params.extra_args``.

        Called by vLLM at request creation time to reject bad keys early.

        Args:
            params: vLLM ``SamplingParams`` object with ``extra_args`` dict.

        Raises:
            ConfigValidationError: If any ``qr_*`` key is unknown or
                non-overridable.
        """
        extra_args = getattr(params, "extra_args", None) or {}
        if extra_args:
            validate_extra_args(extra_args)

    def update_state(self, batch_update: Any | None) -> None:
        """Process batch composition changes.

        Must be called every engine step before ``apply()``. Processes
        changes in the required order: removed -> moved -> added.

        Args:
            batch_update: A ``BatchUpdate`` with ``removed``, ``moved``,
                and ``added`` sequences, or ``None`` if no changes.
        """
        if batch_update is None:
            return

        # 1. Process removals.
        for removed in getattr(batch_update, "removed", []):
            req_idx = removed if isinstance(removed, int) else getattr(removed, "req_index", None)
            if req_idx is not None:
                self._request_states.pop(req_idx, None)

        # 2. Process moves (index reassignments).
        for moved in getattr(batch_update, "moved", []):
            if hasattr(moved, "src_index") and hasattr(moved, "dst_index"):
                state = self._request_states.pop(moved.src_index, None)
                if state is not None:
                    self._request_states[moved.dst_index] = state

        # 3. Process additions.
        for added in getattr(batch_update, "added", []):
            req_idx = getattr(added, "req_index", None)
            if req_idx is None:
                continue

            extra_args = getattr(getattr(added, "sampling_params", None), "extra_args", None) or {}
            req_config = resolve_config(self._default_config, extra_args)

            if req_config is self._default_config:
                amplifier = self._default_amplifier
                strategy = self._default_strategy
                hash_str = self._default_config_hash
            else:
                amplifier = AmplifierRegistry.build(req_config)
                if hasattr(amplifier, "calibrate"):
                    amplifier.calibrate(self._entropy_source, req_config)
                strategy = TemperatureStrategyRegistry.build(req_config, self._vocab_size)
                hash_str = _config_hash(req_config)

            # Initialize persistent stage state.
            stage_state: dict[str, Any] = {
                "selection_drift.position": req_config.drift_initial_position,
                "mirostat.mu": 2.0 * req_config.mirostat_tau,
                "token_history": [],
            }

            self._request_states[req_idx] = _RequestState(
                config=req_config,
                amplifier=amplifier,
                strategy=strategy,
                config_hash_str=hash_str,
                stage_state=stage_state,
            )

    def apply(self, logits: Any) -> Any:
        """Run the pipeline on each row of the logit tensor.

        For each request in the batch:
            1. Build a ``SamplingContext`` from per-request state
            2. Run all pipeline stages
            3. Force one-hot logits
            4. Log sampling record

        Args:
            logits: 2-D tensor of shape ``(num_requests, vocab_size)``.
                May be a ``torch.Tensor`` or a ``numpy.ndarray``.

        Returns:
            The modified logits tensor (in-place).
        """
        if hasattr(logits, "shape"):
            num_requests = logits.shape[0] if len(logits.shape) > 1 else 1
        else:
            return logits

        if num_requests == 0:
            return logits

        is_numpy = isinstance(logits, np.ndarray)
        is_1d = len(logits.shape) == 1

        for i in range(num_requests):
            self._apply_row(logits, i, is_numpy, is_1d)

        return logits

    def _apply_row(self, logits: Any, i: int, is_numpy: bool, is_1d: bool) -> None:
        """Process a single row through the pipeline.

        Args:
            logits: Full logit tensor (modified in-place).
            i: Row index in the batch.
            is_numpy: Whether logits is a numpy array.
            is_1d: Whether logits is 1-D (single request, no batch dim).
        """
        t_start_ns = time.perf_counter_ns()

        # --- Resolve per-request state ---
        state = self._request_states.get(i)
        if state is not None:
            config = state.config
            amplifier = state.amplifier
            strategy = state.strategy
            hash_str = state.config_hash_str
            stage_state = state.stage_state
        else:
            config = self._default_config
            amplifier = self._default_amplifier
            strategy = self._default_strategy
            hash_str = self._default_config_hash
            stage_state = {}
            logger.debug("No request state for row %d, using defaults", i)

        # --- Extract row as numpy ---
        if is_1d:
            row = logits if is_numpy else self._to_numpy(logits)
        else:
            row = logits[i] if is_numpy else self._to_numpy(logits[i])

        # --- Build context ---
        ctx = SamplingContext(
            row=row,
            config=config,
            entropy_source=self._entropy_source,
            amplifier=amplifier,
            temperature_strategy=strategy,
            config_hash=hash_str,
            stage_state=stage_state,
        )

        # --- Run pipeline ---
        for stage in self._pipeline:
            stage(ctx)

        # --- Append selected token to history (used by DRY penalty) ---
        if ctx.token_id >= 0:
            ctx.stage_state.setdefault("token_history", []).append(ctx.token_id)

        # --- Persist stage state back to request state ---
        if state is not None:
            state.stage_state = ctx.stage_state

        # --- Force one-hot logits ---
        if ctx.token_id < 0:
            logger.error("Pipeline produced no token selection; skipping one-hot forcing")
            return
        if is_1d:
            self._force_onehot(logits, ctx.token_id, is_numpy)
        else:
            self._force_onehot_row(logits, i, ctx.token_id, is_numpy)

        # --- Log sampling record ---
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
            temperature_strategy=config.temperature_strategy,
            shannon_entropy=ctx.shannon_entropy,
            temperature_used=ctx.temperature,
            token_id=ctx.token_id,
            token_rank=ctx.token_rank,
            token_prob=ctx.token_prob,
            num_candidates=ctx.num_candidates,
            config_hash=hash_str,
            injection_alpha=ctx.effective_alpha,
            injection_beta=ctx.effective_beta,
            injection_step=ctx.effective_step,
            injection_scale=ctx.injection_scale,
        )
        self._logger.log_token(record)

    @staticmethod
    def _to_numpy(tensor: Any) -> FloatArray:
        """Convert a tensor to a numpy array with zero-copy where possible."""
        if isinstance(tensor, np.ndarray):
            return tensor
        try:
            if bool(getattr(tensor, "is_cuda", False)):
                result: FloatArray = tensor.detach().cpu().numpy()
            elif hasattr(tensor, "is_cpu") and not bool(tensor.is_cpu):
                result = tensor.detach().cpu().numpy()
            else:
                result = tensor.detach().numpy()
            return result
        except AttributeError:
            return np.asarray(tensor)

    def _force_onehot(self, logits: Any, token_id: int, is_numpy: bool) -> None:
        """Force 1-D logits to one-hot: all -inf except token_id = 0.0."""
        if is_numpy:
            logits[:] = float("-inf")
            logits[token_id] = 0.0
        else:
            logits.copy_(self._onehot_template, non_blocking=True)
            logits[token_id] = 0.0

    def _force_onehot_row(self, logits: Any, row_idx: int, token_id: int, is_numpy: bool) -> None:
        """Force a batch row to one-hot: all -inf except token_id = 0.0."""
        if is_numpy:
            logits[row_idx, :] = float("-inf")
            logits[row_idx, token_id] = 0.0
        else:
            logits[row_idx].copy_(self._onehot_template, non_blocking=True)
            logits[row_idx, token_id] = 0.0

    @property
    def pipeline(self) -> list[PipelineStage]:
        """The active pipeline stages."""
        return self._pipeline

    @property
    def entropy_source(self) -> EntropySource:
        """The active entropy source (may be a FallbackEntropySource wrapper)."""
        return self._entropy_source

    @property
    def default_config(self) -> QRSamplerConfig:
        """The default configuration loaded from environment."""
        return self._default_config

    @property
    def sampling_logger(self) -> SamplingLogger:
        """The diagnostic logger for this processor."""
        return self._logger

    def close(self) -> None:
        """Release all resources held by the processor."""
        self._entropy_source.close()
