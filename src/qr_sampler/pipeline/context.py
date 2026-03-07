"""Mutable context passed through pipeline stages for one token."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from qr_sampler.amplification.base import SignalAmplifier
    from qr_sampler.config import QRSamplerConfig
    from qr_sampler.entropy.base import EntropySource
    from qr_sampler.temperature.base import TemperatureStrategy


@dataclass
class SamplingContext:
    """Mutable state bag for a single token's sampling pipeline.

    Created by the processor for each batch row, passed through every
    pipeline stage, then read back by the processor for one-hot forcing
    and diagnostic logging.

    Stages read the fields they need and write the fields they produce.
    Convention: stages should not overwrite fields set by earlier stages
    unless that is their explicit purpose (e.g., CorrelatedWalk replaces ``u``).
    """

    # --- Input (set by processor before pipeline runs) ---

    row: np.ndarray[Any, np.dtype[np.floating[Any]]]
    """Logit row for this request (1-D, mutable)."""

    config: QRSamplerConfig
    """Resolved per-request configuration."""

    entropy_source: EntropySource
    """Entropy source (may be a FallbackEntropySource wrapper)."""

    amplifier: SignalAmplifier
    """Signal amplifier for this request."""

    temperature_strategy: TemperatureStrategy
    """Temperature strategy for this request."""

    config_hash: str
    """Short hash of the config for logging."""

    # --- Per-request persistent state (mutable reference) ---

    stage_state: dict[str, Any] = field(default_factory=dict)
    """Persistent state that survives across apply() calls.

    Keyed by stage name (e.g., ``"correlated_walk.position"``).
    The processor copies this from/to the per-request state store.
    """

    # --- Pipeline outputs (written by stages) ---

    temperature: float = 0.0
    """Final temperature after all modulations."""

    shannon_entropy: float = 0.0
    """Shannon entropy of the logit distribution (nats)."""

    u: float = 0.0
    """Uniform value from amplification, in (0, 1). Used for CDF selection."""

    token_id: int = -1
    """Vocabulary index of the selected token."""

    token_rank: int = -1
    """Rank of selected token (0 = most probable)."""

    token_prob: float = 0.0
    """Probability of the selected token."""

    num_candidates: int = 0
    """Number of tokens surviving filtering."""

    # --- Entropy diagnostics (written by EntropyFetchStage) ---

    entropy_source_name: str = ""
    """Name of the entropy source that provided bytes."""

    entropy_is_fallback: bool = False
    """True if a fallback source was used."""

    sample_mean: float = 0.0
    """Mean of raw entropy bytes (expected ~127.5 unbiased)."""

    z_score: float = 0.0
    """Z-score from signal amplification."""

    # --- Adaptive injection scale (written by AdaptiveInjectionStage) ---

    injection_scale: float = 1.0
    """Scaling factor for injection methods (0.0 = skip, 1.0 = full strength).

    Set by ``AdaptiveInjectionStage`` based on distribution entropy/varentropy.
    Read by injection stages (M1, M2, M3) to modulate their intensity.
    """

    # --- Timing (accumulated by stages) ---

    entropy_fetch_ms: float = 0.0
    """Total time for entropy fetching (including injection fetches)."""
