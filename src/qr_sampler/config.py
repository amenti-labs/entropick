"""Configuration system for qr-sampler.

Uses pydantic-settings for declarative, layered configuration:
init kwargs -> environment variables (QR_*) -> .env file -> field defaults.

Per-request overrides are applied via resolve_config() which creates a new
config instance without mutating the defaults. Infrastructure fields are
protected from per-request override.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from qr_sampler.exceptions import ConfigValidationError

# Fields that can be overridden per-request via SamplingParams.extra_args.
# Infrastructure fields (server address, timeout, retry, fallback mode, etc.)
# are deliberately excluded — they cannot change per-request.
_PER_REQUEST_FIELDS: frozenset[str] = frozenset(
    {
        "signal_amplifier_type",
        "sample_count",
        "population_mean",
        "population_std",
        "uniform_clamp_epsilon",
        "temperature_strategy",
        "fixed_temperature",
        "edt_base_temp",
        "edt_exponent",
        "edt_min_temp",
        "edt_max_temp",
        "top_k",
        "top_p",
        "min_p",
        "xtc_probability",
        "xtc_threshold",
        "adaptive_injection",
        "adaptive_injection_low_h",
        "adaptive_injection_high_h",
        "mirostat_mode",
        "mirostat_tau",
        "mirostat_eta",
        "top_n_sigma",
        "tfs_z",
        "typical_p",
        "eta_cutoff",
        "dry_multiplier",
        "dry_base",
        "dry_allowed_length",
        "dry_penalty_last_n",
        "dry_sequence_breakers",
        "entropix_varentropy",
        "entropix_varentropy_thresh",
        "gumbel_selection",
        "log_level",
        "diagnostic_mode",
        "logit_perturbation_alpha",
        "logit_perturbation_sigma",
        "temp_modulation_beta",
        "drift_step",
        "drift_initial_position",
        "injection_verbose",
    }
)

# All known config field names (populated after class definition).
_ALL_FIELDS: frozenset[str] = frozenset()


class QRSamplerConfig(BaseSettings):
    """Configuration for qr-sampler.

    Resolution order: init kwargs -> env vars (QR_*) -> .env file -> defaults.

    Fields are divided into two groups:
    - **Infrastructure**: Server addresses, timeouts, transport mode — NOT
      overridable per-request.
    - **Sampling parameters**: Amplification, temperature, selection, logging
      — overridable per-request via SamplingParams.extra_args with qr_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="QR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Infrastructure (NOT per-request overridable) ---

    grpc_server_address: str = Field(
        default="localhost:50051",
        description="gRPC entropy server address (host:port or unix:///path)",
    )
    grpc_timeout_ms: float = Field(
        default=5000.0,
        description="gRPC call timeout in milliseconds",
    )
    grpc_retry_count: int = Field(
        default=2,
        description="Number of retries after gRPC failure",
    )
    grpc_mode: str = Field(
        default="unary",
        description="gRPC transport mode: 'unary', 'server_streaming', 'bidi_streaming'",
    )
    grpc_method_path: str = Field(
        default="/qr_entropy.EntropyService/GetEntropy",
        description="gRPC method path for unary RPC (e.g. '/qrng.QuantumRNG/GetRandomBytes')",
    )
    grpc_stream_method_path: str = Field(
        default="/qr_entropy.EntropyService/StreamEntropy",
        description="gRPC method path for streaming RPC (empty string disables streaming modes)",
    )
    grpc_api_key: str = Field(
        default="",
        description="API key sent via gRPC metadata (empty = no auth)",
    )
    grpc_api_key_header: str = Field(
        default="api-key",
        description="gRPC metadata header name for the API key",
    )
    grpc_tls_enabled: bool = Field(
        default=False,
        description="Enable TLS for gRPC channel (use secure_channel instead of insecure_channel)",
    )
    grpc_tls_ca_cert: str = Field(
        default="",
        description="Path to CA certificate PEM file for TLS verification",
    )
    grpc_tls_client_cert: str = Field(
        default="",
        description="Path to client certificate PEM file for mutual TLS (mTLS)",
    )
    grpc_tls_client_key: str = Field(
        default="",
        description="Path to client private key PEM file for mutual TLS (mTLS)",
    )
    fallback_mode: str = Field(
        default="system",
        description="Fallback entropy source: 'error', 'system', 'mock_uniform'",
    )
    entropy_source_type: str = Field(
        default="system",
        description="Primary entropy source identifier",
    )

    # --- Circuit Breaker (NOT per-request overridable) ---

    cb_window_size: int = Field(
        default=100,
        description="Rolling latency window size for P99 computation",
    )
    cb_min_timeout_ms: float = Field(
        default=5.0,
        description="Minimum adaptive timeout in milliseconds",
    )
    cb_timeout_multiplier: float = Field(
        default=1.5,
        description="Multiplier applied to P99 latency for adaptive timeout",
    )
    cb_recovery_window_s: float = Field(
        default=10.0,
        description="Seconds to wait before half-open retry after circuit opens",
    )
    cb_max_consecutive_failures: int = Field(
        default=3,
        description="Consecutive failures before circuit breaker opens",
    )

    # --- Signal Amplification (per-request overridable) ---

    signal_amplifier_type: str = Field(
        default="zscore_mean",
        description="Signal amplification algorithm",
    )
    sample_count: int = Field(
        default=20480,
        description="Number of entropy bytes to fetch per token",
        gt=0,
    )
    population_mean: float = Field(
        default=127.5,
        description="Null-hypothesis mean of byte values {0..255}",
    )
    population_std: float = Field(
        default=73.61215932167728,
        description="Population std for continuous uniform [0, 255]",
    )
    uniform_clamp_epsilon: float = Field(
        default=1e-10,
        description="Clamp u to (epsilon, 1-epsilon) to avoid degenerate CDF",
    )
    ecdf_calibration_samples: int = Field(
        default=2000,
        ge=100,
        description="Samples for ECDF calibration",
    )

    # --- Temperature Strategy (per-request overridable) ---

    temperature_strategy: str = Field(
        default="fixed",
        description="Temperature strategy: 'fixed' or 'edt'",
    )
    fixed_temperature: float = Field(
        default=0.7,
        description="Constant temperature for fixed strategy",
    )
    edt_base_temp: float = Field(
        default=0.8,
        description="Base coefficient for EDT",
    )
    edt_exponent: float = Field(
        default=0.5,
        description="Power-law exponent for EDT",
    )
    edt_min_temp: float = Field(
        default=0.1,
        description="EDT temperature floor",
    )
    edt_max_temp: float = Field(
        default=2.0,
        description="EDT temperature ceiling",
    )

    # --- Token Selection (per-request overridable) ---

    top_k: int = Field(
        default=0,
        description="Top-k filtering (<=0 disables)",
    )
    top_p: float = Field(
        default=1.0,
        description="Nucleus sampling threshold (1.0 disables)",
    )
    min_p: float = Field(
        default=0.0,
        description="Min-P threshold: remove tokens with p < min_p * max(p). 0 = disabled.",
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )

    # --- XTC: Exclude Top Choices (per-request overridable) ---

    xtc_probability: float = Field(
        default=0.0,
        description="XTC: probability of excluding each top token. 0 = disabled.",
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    xtc_threshold: float = Field(
        default=0.1,
        description="XTC: only tokens with p >= threshold are candidates for exclusion.",
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )

    # --- Adaptive Injection (per-request overridable) ---

    adaptive_injection: bool = Field(
        default=False,
        description="Scale injection intensity by distribution entropy/varentropy.",
    )
    adaptive_injection_low_h: float = Field(
        default=1.0,
        description="Entropy threshold below which model is 'confident' (nats).",
        ge=0.0,
        allow_inf_nan=False,
    )
    adaptive_injection_high_h: float = Field(
        default=3.0,
        description="Entropy threshold above which model is 'uncertain' (nats).",
        ge=0.0,
        allow_inf_nan=False,
    )

    # --- Mirostat (per-request overridable) ---

    mirostat_mode: int = Field(
        default=0,
        description="Mirostat mode: 0=disabled, 2=mirostat v2",
        ge=0,
        le=2,
    )
    mirostat_tau: float = Field(
        default=5.0,
        description="Mirostat target surprise rate (nats)",
        ge=0.0,
        allow_inf_nan=False,
    )
    mirostat_eta: float = Field(
        default=0.1,
        description="Mirostat learning rate",
        ge=0.0,
        allow_inf_nan=False,
    )

    # --- Additional filtering stages (per-request overridable) ---

    top_n_sigma: float = Field(
        default=0.0,
        description="Top-n-sigma: keep logits within n std of max. 0 = disabled.",
        ge=0.0,
        allow_inf_nan=False,
    )
    tfs_z: float = Field(
        default=1.0,
        description="Tail-free sampling z threshold. 1.0 = disabled.",
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    typical_p: float = Field(
        default=1.0,
        description="Locally typical sampling threshold. 1.0 = disabled.",
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    eta_cutoff: float = Field(
        default=0.0,
        description="Eta sampling cutoff (in 1e-4 units). 0 = disabled.",
        ge=0.0,
        allow_inf_nan=False,
    )

    # --- DRY penalty (per-request overridable) ---

    dry_multiplier: float = Field(
        default=0.0,
        description="DRY repetition penalty multiplier. 0 = disabled.",
        ge=0.0,
        allow_inf_nan=False,
    )
    dry_base: float = Field(
        default=1.75,
        description="DRY penalty base for exponential scaling.",
        ge=1.0,
        allow_inf_nan=False,
    )
    dry_allowed_length: int = Field(
        default=2,
        description="DRY minimum repeated sequence length to penalize.",
        ge=1,
    )
    dry_penalty_last_n: int = Field(
        default=-1,
        description="DRY lookback window in tokens. -1 = full context.",
    )
    dry_sequence_breakers: str = Field(
        default="",
        description="Comma-separated integer token IDs that break DRY sequence matching.",
    )

    # --- Entropix / varentropy (per-request overridable) ---

    entropix_varentropy: bool = Field(
        default=False,
        description="Enable varentropy-based regime switching (extends adaptive injection).",
    )
    entropix_varentropy_thresh: float = Field(
        default=3.0,
        description="Varentropy threshold for 'confused' regime (nats^2).",
        ge=0.0,
        allow_inf_nan=False,
    )

    # --- Gumbel selection (per-request overridable) ---

    gumbel_selection: bool = Field(
        default=False,
        description="Use Gumbel-Max selection instead of CDF binary search.",
    )

    # --- Sham QRNG (infrastructure — latency set at init) ---

    sham_qrng_latency_ms: float = Field(
        default=0.0,
        description="Simulate QRNG latency with os.urandom (ms). 0 = disabled. For controls.",
        ge=0.0,
        allow_inf_nan=False,
    )

    # --- Injection methods (per-request overridable) ---

    logit_perturbation_alpha: float = Field(
        default=0.0,
        description="Logit perturbation magnitude. 0 = disabled.",
        ge=0.0,
        allow_inf_nan=False,
    )
    logit_perturbation_sigma: float = Field(
        default=1.0,
        description="Logit perturbation: standard deviation of noise before scaling by alpha.",
        ge=0.0,
        allow_inf_nan=False,
    )
    temp_modulation_beta: float = Field(
        default=0.0,
        description="Temperature modulation magnitude. 0 = disabled.",
        ge=0.0,
        allow_inf_nan=False,
    )
    drift_step: float = Field(
        default=0.0,
        description="Selection drift step size. 0 = disabled.",
        ge=0.0,
        allow_inf_nan=False,
    )
    drift_initial_position: float = Field(
        default=0.5,
        description="Selection drift initial position in [0, 1).",
        ge=0.0,
        lt=1.0,
        allow_inf_nan=False,
    )
    injection_verbose: bool = Field(
        default=False,
        description="Log injection method activity at each token.",
    )
    # --- Logging (per-request overridable) ---

    log_level: str = Field(
        default="summary",
        description="Logging verbosity: 'none', 'summary', 'full'",
    )
    diagnostic_mode: bool = Field(
        default=False,
        description="Store all token records in memory for analysis",
    )

    # --- OpenEntropy (all infrastructure — source constructed at init) ---

    oe_conditioning: str = Field(
        default="raw",
        description="OpenEntropy conditioning mode: raw, sha256, vonneumann",
    )
    oe_sources: str = Field(
        default="",
        description="Comma-separated OpenEntropy source names. Empty = all available.",
    )
    oe_parallel: bool = Field(
        default=True,
        description="Collect OpenEntropy sources in parallel",
    )
    oe_timeout: float = Field(
        default=5.0,
        description="OpenEntropy collection timeout in seconds",
    )


# Populate _ALL_FIELDS now that the class is defined.
_ALL_FIELDS = frozenset(QRSamplerConfig.model_fields.keys())


def _strip_prefix(key: str) -> str:
    """Strip the 'qr_' prefix from an extra_args key.

    Args:
        key: The key with or without 'qr_' prefix.

    Returns:
        The key with 'qr_' prefix removed if present.
    """
    if key.startswith("qr_"):
        return key[3:]
    return key


def validate_extra_args(extra_args: dict[str, Any]) -> None:
    """Validate all qr_* keys in extra_args without creating a config.

    This is called by validate_params() at request creation time to
    reject bad keys early, before the request enters the batch.

    Args:
        extra_args: Dictionary of extra arguments, potentially with qr_ prefix.

    Raises:
        ConfigValidationError: If any qr_* key is unknown or non-overridable.
    """
    for key in extra_args:
        if not key.startswith("qr_"):
            continue
        field_name = _strip_prefix(key)
        if field_name not in _ALL_FIELDS:
            raise ConfigValidationError(
                f"Unknown config field: '{key}' (no field '{field_name}' exists)"
            )
        if field_name not in _PER_REQUEST_FIELDS:
            raise ConfigValidationError(
                f"Field '{field_name}' is an infrastructure field and cannot be "
                f"overridden per-request via extra_args"
            )


def resolve_config(
    defaults: QRSamplerConfig,
    extra_args: dict[str, Any] | None,
) -> QRSamplerConfig:
    """Create a new config instance merging defaults with per-request overrides.

    The extra_args keys use 'qr_' prefix (e.g., 'qr_top_k': 100).
    Only fields in _PER_REQUEST_FIELDS are overridable. Keys without the
    'qr_' prefix are silently ignored (they belong to other processors).

    Args:
        defaults: The base configuration loaded from environment.
        extra_args: Per-request overrides from SamplingParams.extra_args.

    Returns:
        A new QRSamplerConfig with overrides applied.

    Raises:
        ConfigValidationError: If any qr_* key is unknown or non-overridable.
    """
    if not extra_args:
        return defaults

    # Validate all qr_* keys first.
    validate_extra_args(extra_args)

    # Extract and apply valid overrides.
    overrides: dict[str, Any] = {}
    for key, value in extra_args.items():
        if not key.startswith("qr_"):
            continue
        field_name = _strip_prefix(key)
        overrides[field_name] = value

    if not overrides:
        return defaults

    # Use model_validate on a merged dict to ensure type coercion.
    # model_copy(update=...) skips validation, so string "100" would not
    # be coerced to int 100. model_validate runs the full validator.
    merged = defaults.model_dump()
    merged.update(overrides)
    return QRSamplerConfig.model_validate(merged)
