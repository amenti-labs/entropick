# CLAUDE.md -- Codebase Guide for Coding Agents

## What this project is

`entropick` (package: `qr_sampler`) is a vLLM V1 LogitsProcessor plugin that replaces standard token sampling with external-entropy-driven selection. It fetches random bytes from any entropy source (QRNGs via gRPC, OpenEntropy hardware noise, OS randomness, CPU timing jitter), amplifies the signal into a uniform float via z-score statistics, and uses that float to select a token from a probability-ordered CDF.

This is a **pure plugin** -- it does not modify vLLM source code. It registers via the `vllm.logits_processors` entry point in `pyproject.toml`. It also provides adapters for Hugging Face Transformers and llama.cpp. The primary use case is consciousness-research: studying whether conscious intent can influence quantum-random processes in LLM token selection.

## Commands

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_config.py -v
pytest tests/test_amplification/ -v
pytest tests/test_temperature/ -v
pytest tests/test_selection/ -v
pytest tests/test_logging/ -v
pytest tests/test_entropy/ -v
pytest tests/test_processor.py -v
pytest tests/test_pipeline/ -v
pytest tests/test_statistical_properties.py -v

# Run with coverage
pytest tests/ -v --cov=src/qr_sampler --cov-report=term-missing

# Install in editable mode
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# Lint and format
ruff check src/ tests/
ruff format --check src/ tests/

# Type check
mypy --strict src/
```

## File map

```
src/qr_sampler/
+-- __init__.py                    # Package version (setuptools-scm), re-exports
+-- config.py                      # QRSamplerConfig (pydantic BaseSettings), resolve_config(), validate_extra_args()
+-- exceptions.py                  # QRSamplerError -> {EntropyUnavailableError, ConfigValidationError, SignalAmplificationError, TokenSelectionError}
+-- processor.py                   # QRSamplerLogitsProcessor -- thin adapter, builds SamplingContext and runs pipeline
+-- py.typed                       # PEP 561 marker
+-- pipeline/
|   +-- __init__.py                # Re-exports PipelineStage, SamplingContext, StageRegistry
|   +-- stage.py                   # PipelineStage protocol (runtime_checkable): name + __call__(ctx)
|   +-- context.py                 # SamplingContext mutable dataclass -- state bag passed through all stages
|   +-- registry.py                # StageRegistry: @register() decorator + entry-point auto-discovery
+-- stages/
|   +-- __init__.py                # Exports all 13 stage classes + build_default_pipeline()
|   +-- _utils.py                  # Shared stable_softmax(), shannon_entropy_from_probs()
|   +-- adaptive_injection.py      # AdaptiveInjectionStage: entropy-based injection scaling
|   +-- logit_perturbation.py             # LogitPerturbationStage: per-logit quantum noise
|   +-- dry.py                     # DRYPenaltyStage: Don't Repeat Yourself n-gram penalty
|   +-- top_n_sigma.py             # TopNSigmaStage: logit-space sigma filtering (pre-softmax)
|   +-- temperature.py             # TemperatureStage: compute temperature via strategy
|   +-- temp_modulation.py           # TemperatureModulationStage: quantum temperature modulation
|   +-- min_p.py                   # MinPStage: dynamic probability floor filtering
|   +-- xtc.py                     # XTCStage: quantum-driven top-token exclusion
|   +-- entropy_fetch.py           # EntropyFetchStage: JIT entropy fetch + signal amplification
|   +-- selection_drift.py         # SelectionDriftStage: per-request selection drift
|   +-- mirostat.py                # MirostatStage: Mirostat v2 adaptive perplexity control
|   +-- gumbel_selection.py        # GumbelSelectionStage: Gumbel-Max trick for selection
|   +-- selection.py               # SelectionStage: CDF-based token selection (skipped if mirostat/gumbel active)
+-- injection/
|   +-- __init__.py                # Re-exports LogitPerturbation, TemperatureModulation, SelectionDrift
|   +-- _entropy_utils.py          # Shared bytes_to_uniform() helper (z-score -> CDF -> uniform)
|   +-- logit_perturbation.py             # Gaussian logit perturbation (quantum-seeded)
|   +-- temp_modulation.py           # Temperature modulation via quantum entropy
|   +-- selection_drift.py         # Per-request selection drift position
+-- amplification/
|   +-- __init__.py                # Re-exports
|   +-- base.py                    # SignalAmplifier ABC, AmplificationResult frozen dataclass
|   +-- registry.py                # AmplifierRegistry (decorator + build pattern)
|   +-- zscore.py                  # ZScoreMeanAmplifier (z-score -> normal CDF -> uniform)
|   +-- ecdf.py                    # ECDFAmplifier: empirical CDF amplification (no distributional assumptions)
|   +-- calibration.py             # calibrate_population_stats(), measure_entropy_rate() for QRNG devices
+-- entropy/
|   +-- __init__.py                # Re-exports
|   +-- base.py                    # EntropySource ABC (name, is_available, get_random_bytes, get_random_float64, close, health_check)
|   +-- registry.py                # EntropySourceRegistry with entry-point auto-discovery from qr_sampler.entropy_sources
|   +-- quantum.py                 # QuantumGrpcSource: 3 modes (unary, server_streaming, bidi_streaming), circuit breaker, grpc.aio
|   +-- system.py                  # SystemEntropySource: os.urandom()
|   +-- mock.py                    # MockUniformSource: configurable seed/bias for testing
|   +-- fallback.py                # FallbackEntropySource: composition wrapper, catches only EntropyUnavailableError
|   +-- sham.py                    # ShamQrngSource: os.urandom() + simulated QRNG latency (for double-blind controls)
+-- logging/
|   +-- __init__.py                # Re-exports
|   +-- types.py                   # TokenSamplingRecord frozen dataclass (20 fields, __slots__)
|   +-- logger.py                  # SamplingLogger: none/summary/full log levels, diagnostic_mode
+-- analysis/
|   +-- __init__.py                # Re-exports all analysis functions
|   +-- _utils.py                  # Shared helpers (_require_scipy_stats)
|   +-- persistence.py             # save_records() / load_records() for JSONL files
|   +-- statistics.py              # Statistical test battery: autocorrelation, runs, Hurst, ApEn, cumulative deviation, Bayesian
|   +-- compare.py                 # Two-sample comparison: Mann-Whitney, KS, Welch's t, Cohen's d, Stouffer's z
+-- adapters/
|   +-- __init__.py                # Lazy-import adapter classes (no framework deps at import time)
|   +-- _base.py                   # AdapterComponents, _AdapterBase, _run_pipeline_and_log, _init_stage_state
|   +-- transformers.py            # QRSamplerLogitsProcessorHF: Hugging Face model.generate() adapter
|   +-- llamacpp.py                # QRSamplerCallback: llama-cpp-python callback adapter
+-- proto/
|   +-- __init__.py
|   +-- entropy_service.proto      # gRPC proto: GetEntropy (unary) + StreamEntropy (bidi)
|   +-- entropy_service_pb2.py     # Hand-written protobuf message stubs
|   +-- entropy_service_pb2_grpc.py # Hand-written gRPC client + server stubs
+-- selection/
|   +-- __init__.py                # Re-exports
|   +-- types.py                   # SelectionResult frozen dataclass
|   +-- selector.py                # TokenSelector: top-k -> softmax -> top-p -> CDF -> searchsorted
+-- temperature/
    +-- __init__.py                # Re-exports
    +-- base.py                    # TemperatureStrategy ABC, TemperatureResult, compute_shannon_entropy()
    +-- registry.py                # TemperatureStrategyRegistry (passes vocab_size if constructor accepts it)
    +-- fixed.py                   # FixedTemperatureStrategy: constant temperature
    +-- edt.py                     # EDTTemperatureStrategy: entropy-dependent, H_norm^exp scaling

tests/
+-- __init__.py
+-- conftest.py                    # Shared fixtures: default_config, sample_logits, batch_logits
+-- helpers.py                     # Shared mock objects (MockVllmConfig, etc.) and test utilities
+-- test_config.py                 # Config defaults, env vars, per-request resolution, validation
+-- test_contracts.py              # Auto-discovers all implementations and verifies ABC/protocol contracts
+-- test_processor.py              # Integration: full pipeline, batch processing, update_state, one-hot
+-- test_pipeline/
|   +-- test_stages.py             # Protocol compliance, registry, default pipeline, custom pipelines, stage_state
|   +-- test_new_stages.py         # Min-P, XTC, adaptive injection stage tests + edge cases
|   +-- test_top_n_sigma.py        # Top-n-sigma filtering tests
|   +-- test_mirostat.py           # Mirostat v2 tests
|   +-- test_dry.py                # DRY penalty tests
|   +-- test_gumbel.py             # Gumbel-Max unit tests (standalone stage)
|   +-- test_gumbel_selection.py   # Gumbel-Max integration tests (full pipeline)
+-- test_injection/
|   +-- test_logit_perturbation.py  # Logit perturbation: enabled/disabled, reproducibility, scaling, entropy failure
|   +-- test_temp_modulation.py    # Temperature modulation: enabled/disabled, clamping, range, entropy failure
|   +-- test_selection_drift.py    # Selection drift: drift, bounds, no-op, entropy failure
|   +-- test_integration.py        # Combined: all-methods, backward-compat, per-request-override
+-- test_analysis/
|   +-- test_persistence.py        # JSONL save/load round-trip
|   +-- test_statistics.py         # Statistical test battery (requires scipy)
|   +-- test_compare.py            # Two-sample comparison tests (requires scipy)
+-- test_adapters/
|   +-- test_transformers_adapter.py # HF adapter + llama-cpp tests with mock tensors
+-- test_statistical_properties.py # KS-test uniformity, bias detection, EDT monotonicity (requires scipy)
+-- test_amplification/
|   +-- test_zscore.py             # Known values, SEM derivation, edge cases, frozen immutability
|   +-- test_calibration.py        # Calibration utility tests
+-- test_entropy/
|   +-- test_system.py             # Correct byte count, always available
|   +-- test_timing.py             # Correct byte count, non-zero output
|   +-- test_mock.py               # Seeded reproducibility, bias simulation
|   +-- test_fallback.py           # Primary delegation, fallback trigger, error propagation
|   +-- test_registry.py           # Decorator registration, entry-point discovery, lazy loading
|   +-- test_quantum.py            # Mocked gRPC for 3 modes, circuit breaker, error mapping
|   +-- test_sham.py               # Sham QRNG source tests
|   +-- test_openentropy.py        # OpenEntropy source tests (mocked)
+-- test_logging/
|   +-- test_logger.py             # Record immutability, log levels, diagnostic mode, summary stats
+-- test_selection/
|   +-- test_selector.py           # CDF known values, top-k/top-p, edge cases
+-- test_temperature/
    +-- test_fixed.py              # Constant output, Shannon entropy computation
    +-- test_edt.py                # Monotonicity, clamping, exponent effects
+-- test_wire_format.py            # Protobuf wire format compatibility tests

deployments/                       # Production deployment configs per entropy source
+-- _template/                     # Starter template for new deployments
+-- openentropy/                   # OpenEntropy hardware entropy deployment
+-- urandom/                       # OS urandom (baseline) deployment

experiments/                       # YAML experiment presets with env var overrides
+-- baseline.yaml                  # No injection methods (control)
+-- logit_perturbation.yaml         # Logit perturbation only: per-logit quantum noise
+-- temp_modulation.yaml           # Temperature modulation only: quantum temperature modulation
+-- selection_drift.yaml           # Selection drift only: drift-based token selection
+-- combined.yaml                  # All three injection methods active
+-- min_p_filtering.yaml           # Min-P dynamic probability floor
+-- xtc_quantum.yaml               # XTC quantum top-token exclusion
+-- adaptive_injection.yaml        # Adaptive injection scaling

examples/
+-- servers/
|   +-- simple_urandom_server.py   # Minimal reference server (~50 lines of logic)
|   +-- qrng_template_server.py    # Annotated template with 3 TODO sections
+-- docker/
|   +-- Dockerfile.entropy-server  # Slim Python image for any example server
|   +-- docker-compose.yml         # Full stack: entropy-server + vLLM
+-- systemd/
    +-- qr-entropy-server.service  # systemd unit with restart-on-failure
    +-- qr-entropy-server.env      # Environment variables
```

## Architecture invariants -- DO NOT break these

1. **No hardcoded values.** Every numeric constant traces to a named field in `QRSamplerConfig` (pydantic-settings `BaseSettings` in `config.py`). Mathematical constants like `sqrt(2)` and `0.5 * (1 + erf(...))` are acceptable -- they are math, not configuration.

2. **Registry pattern for all strategies.** New `EntropySource`, `SignalAmplifier`, or `TemperatureStrategy` implementations are registered via class method decorators (`@AmplifierRegistry.register("name")`, `@TemperatureStrategyRegistry.register("name")`, `@register_entropy_source("name")`). The processor never instantiates strategies directly -- it goes through registry `.build()` methods. No if/else chains for strategy selection.

3. **ABCs define contracts.** `EntropySource` (in `entropy/base.py`), `SignalAmplifier` (in `amplification/base.py`), and `TemperatureStrategy` (in `temperature/base.py`) are ABCs. All concrete implementations must subclass them. The processor only references abstract types.

4. **FallbackEntropySource is a composition wrapper**, not a subclass of a specific source. It takes any `EntropySource` as primary and any as fallback. It only catches `EntropyUnavailableError` -- all other exceptions propagate.

5. **SEM is derived, never stored.** Standard error of mean = `population_std / sqrt(N)`. It is computed at amplification time from config fields. There is no `sem` config field.

6. **Frozen dataclasses for all result types.** `AmplificationResult`, `TemperatureResult`, `SelectionResult`, and `TokenSamplingRecord` use `@dataclass(frozen=True, slots=True)`. `QRSamplerConfig` is immutable via pydantic BaseSettings. Do not make these mutable.

7. **Per-request config resolution.** `resolve_config(defaults, extra_args)` creates a new config instance via `model_validate()` on a merged dict. It never mutates the default config. Infrastructure fields (`grpc_server_address`, `grpc_timeout_ms`, `grpc_retry_count`, `grpc_mode`, `fallback_mode`, `entropy_source_type`) are NOT overridable per-request. This is enforced by `_PER_REQUEST_FIELDS` frozenset in `config.py`.

8. **The processor forces one-hot logits.** After selecting a token, `apply()` sets the entire logit row to `-inf` except the selected token (set to `0.0`). This forces vLLM's downstream sampler to pick exactly that token.

9. **Logging uses `logging.getLogger("qr_sampler")`.** No `print()` statements anywhere in production code. All per-token logging goes through `SamplingLogger`.

10. **Just-in-time entropy.** Physical entropy generation occurs ONLY when `get_random_bytes()` is called -- after logits are available. No pre-buffering, no caching. The gRPC request is sent only when the processor needs bytes for a specific token.

11. **Entry-point auto-discovery for entropy sources.** Third-party packages register sources via the `qr_sampler.entropy_sources` entry-point group. The `EntropySourceRegistry` loads entry points lazily on first `get()` call. Built-in decorator registrations take precedence over entry points.

12. **Circuit breaker protects gRPC source.** `QuantumGrpcSource` tracks rolling P99 latency (deque, 100 samples), computes adaptive timeout = `max(5ms, P99 * 1.5)`, opens after 3 consecutive failures, enters half-open state after 10s.

13. **Pipeline-as-stages architecture.** The sampling pipeline is a list of `PipelineStage`
    instances. Each stage satisfies a `@runtime_checkable` protocol with `name: str` and
    `__call__(self, ctx: SamplingContext) -> None`. Stages are registered via
    `@StageRegistry.register("name")` and discoverable via the `qr_sampler.pipeline_stages`
    entry-point group. The processor never calls injection methods directly -- it iterates
    `for stage in self._pipeline: stage(ctx)`. The underlying injection utilities in
    `src/qr_sampler/injection/` remain stateless; all state lives in `SamplingContext.stage_state`
    (a `dict[str, Any]` that persists across `apply()` calls via `_RequestState`).

14. **`_RequestState` uses `stage_state` dict.** Per-request persistent state (e.g., selection
    drift position) is stored in `stage_state: dict[str, Any]`, keyed by convention as
    `"stage_name.field"` (e.g., `"selection_drift.position"`). The `drift_position` property on
    `_RequestState` is a convenience accessor that reads/writes `stage_state`.

## Coding conventions

- **Python 3.10+** -- use `X | Y` union syntax, not `Union[X, Y]`
- **Type hints** on all function signatures and return types
- **Docstrings** -- Google style on every public class and method
- **Imports** -- standard library first, third-party second, local third. No wildcard imports.
- **Line length** -- 100 characters (configured in `pyproject.toml` ruff section)
- **Errors** -- custom exception hierarchy rooted in `QRSamplerError` (in `exceptions.py`). Never catch bare `Exception` (health checks are the sole documented exception with `# noqa` comments).
- **No global mutable state** outside processor instances. Registries are populated at module import time and are effectively read-only after that.
- **No `print()`** -- use `logging` module with the `"qr_sampler"` logger
- **`QR_` prefix** for environment variables, `qr_` prefix for extra_args keys
- **Pydantic-settings BaseSettings** for configuration (not raw dataclasses)

## Key data flows

### Per-token sampling pipeline (in `processor.py` `_apply_row()`)

```
logits (torch.Tensor or numpy, one row per batch request)
  |
  +-> convert to numpy (zero-copy if CPU tensor)
  |
  +-> Build SamplingContext(row, config, entropy_source, amplifier, strategy, stage_state)
  |
  +-> for stage in pipeline:  stage(ctx)
  |      1. AdaptiveInjectionStage -- scale injection intensity by model uncertainty
  |      2. LogitPerturbationStage -- quantum-seeded Gaussian noise on logits
  |      3. DRYPenaltyStage        -- n-gram repetition penalty
  |      4. TopNSigmaStage         -- logit-space sigma filtering (pre-softmax)
  |      5. TemperatureStage       -- compute temperature + Shannon entropy
  |      6. TemperatureModulationStage -- quantum temperature modulation
  |      7. MinPStage              -- dynamic probability floor filtering
  |      8. XTCStage               -- quantum-driven top-token exclusion
  |      9. EntropyFetchStage      -- JIT entropy fetch + amplification -> ctx.u
  |     10. SelectionDriftStage    -- drift-based u replacement
  |     11. MirostatStage          -- Mirostat v2 adaptive perplexity control
  |     12. GumbelSelectionStage   -- Gumbel-Max trick for selection
  |     13. SelectionStage         -- CDF-based token selection (skipped if mirostat/gumbel active)
  |
  +-> Persist ctx.stage_state back to _RequestState
  |
  +-> Force one-hot logits: row = -inf everywhere, 0.0 at token_id
  |
  +-> SamplingLogger.log_token(TokenSamplingRecord)
```

Each stage reads/writes `SamplingContext` fields. Stages no-op when their config
parameter is disabled (e.g., `logit_perturbation_alpha <= 0`). The pipeline is a plain
`list[PipelineStage]` -- users can reorder, add, or remove stages.

### Config resolution flow

```
Environment variables (QR_*)
  -> QRSamplerConfig() -> pydantic-settings auto-loads from env + .env file

Per-request extra_args (qr_*)
  -> resolve_config(defaults, extra_args) -> new QRSamplerConfig instance
```

### Component construction flow (in processor.__init__)

```
QRSamplerConfig
  -> EntropySourceRegistry.get(config.entropy_source_type)
      -> source class, instantiated with config if constructor accepts it
      -> wrapped in FallbackEntropySource if fallback_mode != "error"
  -> AmplifierRegistry.build(config)
      -> ZScoreMeanAmplifier(config) (from registry by signal_amplifier_type)
  -> TemperatureStrategyRegistry.build(config, vocab_size)
      -> FixedTemperatureStrategy() or EDTTemperatureStrategy(vocab_size) (from registry)
```

### gRPC transport modes (in `entropy/quantum.py`)

```
Unary (grpc_mode="unary"):
  Client --EntropyRequest--> Server --EntropyResponse--> Client
  (one HTTP/2 stream per call, ~1-2ms overhead)

Server streaming (grpc_mode="server_streaming"):
  Client --EntropyRequest--> Server
  Server --EntropyResponse--> Client
  (short-lived stream, ~0.5-1ms)

Bidirectional (grpc_mode="bidi_streaming"):
  Client <--persistent stream--> Server
  (stream stays open for entire session, ~50-100us same-machine)
```

All modes use `grpc.aio` (asyncio) on a background thread with sync wrappers via `run_coroutine_threadsafe()`.

## How to add new components

### New signal amplifier

1. Create a class in `src/qr_sampler/amplification/` subclassing `SignalAmplifier`
2. Implement `amplify(self, raw_bytes: bytes) -> AmplificationResult`
3. Constructor takes `config: QRSamplerConfig` as first arg
4. Register: `@AmplifierRegistry.register("my_name")`
5. Use via config: `signal_amplifier_type = "my_name"` or `extra_args={"qr_signal_amplifier_type": "my_name"}`
6. Add tests in `tests/test_amplification/`

### New temperature strategy

1. Create a class in `src/qr_sampler/temperature/` subclassing `TemperatureStrategy`
2. Implement `compute_temperature(self, logits, config) -> TemperatureResult`
3. Always compute and return `shannon_entropy` even if not used in formula (logging depends on it)
4. Register: `@TemperatureStrategyRegistry.register("my_name")`
5. If the constructor needs `vocab_size`, accept it as first positional arg -- the registry detects this via try/except
6. Add tests in `tests/test_temperature/`

### New entropy source

1. Create a class in `src/qr_sampler/entropy/` subclassing `EntropySource`
2. Implement: `name` (property), `is_available` (property), `get_random_bytes(n)`, `close()`
3. Raise `EntropyUnavailableError` from `get_random_bytes()` if the source cannot provide bytes
4. Register: `@register_entropy_source("my_name")` (from `entropy.registry`)
5. Add entry point in `pyproject.toml` under `[project.entry-points."qr_sampler.entropy_sources"]`
6. Add tests in `tests/test_entropy/`

### New config field

1. Add the field to `QRSamplerConfig` in `config.py` with `Field(default=..., description=...)`
2. If per-request overridable, add the field name to `_PER_REQUEST_FIELDS` frozenset
3. The env var `QR_{FIELD_NAME_UPPER}` is automatically supported by pydantic-settings
4. The extra_args key `qr_{field_name}` is automatically supported by `resolve_config()`
5. Add tests in `tests/test_config.py`

### New pipeline stage

1. Create a class in `src/qr_sampler/stages/` with `name: str` attribute and `__call__(self, ctx: SamplingContext) -> None`
2. Register: `@StageRegistry.register("my_name")` (from `pipeline.registry`)
3. Stages read/write `SamplingContext` fields -- check `pipeline/context.py` for available fields
4. For persistent per-request state, use `ctx.stage_state["my_stage.field_name"]`
5. Add config fields to `QRSamplerConfig` if needed (and to `_PER_REQUEST_FIELDS` if per-request overridable)
6. Add entry point in `pyproject.toml` under `[project.entry-points."qr_sampler.pipeline_stages"]`
7. Export from `src/qr_sampler/stages/__init__.py`
8. Add tests in `tests/test_pipeline/` or `tests/test_injection/`
9. To include in the default pipeline, add to `build_default_pipeline()` in `stages/__init__.py`

## Testing approach

- **No real QRNG server or GPU needed.** Tests use `MockUniformSource` and numpy arrays (processor handles both torch tensors and numpy).
- **Dependency injection everywhere.** The processor accepts `vllm_config=None` for testing.
- **Shared test utilities** live in `tests/helpers.py` (not conftest -- it can't be imported as a module from subdirectories). Provides `make_processor()`, `assert_onehot()`, `register_request()`, mock objects (`MockVllmConfig`, `MockSamplingParams`, etc.), and constants (`SAMPLE_LOGITS`, `VOCAB_SIZE`).
- **Pipeline tests** in `tests/test_pipeline/` verify protocol compliance, registry, default pipeline, custom pipelines, and stage_state persistence.
- **Statistical tests** in `test_statistical_properties.py` require `scipy` (dev dependency). They validate mathematical properties: KS-test for u-value uniformity, bias detection, EDT monotonicity.
- **Frozen dataclass tests** verify immutability of all result types.
- **Edge case coverage** is thorough: empty inputs, single-token vocab, all-identical logits, all-inf-except-one logits, zero temperature.
- **gRPC tests** mock the gRPC channel/stub to test all 3 transport modes and the circuit breaker without a real server.

## Proto stubs

The files in `src/qr_sampler/proto/` are hand-written minimal stubs (not generated by `protoc`). They define just enough for the gRPC client and example servers to work:

- `entropy_service.proto` -- the canonical protocol definition
- `entropy_service_pb2.py` -- `EntropyRequest` and `EntropyResponse` message classes
- `entropy_service_pb2_grpc.py` -- `EntropyServiceStub` (client) and `EntropyServiceServicer` (server base) + `add_EntropyServiceServicer_to_server()`

If the proto definition changes, these stubs must be updated manually or regenerated with `grpc_tools.protoc`.

## Dependencies

- **Runtime:** `numpy>=1.24.0`, `pydantic>=2.0.0`, `pydantic-settings>=2.0.0`, `grpcio>=1.60.0`, `protobuf>=4.21.0`
- **Dev:** `pytest>=7.0`, `pytest-cov>=4.0`, `scipy>=1.10.0`, `ruff>=0.4.0`, `mypy>=1.8.0`, `pre-commit>=3.0`, `bandit>=1.7.0`
- **Implicit:** vLLM V1 (provides `LogitsProcessor` base class, `torch`). Not listed as a dependency since the plugin runs inside vLLM's process.
