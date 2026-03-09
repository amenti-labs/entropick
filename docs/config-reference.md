# Config Reference

This is the exhaustive configuration matrix for entropick.

If you are new to the repo, start with the shorter guidance in [README.md](../README.md). That file covers the normal user path. This document is for the full variable list.

All configuration uses the `QR_` environment variable prefix. Per-request overrides use `qr_` in `extra_args`.

## Infrastructure fields

These are process-wide settings. They control where entropy comes from, how the server reaches it, and how much operational visibility or fault tolerance you want.

### 1. Pick an entropy backend

| Variable | Default | When you use it |
|----------|---------|-----------------|
| `QR_ENTROPY_SOURCE_TYPE` | `system` | Pick the backend: `system` for simplest local use, `quantum_grpc` for a remote/local gRPC entropy service, `openentropy` for native hardware noise, `mock_uniform` for tests, `sham_qrng` for controls. |
| `QR_FALLBACK_MODE` | `system` | Decide what happens when the primary source fails: hard error, fall back to system entropy, or fall back to mock entropy. |
| `QR_SHAM_QRNG_LATENCY_MS` | `0.0` | Only for `sham_qrng`. Adds fake device latency so sham runs look like real QRNG runs. |

### 2. gRPC entropy server

Ignore this group unless `QR_ENTROPY_SOURCE_TYPE=quantum_grpc`.

| Variable | Default | What it controls |
|----------|---------|------------------|
| `QR_GRPC_SERVER_ADDRESS` | `localhost:50051` | Where the entropy server lives. Usually the only gRPC field you must set. |
| `QR_GRPC_MODE` | `unary` | Transport style. Start with `unary`; switch to streaming only when chasing latency. |
| `QR_GRPC_TIMEOUT_MS` | `5000` | Request timeout for entropy fetches. Increase only for slow hardware or slow networks. |
| `QR_GRPC_RETRY_COUNT` | `2` | How many times to retry after a fetch failure. |
| `QR_GRPC_METHOD_PATH` | `/qr_entropy.EntropyService/GetEntropy` | Unary RPC path. Change only when your server uses a different proto service or method name. |
| `QR_GRPC_STREAM_METHOD_PATH` | `/qr_entropy.EntropyService/StreamEntropy` | Streaming RPC path. Needed only for `server_streaming` or `bidi_streaming`. |
| `QR_GRPC_API_KEY` | *(empty)* | Optional auth token sent to the server. |
| `QR_GRPC_API_KEY_HEADER` | `api-key` | Metadata header name for the API key. |

### 3. Optional gRPC TLS / mTLS

Ignore this unless your gRPC server requires TLS.

| Variable | Default | What it controls |
|----------|---------|------------------|
| `QR_GRPC_TLS_ENABLED` | `false` | Turns on secure gRPC channels. |
| `QR_GRPC_TLS_CA_CERT` | *(empty)* | CA certificate used to verify the server certificate. |
| `QR_GRPC_TLS_CLIENT_CERT` | *(empty)* | Client certificate for mTLS. |
| `QR_GRPC_TLS_CLIENT_KEY` | *(empty)* | Client private key for mTLS. |

### 4. OpenEntropy

Ignore this group unless `QR_ENTROPY_SOURCE_TYPE=openentropy`.

| Variable | Default | What it controls |
|----------|---------|------------------|
| `QR_OE_SOURCES` | *(empty)* | Which OpenEntropy source(s) to use. Set a specific source such as `clock_jitter` when you want predictable behavior. |
| `QR_OE_CONDITIONING` | `raw` | Post-processing mode. `sha256` is the safest default for general use; `raw` is mainly for research. |
| `QR_OE_PARALLEL` | `true` | Whether OpenEntropy should collect from multiple sources in parallel. |
| `QR_OE_TIMEOUT` | `5.0` | Collection timeout in seconds. Increase only if the selected sources are slow. |

### 5. Amplifier calibration and reliability

These are advanced operational knobs.

| Variable | Default | What it controls |
|----------|---------|------------------|
| `QR_ECDF_CALIBRATION_SAMPLES` | `2000` | Only used by the `ecdf` amplifier. Controls how many calibration samples are collected up front. |
| `QR_CB_WINDOW_SIZE` | `100` | Rolling latency window size for the gRPC circuit breaker. |
| `QR_CB_MIN_TIMEOUT_MS` | `5.0` | Floor for adaptive gRPC timeouts. |
| `QR_CB_TIMEOUT_MULTIPLIER` | `1.5` | Multiplier applied to observed P99 latency. |
| `QR_CB_RECOVERY_WINDOW_S` | `10.0` | How long the circuit stays open before a retry is allowed. |
| `QR_CB_MAX_CONSECUTIVE_FAILURES` | `3` | Failures required before the circuit opens. |

### 6. Logging and diagnostics

These affect visibility and memory use, not the actual sampling algorithm.

| Variable | Default | What it controls |
|----------|---------|------------------|
| `QR_LOG_LEVEL` | `summary` | How much per-token information gets logged. Use `full` only when debugging. |
| `QR_DIAGNOSTIC_MODE` | `false` | Stores token-level records in memory for later analysis. Useful for experiments, unnecessary for ordinary serving. |

## Sampling parameters

These may be set process-wide with `QR_*` env vars or overridden per request with `extra_args`.

The mapping rule is:

- Environment variable: `QR_FIXED_TEMPERATURE`
- Request key: `qr_fixed_temperature`

### 1. Baseline sampler

| Variable | `extra_args` key | Default | Why it matters |
|----------|-----------------|---------|----------------|
| `QR_TEMPERATURE_STRATEGY` | `qr_temperature_strategy` | `fixed` | Chooses whether temperature is constant (`fixed`) or entropy-dependent (`edt`). |
| `QR_FIXED_TEMPERATURE` | `qr_fixed_temperature` | `0.7` | Baseline temperature when using the standard fixed strategy. |
| `QR_TOP_K` | `qr_top_k` | `0` | Limits selection to the top-k tokens. `0` disables it. |
| `QR_TOP_P` | `qr_top_p` | `1.0` | Nucleus sampling threshold. `1.0` disables it. |
| `QR_SAMPLE_COUNT` | `qr_sample_count` | `20480` | How many entropy bytes are consumed per token. Higher values amplify tiny biases but cost more latency. |
| `QR_SIGNAL_AMPLIFIER_TYPE` | `qr_signal_amplifier_type` | `zscore_mean` | How raw entropy bytes are converted into a uniform `u` for token selection. |

### 2. Amplifier math

| Variable | `extra_args` key | Default | Why it matters |
|----------|-----------------|---------|----------------|
| `QR_POPULATION_MEAN` | `qr_population_mean` | `127.5` | Null-hypothesis byte mean used by the z-score amplifier. |
| `QR_POPULATION_STD` | `qr_population_std` | `73.612...` | Null-hypothesis byte standard deviation used by the z-score amplifier. |
| `QR_UNIFORM_CLAMP_EPSILON` | `qr_uniform_clamp_epsilon` | `1e-10` | Prevents `u` from landing exactly at 0 or 1. Mostly a numerical-safety knob. |

### 3. Entropy-dependent temperature

Only relevant if `QR_TEMPERATURE_STRATEGY=edt`.

| Variable | `extra_args` key | Default | Why it matters |
|----------|-----------------|---------|----------------|
| `QR_EDT_BASE_TEMP` | `qr_edt_base_temp` | `0.8` | Base coefficient for entropy-dependent temperature. |
| `QR_EDT_EXPONENT` | `qr_edt_exponent` | `0.5` | How strongly temperature responds to entropy. |
| `QR_EDT_MIN_TEMP` | `qr_edt_min_temp` | `0.1` | Lower clamp for EDT temperature. |
| `QR_EDT_MAX_TEMP` | `qr_edt_max_temp` | `2.0` | Upper clamp for EDT temperature. |

### 4. Candidate filters and repetition control

| Variable | `extra_args` key | Default | Why it matters |
|----------|-----------------|---------|----------------|
| `QR_MIN_P` | `qr_min_p` | `0.0` | Drops tokens whose probability is too small relative to the current best token. |
| `QR_TOP_N_SIGMA` | `qr_top_n_sigma` | `0.0` | Keeps only logits that are within N standard deviations of the maximum. |
| `QR_DRY_MULTIPLIER` | `qr_dry_multiplier` | `0.0` | Enables the DRY repetition penalty. |
| `QR_DRY_BASE` | `qr_dry_base` | `1.75` | Exponential base for the DRY penalty. |
| `QR_DRY_ALLOWED_LENGTH` | `qr_dry_allowed_length` | `2` | Minimum repeated sequence length before DRY applies. |
| `QR_DRY_PENALTY_LAST_N` | `qr_dry_penalty_last_n` | `-1` | Limits how much context DRY looks at. `-1` means full context. |
| `QR_DRY_SEQUENCE_BREAKERS` | `qr_dry_sequence_breakers` | *(empty)* | Token IDs that reset DRY matching. |

### 5. Alternative selectors

These replace the default CDF selector.

| Variable | `extra_args` key | Default | Why it matters |
|----------|-----------------|---------|----------------|
| `QR_MIROSTAT_MODE` | `qr_mirostat_mode` | `0` | Enables Mirostat v2 when set to `2`. |
| `QR_MIROSTAT_TAU` | `qr_mirostat_tau` | `5.0` | Target surprise rate for Mirostat. |
| `QR_MIROSTAT_ETA` | `qr_mirostat_eta` | `0.1` | Learning rate for Mirostat. |
| `QR_GUMBEL_SELECTION` | `qr_gumbel_selection` | `false` | Switches from CDF sampling to Gumbel-Max selection. |

### 6. Experimental entropy injection stages

These are the most research-oriented knobs.

| Variable | `extra_args` key | Default | Why it matters |
|----------|-----------------|---------|----------------|
| `QR_XTC_PROBABILITY` | `qr_xtc_probability` | `0.0` | Probability of excluding top tokens via XTC. |
| `QR_XTC_THRESHOLD` | `qr_xtc_threshold` | `0.1` | Minimum probability for a token to become an XTC candidate. |
| `QR_ADAPTIVE_INJECTION` | `qr_adaptive_injection` | `false` | Scales injection intensity based on model entropy. |
| `QR_ADAPTIVE_INJECTION_LOW_H` | `qr_adaptive_injection_low_h` | `1.0` | Lower entropy threshold for adaptive injection. |
| `QR_ADAPTIVE_INJECTION_HIGH_H` | `qr_adaptive_injection_high_h` | `3.0` | Upper entropy threshold for adaptive injection. |
| `QR_LOGIT_PERTURBATION_ALPHA` | `qr_logit_perturbation_alpha` | `0.0` | Strength of direct logit noise injection. |
| `QR_LOGIT_PERTURBATION_SIGMA` | `qr_logit_perturbation_sigma` | `1.0` | Standard deviation of the injected logit noise before scaling. |
| `QR_TEMP_MODULATION_BETA` | `qr_temp_modulation_beta` | `0.0` | Strength of per-token temperature modulation. |
| `QR_DRIFT_STEP` | `qr_drift_step` | `0.0` | Step size for persistent selection drift across a request. |
| `QR_DRIFT_INITIAL_POSITION` | `qr_drift_initial_position` | `0.5` | Starting drift position in `[0, 1)`. |
| `QR_INJECTION_VERBOSE` | `qr_injection_verbose` | `false` | Extra debug logging for injection stages. |
