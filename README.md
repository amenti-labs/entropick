# qr-sampler

**Plug any randomness source into LLM token sampling via vLLM.**

qr-sampler is a [vLLM V1](https://github.com/vllm-project/vllm) LogitsProcessor plugin that replaces standard pseudorandom token sampling with entropy from external sources — quantum random number generators (QRNGs), hardware noise via [OpenEntropy](https://github.com/amenti-labs/openentropy), processor timing jitter, or any device you connect via gRPC. It provides a modular pipeline of stages that can perturb logits, modulate temperature, filter candidates, and select tokens — all driven by physical randomness.

> **Fork notice:** This project is forked from [alchemystack/Quantum-random-vLLM-sampler](https://github.com/alchemystack/Quantum-random-vLLM-sampler). Original work by [alchemystack](https://github.com/alchemystack). This fork extends the project with a pipeline-as-stages architecture, additional injection methods (Min-P, XTC, Adaptive Injection), and OpenEntropy integration.

```
pip install qr-sampler
```

---

## Why qr-sampler?

Standard LLM inference uses pseudorandom number generators (PRNGs) for token sampling. PRNGs are deterministic — given the same seed, they produce the same output every time. qr-sampler replaces this with *true* randomness from physical processes:

- **Quantum RNGs** — photon detectors, vacuum fluctuation devices, or any hardware QRNG over gRPC
- **OpenEntropy** — 58+ hardware noise sources on the local machine (thermal, timing, microarchitectural) — no network needed
- **Processor timing jitter** — CPU clock variations as an entropy source (experimental)
- **Your own source** — implement the `EntropySource` ABC or connect any hardware via the gRPC protocol
- **OS entropy** — `/dev/urandom` as a fallback or baseline

### Consciousness-research context

qr-sampler provides infrastructure for studying whether conscious intent can influence quantum-random processes in LLM token selection. The signal amplification system converts thousands of random bytes into a single token choice, designed so that even a tiny statistical bias (e.g., 0.1% shift in byte means) produces a measurable effect on which token gets selected. All entropy is generated **just-in-time** — the quantum measurement happens *after* logits are computed, never before.

This is a research tool. It makes no claims about consciousness or quantum mechanics — it provides the infrastructure to run rigorous experiments.

---

## How it works

qr-sampler processes logits through a **pipeline of 9 stages**, each operating on a shared `SamplingContext`. Stages are modular — disable any by setting its config to zero/false.

```
Logits from vLLM (one row per batch request)
  │
  ├─ Adaptive Injection ────── Compute injection_scale from Shannon entropy H
  │   (scales M1/M2/M3)          of the logit distribution
  │
  ├─ M1: Logit Noise ──────── Add per-logit quantum Gaussian noise
  │   (α * σ * probit(qbytes))    fetches vocab_size×4 quantum bytes
  │
  ├─ Temperature ───────────── Compute temperature via strategy
  │   (fixed or EDT)              from the logit distribution
  │
  ├─ M2: Temp Variance ────── Modulate temperature with quantum entropy
  │   (T × (1 + β(u-0.5)))       per-token quantum modulation
  │
  ├─ Min-P ─────────────────── Remove tokens where p < min_p × max(p)
  │   (dynamic floor)             confidence-adaptive filtering
  │
  ├─ XTC ───────────────────── Quantum coin-flip exclusion of top tokens
  │   (exclude top choices)       binary decisions from quantum bytes
  │
  ├─ Entropy Fetch ─────────── Fetch fresh random bytes + amplify to u∈(0,1)
  │   (20,480 bytes → z-score)    just-in-time, after logits exist
  │
  ├─ M3: Correlated Walk ──── Drift selection point with temporal memory
  │   (walk_position ± step)      creates correlations across tokens
  │
  └─ Selection ─────────────── top-k → softmax → top-p → CDF → select token
      (force one-hot logits)      vLLM picks exactly this token
```

The processor registers via Python entry points — no vLLM source code modifications needed.

---

## Injection methods and filters

All methods are disabled by default (control parameter = 0 or false). Enable individually or combine.

### M1: Logit Noise (`qr_logit_noise_alpha`)

Adds per-logit Gaussian noise derived from quantum entropy. Fetches `vocab_size × 4` bytes, maps to zero-mean noise via the probit transform, scales by `α × σ`.

| Parameter | Default | Effect |
|-----------|---------|--------|
| `qr_logit_noise_alpha` | `0.0` | Noise magnitude (0 = disabled) |
| `qr_logit_noise_sigma` | `1.0` | Gaussian std before alpha scaling |

**Impact:** Higher α broadens the effective probability distribution, increasing diversity. At α=1.0 with OpenEntropy, mean token rank shifts from ~3.1 to ~2.0.

### M2: Temperature Variance (`qr_temp_variance_beta`)

Modulates temperature per-token using quantum entropy: `T_new = T × (1 + β × (u - 0.5))`.

| Parameter | Default | Effect |
|-----------|---------|--------|
| `qr_temp_variance_beta` | `0.0` | Modulation magnitude (0 = disabled) |

**Impact:** β=0.5 varies temperature ±12%; β=1.5 varies ±37% (range [0.18, 1.08] at base temp 0.7).

### M3: Correlated Walk (`qr_walk_step`)

Maintains a per-request walk position that drifts based on quantum entropy, replacing the amplified `u` value. Creates temporal correlations across tokens within a request.

| Parameter | Default | Effect |
|-----------|---------|--------|
| `qr_walk_step` | `0.0` | Step size (0 = disabled) |
| `qr_walk_initial_position` | `0.5` | Starting position in [0, 1) |

**Impact:** step=0.05 gives mean |Δu|=0.014 (vs ~0.3 IID baseline) — strong temporal coherence. Token selections drift smoothly rather than jumping randomly.

### Min-P Filter (`qr_min_p`)

Dynamic probability floor — removes tokens where `p < min_p × max(p)`. Adapts to model confidence: when the model is sure, fewer tokens survive; when uncertain, more pass through.

| Parameter | Default | Effect |
|-----------|---------|--------|
| `qr_min_p` | `0.0` | Threshold (0 = disabled, 1.0 = top token only) |

**Impact:** min_p=0.1 reduces candidates from 32 to ~5; min_p=0.5 locks to the top 3 tokens. No extra entropy cost.

### XTC: Exclude Top Choices (`qr_xtc_probability`)

Probabilistically excludes top tokens using quantum random bytes. Each token above the probability threshold gets an independent quantum coin flip for exclusion. Aligns with the PEAR consciousness-research paradigm (binary quantum decisions).

| Parameter | Default | Effect |
|-----------|---------|--------|
| `qr_xtc_probability` | `0.0` | Exclusion probability per token (0 = disabled) |
| `qr_xtc_threshold` | `0.1` | Only tokens with p ≥ threshold are candidates |

**Impact:** p=1.0/t=0.05 aggressively excludes top tokens — rank-0 drops to 5%, mean rank jumps to 9.2. Always keeps at least one token.

### Adaptive Injection (`qr_adaptive_injection`)

Scales all injection methods (M1/M2/M3) by the Shannon entropy H of the logit distribution. When the model is confident (low H), injection is suppressed; when uncertain (high H), full injection runs.

| Parameter | Default | Effect |
|-----------|---------|--------|
| `qr_adaptive_injection` | `false` | Enable/disable |
| `qr_adaptive_injection_low_h` | `1.0` | H below this → scale=0 (nats) |
| `qr_adaptive_injection_high_h` | `3.0` | H above this → scale=1 (nats) |

**Formula:** `scale = clamp((H - low_h) / (high_h - low_h), 0, 1)`

---

## Quick start

### Docker with an external entropy source (recommended)

Each entropy source has a self-contained deployment profile under `deployments/`. Pick the one that matches your setup:

| Profile | Entropy source | Description |
|---------|---------------|-------------|
| [`urandom/`](deployments/urandom/) | `os.urandom()` via gRPC | Local gRPC server for testing the full pipeline. **Start here.** |
| [`firefly-1/`](deployments/firefly-1/) | Quantum RNG via gRPC | External QRNG server with API key auth. |
| [`openentropy/`](deployments/openentropy/) | Local hardware noise | 58+ sources, no network. |
| [`_template/`](deployments/_template/) | Your hardware | Copy and customize for your own entropy source. |

#### 1. Choose a profile and configure

```bash
cd deployments/urandom
cp .env.example .env
# Edit .env — set HF_TOKEN if using a gated model
```

#### 2. Launch

```bash
docker compose up --build
```

This builds a vLLM image with qr-sampler baked in, starts any required entropy server containers, and connects everything automatically.

#### 3. Send a request

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "prompt": "The nature of consciousness is",
    "max_tokens": 100
  }'
```

To connect your own QRNG hardware, copy the template and follow the [Setting up your own entropy source](#setting-up-your-own-entropy-source) guide:

```bash
cp -r deployments/_template deployments/my-qrng
# Edit deployments/my-qrng/.env and deployments/my-qrng/docker-compose.yml
```

See [deployments/README.md](deployments/README.md) for the full guide.

### Bare-metal install (without Docker)

```bash
# Install qr-sampler (includes gRPC support)
pip install qr-sampler

# Start vLLM — qr-sampler registers automatically via entry points
vllm serve Qwen/Qwen2.5-1.5B-Instruct --dtype half --max-model-len 8096 --gpu-memory-utilization 0.80
```

Configure the entropy source via environment variables:

```bash
export QR_ENTROPY_SOURCE_TYPE=quantum_grpc
export QR_GRPC_SERVER_ADDRESS=localhost:50051
vllm serve Qwen/Qwen2.5-1.5B-Instruct --dtype half --max-model-len 8096 --gpu-memory-utilization 0.80
```

### OpenEntropy (no network, no server)

```bash
pip install openentropy

export QR_ENTROPY_SOURCE_TYPE=openentropy
export QR_OE_SOURCES=clock_jitter          # fast single source (~1-2ms/call)
export QR_OE_CONDITIONING=sha256           # prevents bias from XOR cancellation
export QR_FALLBACK_MODE=system
vllm serve Qwen/Qwen2.5-1.5B-Instruct --dtype half --max-model-len 8096 --gpu-memory-utilization 0.80
```

### System entropy fallback

Without an external entropy source, qr-sampler falls back to `os.urandom()`. This is useful for development and testing but does not provide the quantum randomness needed for consciousness-research experiments. To use system entropy, set `QR_ENTROPY_SOURCE_TYPE=system` (this is the default).

---

## Per-request parameter overrides

Override sampling parameters on individual requests via `extra_args`:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "prompt": "The nature of consciousness is",
    "max_tokens": 100,
    "extra_args": {
      "qr_temperature_strategy": "edt",
      "qr_top_k": 100,
      "qr_top_p": 0.95,
      "qr_min_p": 0.1,
      "qr_xtc_probability": 0.5,
      "qr_xtc_threshold": 0.1,
      "qr_adaptive_injection": true,
      "qr_logit_noise_alpha": 0.3,
      "qr_temp_variance_beta": 0.5,
      "qr_walk_step": 0.05,
      "qr_diagnostic_mode": true
    }
  }'
```

Only fields listed in the **Sampling parameters** table are per-request overridable. Infrastructure fields (for example `QR_GRPC_SERVER_ADDRESS`, `QR_GRPC_METHOD_PATH`, `QR_GRPC_API_KEY`) are process-level settings and cannot be overridden per request.

### Example: running each mode

Set helper variables for `curl` examples:

```bash
export VLLM_URL=http://localhost:8000/v1/completions
export MODEL=Qwen/Qwen2.5-1.5B-Instruct
```

**Baseline (no injection):**

```bash
curl "$VLLM_URL" -H "Content-Type: application/json" -d '{
  "model": "'"$MODEL"'",
  "prompt": "The nature of consciousness is",
  "max_tokens": 100
}'
```

**M1 — Logit Noise:**

```bash
curl "$VLLM_URL" -H "Content-Type: application/json" -d '{
  "model": "'"$MODEL"'",
  "prompt": "The nature of consciousness is",
  "max_tokens": 100,
  "extra_args": { "qr_logit_noise_alpha": 0.3 }
}'
```

**M2 — Temperature Variance:**

```bash
curl "$VLLM_URL" -H "Content-Type: application/json" -d '{
  "model": "'"$MODEL"'",
  "prompt": "The nature of consciousness is",
  "max_tokens": 100,
  "extra_args": { "qr_temp_variance_beta": 0.5 }
}'
```

**M3 — Correlated Walk:**

```bash
curl "$VLLM_URL" -H "Content-Type: application/json" -d '{
  "model": "'"$MODEL"'",
  "prompt": "The nature of consciousness is",
  "max_tokens": 100,
  "extra_args": { "qr_walk_step": 0.05 }
}'
```

**Min-P filtering:**

```bash
curl "$VLLM_URL" -H "Content-Type: application/json" -d '{
  "model": "'"$MODEL"'",
  "prompt": "The nature of consciousness is",
  "max_tokens": 100,
  "extra_args": { "qr_min_p": 0.1 }
}'
```

**XTC — Exclude Top Choices:**

```bash
curl "$VLLM_URL" -H "Content-Type: application/json" -d '{
  "model": "'"$MODEL"'",
  "prompt": "The nature of consciousness is",
  "max_tokens": 100,
  "extra_args": { "qr_xtc_probability": 0.5, "qr_xtc_threshold": 0.1 }
}'
```

**Adaptive Injection + M1:**

```bash
curl "$VLLM_URL" -H "Content-Type: application/json" -d '{
  "model": "'"$MODEL"'",
  "prompt": "The nature of consciousness is",
  "max_tokens": 100,
  "extra_args": {
    "qr_adaptive_injection": true,
    "qr_adaptive_injection_low_h": 1.0,
    "qr_adaptive_injection_high_h": 3.0,
    "qr_logit_noise_alpha": 0.5
  }
}'
```

**All modes combined:**

```bash
curl "$VLLM_URL" -H "Content-Type: application/json" -d '{
  "model": "'"$MODEL"'",
  "prompt": "The nature of consciousness is",
  "max_tokens": 100,
  "extra_args": {
    "qr_adaptive_injection": true,
    "qr_logit_noise_alpha": 0.3,
    "qr_temp_variance_beta": 0.5,
    "qr_walk_step": 0.05,
    "qr_min_p": 0.1,
    "qr_xtc_probability": 0.3,
    "qr_xtc_threshold": 0.1
  }
}'
```

To set methods process-wide (instead of per request):

```bash
export QR_LOGIT_NOISE_ALPHA=0.3
export QR_TEMP_VARIANCE_BETA=0.5
export QR_WALK_STEP=0.05
export QR_MIN_P=0.1
export QR_XTC_PROBABILITY=0.3
export QR_XTC_THRESHOLD=0.1
export QR_ADAPTIVE_INJECTION=true
```

Enable injection diagnostics:

```bash
export QR_INJECTION_VERBOSE=true
export QR_LOG_LEVEL=full
```

To disable an injection method, set its control value back to `0.0` (or `false` for adaptive injection).

---

## Web UI

qr-sampler works with [Open WebUI](https://github.com/open-webui/open-webui), a self-hosted ChatGPT-style interface that connects to vLLM's OpenAI-compatible API. Every deployment profile includes it as an optional service — add `--profile ui` to start it alongside vLLM:

```bash
cd deployments/urandom
docker compose --profile ui up --build
```

Then open http://localhost:3000 to start chatting. Without `--profile ui`, Open WebUI does not start and nothing changes.

### Controlling qr-sampler from the UI

A pre-built [filter function](examples/open-webui/) injects qr-sampler per-request parameters into every chat message via the Open WebUI Valves system. This lets you adjust temperature, top-k, top-p, injection methods, and other sampling parameters from the admin panel without editing environment variables or writing API calls.

To set it up:

1. Go to **Admin Panel > Functions** in Open WebUI.
2. Click **Import** and select [`examples/open-webui/qr_sampler_filter.json`](examples/open-webui/qr_sampler_filter.json).
3. Toggle the function to **Global**.
4. Click the **gear icon** to adjust parameters.

See [`examples/open-webui/README.md`](examples/open-webui/README.md) for the full guide.

> Open WebUI is entirely optional. qr-sampler works the same way with direct API calls, `curl`, Python clients, or any OpenAI-compatible tool.

---

## Configuration reference

All configuration is done via environment variables with the `QR_` prefix. Per-request overrides use the `qr_` prefix in `extra_args`.

### Infrastructure fields (NOT per-request overridable)

| Environment variable | Default | Description |
|---|---|---|
| `QR_ENTROPY_SOURCE_TYPE` | `system` | Primary entropy source: `system`, `quantum_grpc`, `openentropy`, `timing_noise`, `mock_uniform` |
| `QR_GRPC_SERVER_ADDRESS` | `localhost:50051` | gRPC entropy server address (`host:port` or `unix:///path`) |
| `QR_GRPC_TIMEOUT_MS` | `5000` | gRPC call timeout in milliseconds |
| `QR_GRPC_RETRY_COUNT` | `2` | Retry attempts after gRPC failure |
| `QR_GRPC_MODE` | `unary` | Transport mode: `unary`, `server_streaming`, `bidi_streaming` |
| `QR_GRPC_METHOD_PATH` | `/qr_entropy.EntropyService/GetEntropy` | gRPC method path for unary RPC |
| `QR_GRPC_STREAM_METHOD_PATH` | `/qr_entropy.EntropyService/StreamEntropy` | gRPC method path for streaming RPC (empty disables streaming) |
| `QR_GRPC_API_KEY` | *(empty)* | API key sent via gRPC metadata (empty = no auth) |
| `QR_GRPC_API_KEY_HEADER` | `api-key` | gRPC metadata header name for the API key |
| `QR_FALLBACK_MODE` | `system` | Fallback when primary fails: `error`, `system`, `mock_uniform` |
| `QR_OE_SOURCES` | *(empty)* | Comma-separated OpenEntropy source names (empty = all) |
| `QR_OE_PARALLEL` | `true` | Collect OpenEntropy sources in parallel |
| `QR_OE_TIMEOUT` | `5.0` | OpenEntropy collection timeout in seconds |
| `QR_CB_WINDOW_SIZE` | `100` | Rolling latency window size for P99 computation |
| `QR_CB_MIN_TIMEOUT_MS` | `5.0` | Minimum adaptive timeout in milliseconds |
| `QR_CB_TIMEOUT_MULTIPLIER` | `1.5` | Multiplier applied to P99 latency for adaptive timeout |
| `QR_CB_RECOVERY_WINDOW_S` | `10.0` | Seconds before half-open retry after circuit opens |
| `QR_CB_MAX_CONSECUTIVE_FAILURES` | `3` | Consecutive failures before circuit breaker opens |

### Sampling parameters (per-request overridable)

| Environment variable | extra_args key | Default | Description |
|---|---|---|---|
| `QR_SIGNAL_AMPLIFIER_TYPE` | `qr_signal_amplifier_type` | `zscore_mean` | Signal amplification algorithm |
| `QR_SAMPLE_COUNT` | `qr_sample_count` | `20480` | Entropy bytes fetched per token |
| `QR_POPULATION_MEAN` | `qr_population_mean` | `127.5` | Null-hypothesis mean for byte values |
| `QR_POPULATION_STD` | `qr_population_std` | `73.612...` | Population std for uniform [0, 255] |
| `QR_UNIFORM_CLAMP_EPSILON` | `qr_uniform_clamp_epsilon` | `1e-10` | Clamp u to avoid degenerate CDF |
| `QR_TEMPERATURE_STRATEGY` | `qr_temperature_strategy` | `fixed` | Strategy: `fixed` or `edt` |
| `QR_FIXED_TEMPERATURE` | `qr_fixed_temperature` | `0.7` | Constant temperature (fixed strategy) |
| `QR_EDT_BASE_TEMP` | `qr_edt_base_temp` | `0.8` | Base coefficient for EDT |
| `QR_EDT_EXPONENT` | `qr_edt_exponent` | `0.5` | Power-law exponent for EDT |
| `QR_EDT_MIN_TEMP` | `qr_edt_min_temp` | `0.1` | EDT temperature floor |
| `QR_EDT_MAX_TEMP` | `qr_edt_max_temp` | `2.0` | EDT temperature ceiling |
| `QR_TOP_K` | `qr_top_k` | `0` | Top-k filtering (`<=0` disables) |
| `QR_TOP_P` | `qr_top_p` | `1.0` | Nucleus sampling threshold (`1.0` disables) |
| `QR_MIN_P` | `qr_min_p` | `0.0` | Min-P threshold: remove tokens with p < min_p × max(p) |
| `QR_XTC_PROBABILITY` | `qr_xtc_probability` | `0.0` | XTC: exclusion probability per top token |
| `QR_XTC_THRESHOLD` | `qr_xtc_threshold` | `0.1` | XTC: minimum probability to be an exclusion candidate |
| `QR_ADAPTIVE_INJECTION` | `qr_adaptive_injection` | `false` | Scale injection by distribution entropy |
| `QR_ADAPTIVE_INJECTION_LOW_H` | `qr_adaptive_injection_low_h` | `1.0` | Entropy below this → injection suppressed (nats) |
| `QR_ADAPTIVE_INJECTION_HIGH_H` | `qr_adaptive_injection_high_h` | `3.0` | Entropy above this → full injection (nats) |
| `QR_LOGIT_NOISE_ALPHA` | `qr_logit_noise_alpha` | `0.0` | M1: Logit noise magnitude (`0` disables) |
| `QR_LOGIT_NOISE_SIGMA` | `qr_logit_noise_sigma` | `1.0` | M1: Gaussian std dev before alpha scaling |
| `QR_TEMP_VARIANCE_BETA` | `qr_temp_variance_beta` | `0.0` | M2: Temperature modulation magnitude (`0` disables) |
| `QR_WALK_STEP` | `qr_walk_step` | `0.0` | M3: Correlated walk step size (`0` disables) |
| `QR_WALK_INITIAL_POSITION` | `qr_walk_initial_position` | `0.5` | M3: Initial walk position in `[0, 1)` |
| `QR_OE_CONDITIONING` | `qr_oe_conditioning` | `raw` | OpenEntropy conditioning: `raw`, `sha256`, `vonneumann` |
| `QR_INJECTION_VERBOSE` | `qr_injection_verbose` | `false` | Log per-token injection diagnostics at debug level |
| `QR_LOG_LEVEL` | `qr_log_level` | `summary` | Logging: `none`, `summary`, `full` |
| `QR_DIAGNOSTIC_MODE` | `qr_diagnostic_mode` | `false` | Store all token records in memory |

You can also use a `.env` file — pydantic-settings loads it automatically.

---

## Entropy sources

### Built-in sources

| Source | Identifier | Description |
|---|---|---|
| **System** | `system` | `os.urandom()` — OS cryptographic RNG (default/fallback) |
| **Quantum gRPC** | `quantum_grpc` | Remote entropy server via gRPC (any protocol) |
| **OpenEntropy** | `openentropy` | 58+ hardware noise sources (thermal, timing, microarch) — local, no network |
| **Timing noise** | `timing_noise` | CPU timing jitter (experimental) |
| **Mock uniform** | `mock_uniform` | Configurable test source with seed/bias |

### OpenEntropy

[OpenEntropy](https://github.com/amenti-labs/openentropy) harvests entropy from 58+ hardware noise sources on the local machine — thermal sensors, CPU timing jitter, memory timing, dispatch queues, and more. No network, no API keys, no gRPC server needed.

Install:

```bash
pip install openentropy
```

Configure:

```bash
export QR_ENTROPY_SOURCE_TYPE=openentropy
export QR_OE_SOURCES=clock_jitter          # specify source(s) — omitting uses all 58 (~12-23s)
export QR_OE_CONDITIONING=sha256           # raw | vonneumann | sha256
```

List available sources on your machine:

```python
from openentropy import EntropyPool
pool = EntropyPool.auto()
print(pool.source_names())
```

Production recommendation: always set `QR_OE_SOURCES` to one or a few fast sources. `clock_jitter` with `sha256` conditioning gives ~1-2ms per call with good uniformity (mean ~127.5, std ~73.6).

See [`deployments/openentropy/`](deployments/openentropy/) for the full deployment profile.

### Fallback behavior

The `FallbackEntropySource` wraps a primary source with an automatic fallback:

- Only catches `EntropyUnavailableError` — other exceptions propagate
- Logs a warning when fallback is used
- Exposes `last_source_used` for diagnostics

Configure with `QR_FALLBACK_MODE`:
- `system` — fall back to `os.urandom()` (default)
- `mock_uniform` — fall back to the mock source
- `error` — raise immediately, no fallback

### Third-party entropy sources

Any Python package can register entropy sources via entry points:

```toml
# In your package's pyproject.toml
[project.entry-points."qr_sampler.entropy_sources"]
lava_lamp = "my_package:LavaLampEntropySource"
```

The source will be auto-discovered when qr-sampler starts. See [Setting up your own entropy source](#setting-up-your-own-entropy-source) below.

---

## gRPC transport modes

qr-sampler supports three gRPC transport modes for communicating with entropy servers. All modes satisfy the just-in-time constraint — entropy is generated only when requested.

| Mode | `QR_GRPC_MODE` | Latency | Best for |
|---|---|---|---|
| **Unary** | `unary` | ~1-2ms overhead per call | Simplicity, debugging, low-frequency sampling |
| **Server streaming** | `server_streaming` | ~0.5-1ms | Middle ground |
| **Bidirectional** | `bidi_streaming` | ~50-100us (same machine) | Production, lowest latency |

For co-located hardware, use Unix domain sockets for the lowest possible latency:

**(macOS / Linux):**

```bash
# Server
python simple_urandom_server.py --address unix:///var/run/qrng.sock

# Client config
export QR_GRPC_SERVER_ADDRESS=unix:///var/run/qrng.sock
export QR_GRPC_MODE=bidi_streaming
```

### Circuit breaker

The gRPC client includes an adaptive circuit breaker (all thresholds configurable via `QR_CB_*` environment variables):

- Tracks rolling P99 latency over the last `QR_CB_WINDOW_SIZE` requests (default: 100)
- Sets timeout to `max(QR_CB_MIN_TIMEOUT_MS, P99 * QR_CB_TIMEOUT_MULTIPLIER)` or the configured timeout, whichever is lower
- Opens the circuit after `QR_CB_MAX_CONSECUTIVE_FAILURES` consecutive failures (default: 3)
- Enters half-open state after `QR_CB_RECOVERY_WINDOW_S` seconds (default: 10), allowing one test request
- Falls back to the configured fallback source (`QR_FALLBACK_MODE`) when the circuit is open

All fallback-sourced entropy is flagged in diagnostic logs so downstream analysis can account for it.

---

## Signal amplification

The signal amplification system converts raw entropy bytes into a single uniform float `u` in `(0, 1)` that drives token selection from the CDF. The default `zscore_mean` amplifier:

1. Interprets raw bytes as uint8 values
2. Computes the sample mean M
3. Derives SEM = `population_std / sqrt(N)` (never stored — always computed)
4. Computes z-score: `z = (M - population_mean) / SEM`
5. Maps to uniform via normal CDF: `u = 0.5 * (1 + erf(z / sqrt(2)))`
6. Clamps to `(epsilon, 1-epsilon)`

Under the null hypothesis (no bias), `u` is uniformly distributed on (0, 1). A small per-byte bias accumulates over thousands of samples, producing a detectable shift:

```
20,480 bytes with +0.003 mean bias per byte:
  M ~ 127.56, SEM ~ 0.514, z ~ 0.12, u ~ 0.548
```

This makes even tiny biases statistically observable while maintaining a well-defined distribution for token selection.

---

## Temperature strategies

### Fixed temperature (`fixed`)

Returns a constant temperature for every token. Set via `QR_FIXED_TEMPERATURE`.

### Entropy-dependent temperature (`edt`)

Dynamically adjusts temperature based on the Shannon entropy of the logit distribution:

```
H_norm = H / ln(vocab_size)         # Normalized entropy [0, 1]
T = base_temp * H_norm^exponent     # Power-law scaling
T = clamp(T, min_temp, max_temp)    # Bounds enforcement
```

High-entropy (uncertain) distributions get higher temperatures; low-entropy (confident) distributions get lower temperatures. This creates a feedback loop where the model's own uncertainty calibrates the randomness of selection.

---

## Pipeline architecture

The sampling pipeline is built from independent stages that implement the `PipelineStage` protocol:

```python
@runtime_checkable
class PipelineStage(Protocol):
    name: str
    def __call__(self, ctx: SamplingContext) -> None: ...
```

Each stage reads from and writes to a shared `SamplingContext` — a mutable dataclass carrying `row` (logits), `temperature`, `u`, `config`, `entropy_source`, and per-request `stage_state`.

### Default pipeline order

| # | Stage | What it does |
|---|-------|-------------|
| 1 | `adaptive_injection` | Sets `ctx.injection_scale` from logit entropy |
| 2 | `logit_noise` | M1: adds quantum noise to logits |
| 3 | `temperature` | Computes temperature via strategy |
| 4 | `temp_variance` | M2: modulates temperature with quantum entropy |
| 5 | `min_p` | Removes low-probability tokens |
| 6 | `xtc` | Quantum coin-flip top-token exclusion |
| 7 | `entropy_fetch` | Fetches bytes + amplifies to u ∈ (0,1) |
| 8 | `correlated_walk` | M3: drifts selection point |
| 9 | `selection` | CDF-based token selection + one-hot forcing |

### Custom pipelines

Stages are registered via `@StageRegistry.register("name")` and auto-discovered via entry points. You can build custom pipelines by selecting and ordering stages:

```python
from qr_sampler.pipeline.registry import StageRegistry

# Build a minimal pipeline
pipeline = [
    StageRegistry.get("temperature")(),
    StageRegistry.get("entropy_fetch")(),
    StageRegistry.get("selection")(),
]
```

### Adding a new stage

1. Create a class in `src/qr_sampler/stages/` with `name: str` and `__call__(self, ctx: SamplingContext) -> None`
2. Register with `@StageRegistry.register("my_stage")`
3. Add config fields to `QRSamplerConfig` and `_PER_REQUEST_FIELDS`
4. Add to `build_default_pipeline()` at the appropriate position
5. Add tests in `tests/test_pipeline/`

### Experiment presets

Pre-configured experiment files live in `experiments/`:

```
experiments/
├── baseline.yaml              # No injection — control condition
├── m1_logit_noise.yaml        # M1 at multiple alpha values
├── m2_temp_variance.yaml      # M2 at multiple beta values
├── m3_correlated_walk.yaml    # M3 at multiple step sizes
├── min_p_filtering.yaml       # Min-P at multiple thresholds
├── xtc_quantum.yaml           # XTC at multiple probabilities
├── adaptive_injection.yaml    # Adaptive with different H bands
└── combined.yaml              # All methods active
```

---

## Deployment profiles

Each entropy source has a self-contained deployment profile under `deployments/`. A profile contains everything needed to run vLLM with that entropy source:

- **`docker-compose.yml`** — Self-contained compose file with all services and environment variables.
- **`.env.example`** — Annotated template. Copy to `.env` and customize.
- **`README.md`** — Setup guide specific to this entropy source.

```
deployments/
├── README.md                      # Overview and guide for creating profiles
├── .gitignore                     # Excludes .env files with secrets
├── _template/                     # Copy this to create your own profile
├── urandom/                       # os.urandom() via gRPC (start here)
├── openentropy/                   # Local hardware entropy (no network)
└── firefly-1/                     # External QRNG with API key auth
```

### Protocol flexibility

qr-sampler's gRPC client is **protocol-agnostic**. It does not require your server to implement a specific `.proto` — it uses configurable method paths and generic protobuf wire-format encoding. The only requirement is that your proto puts the byte count as field 1 in the request and the random bytes as field 1 in the response. This covers the built-in `qr_entropy.EntropyService` protocol and any server with the same field layout (e.g., `qrng.QuantumRNG`).

Configure via:
- `QR_GRPC_METHOD_PATH` — the unary RPC method (e.g., `/qrng.QuantumRNG/GetRandomBytes`)
- `QR_GRPC_STREAM_METHOD_PATH` — the streaming RPC method (empty to disable streaming)
- `QR_GRPC_API_KEY` / `QR_GRPC_API_KEY_HEADER` — authentication via gRPC metadata

The API key is never logged. Health checks report only `"authenticated": true/false`.

---

## Setting up your own entropy source

qr-sampler is designed to connect *any* randomness source to LLM token sampling. This section walks through connecting your own hardware.

### Approach A: gRPC server (recommended)

The simplest path — implement a gRPC server. You can use the built-in `qr_entropy.EntropyService` protocol (example servers provided), or your own proto as long as field 1 carries the byte count (request) and random bytes (response).

#### 5-minute walkthrough

1. **Copy the template:**

```bash
cp examples/servers/qrng_template_server.py my_qrng_server.py
```

2. **Implement three methods** in the `QRNGHardware` class:

```python
class QRNGHardware:
    def __init__(self, device_path="/dev/qrng0"):
        # Open your hardware connection
        self._device = open(device_path, "rb")

    def generate(self, n_bytes: int) -> bytes:
        # CRITICAL: Generate entropy NOW, not from a buffer.
        # The quantum measurement must happen during this call.
        return self._device.read(n_bytes)

    def close(self):
        self._device.close()
```

3. **Run it:**

```bash
pip install qr-sampler
python my_qrng_server.py --port 50051
```

4. **Create a deployment profile** and launch with Docker:

```bash
cp -r deployments/_template deployments/my-qrng
# Edit deployments/my-qrng/.env:
#   QR_ENTROPY_SOURCE_TYPE=quantum_grpc
#   QR_GRPC_SERVER_ADDRESS=<your-server>:50051
cd deployments/my-qrng
cp .env.example .env
docker compose up --build
```

Or configure directly via environment variables (bare-metal):

```bash
export QR_ENTROPY_SOURCE_TYPE=quantum_grpc
export QR_GRPC_SERVER_ADDRESS=localhost:50051
vllm serve Qwen/Qwen2.5-1.5B-Instruct --dtype half --max-model-len 8096 --gpu-memory-utilization 0.80
```

The template handles all gRPC boilerplate (unary + bidirectional streaming, health checks, graceful shutdown). You only write the hardware-specific code.

#### The gRPC protocol

The proto definition is minimal:

```protobuf
service EntropyService {
  rpc GetEntropy (EntropyRequest) returns (EntropyResponse);
  rpc StreamEntropy (stream EntropyRequest) returns (stream EntropyResponse);
}

message EntropyRequest {
  int32 bytes_needed = 1;
  int64 sequence_id = 2;
}

message EntropyResponse {
  bytes data = 1;
  int64 sequence_id = 2;
  int64 generation_timestamp_ns = 3;
  string device_id = 4;
}
```

Any language that supports gRPC can implement this server — Python, C++, Rust, Go, etc.

#### Just-in-time constraint

The entropy must be generated **after** the client sends the request, not from a pre-generated pool. This means:

- No buffering or caching of previously generated bytes
- The physical quantum measurement (or other random process) happens during the `generate()` call
- `generation_timestamp_ns` in the response proves freshness

This is critical for consciousness-research applications where the timing relationship between logit computation and entropy generation matters.

### Approach B: Python plugin (in-process)

For entropy sources that don't need a separate server, implement the `EntropySource` ABC directly:

```python
from qr_sampler.entropy.base import EntropySource
from qr_sampler.entropy.registry import register_entropy_source

@register_entropy_source("my_source")
class MyEntropySource(EntropySource):
    @property
    def name(self) -> str:
        return "my_source"

    @property
    def is_available(self) -> bool:
        return True

    def get_random_bytes(self, n: int) -> bytes:
        # Your entropy generation logic here
        return my_hardware.read(n)

    def close(self) -> None:
        my_hardware.disconnect()
```

Register via entry points in your package's `pyproject.toml`:

```toml
[project.entry-points."qr_sampler.entropy_sources"]
my_source = "my_package.entropy:MyEntropySource"
```

Then set `QR_ENTROPY_SOURCE_TYPE=my_source`.

### Validation

Test your entropy server with the built-in test infrastructure:

```python
# In a test file
from qr_sampler.entropy.quantum import QuantumGrpcSource
from qr_sampler.config import QRSamplerConfig

config = QRSamplerConfig(
    entropy_source_type="quantum_grpc",
    grpc_server_address="localhost:50051",
)
source = QuantumGrpcSource(config)

# Basic connectivity
data = source.get_random_bytes(1024)
assert len(data) == 1024

# Health check
status = source.health_check()
print(status)  # {'source': 'quantum_grpc', 'healthy': True, ...}

source.close()
```

For statistical validation, check that your source produces uniform byte distributions:

```python
import numpy as np
from scipy import stats

data = source.get_random_bytes(100_000)
samples = np.frombuffer(data, dtype=np.uint8)

# KS test against uniform distribution
stat, p_value = stats.kstest(samples / 255.0, 'uniform')
print(f"KS statistic: {stat:.6f}, p-value: {p_value:.6f}")
# p-value should be > 0.05 for a good entropy source
```

---

## Project structure

```
src/qr_sampler/
├── __init__.py                    # Package version, re-exports
├── config.py                      # Pydantic-settings configuration
├── exceptions.py                  # Exception hierarchy
├── processor.py                   # vLLM V1 LogitsProcessor (thin adapter)
├── py.typed                       # PEP 561 type hint marker
├── pipeline/
│   ├── stage.py                   # PipelineStage protocol
│   ├── context.py                 # SamplingContext mutable dataclass
│   └── registry.py                # StageRegistry + entry-point discovery
├── stages/
│   ├── _utils.py                  # Shared stable_softmax(), shannon_entropy
│   ├── adaptive_injection.py      # Entropy-aware injection scaling
│   ├── logit_noise.py             # M1: per-logit quantum noise stage
│   ├── temperature.py             # Temperature computation stage
│   ├── temp_variance.py           # M2: quantum temperature modulation stage
│   ├── min_p.py                   # Min-P probability floor stage
│   ├── xtc.py                     # XTC quantum top-token exclusion stage
│   ├── entropy_fetch.py           # JIT entropy fetch + amplification stage
│   ├── correlated_walk.py         # M3: correlated walk stage
│   └── selection.py               # CDF token selection stage
├── injection/
│   ├── _entropy_utils.py          # bytes_to_uniform() helper
│   ├── logit_noise.py             # M1: Gaussian noise utility
│   ├── temp_variance.py           # M2: temperature modulation utility
│   └── correlated_walk.py         # M3: walk position utility
├── amplification/
│   ├── base.py                    # SignalAmplifier ABC, AmplificationResult
│   ├── registry.py                # AmplifierRegistry
│   └── zscore.py                  # Z-score mean amplifier
├── entropy/
│   ├── base.py                    # EntropySource ABC
│   ├── registry.py                # Auto-discovery registry + entry points
│   ├── quantum.py                 # gRPC QRNG source (3 transport modes)
│   ├── openentropy.py             # OpenEntropy hardware noise source
│   ├── system.py                  # os.urandom() source
│   ├── timing.py                  # CPU timing jitter source
│   ├── mock.py                    # Configurable test source
│   └── fallback.py                # Fallback wrapper
├── logging/
│   ├── types.py                   # TokenSamplingRecord dataclass
│   └── logger.py                  # SamplingLogger (none/summary/full)
├── proto/
│   ├── entropy_service.proto      # gRPC protocol definition
│   ├── entropy_service_pb2.py     # Protobuf stubs
│   └── entropy_service_pb2_grpc.py # gRPC stubs
├── selection/
│   ├── types.py                   # SelectionResult dataclass
│   └── selector.py                # CDF-based token selector
└── temperature/
    ├── base.py                    # TemperatureStrategy ABC, Shannon entropy
    ├── registry.py                # TemperatureStrategyRegistry
    ├── fixed.py                   # Fixed temperature strategy
    └── edt.py                     # Entropy-dependent temperature

experiments/                       # Pre-configured experiment presets
scripts/
├── injection_mode_test.py         # M1/M2/M3 comparison script
└── openentropy_mode_test.py       # Full-mode test with OpenEntropy

examples/
├── servers/
│   ├── simple_urandom_server.py   # Minimal reference server (~50 lines)
│   ├── timing_noise_server.py     # CPU timing entropy server
│   └── qrng_template_server.py    # Annotated template for custom QRNGs
├── open-webui/
│   ├── qr_sampler_filter.py       # Open WebUI filter function (source)
│   ├── qr_sampler_filter.json     # Open WebUI importable JSON
│   └── README.md                  # Filter function docs
├── docker/
│   ├── Dockerfile.vllm            # vLLM + qr-sampler image
│   └── Dockerfile.entropy-server  # Docker image for entropy servers
└── systemd/
    ├── qr-entropy-server.service  # systemd unit file
    └── qr-entropy-server.env      # Environment file

deployments/
├── _template/                     # Copy to create a new profile
├── urandom/                       # os.urandom() via gRPC (start here)
├── openentropy/                   # Local hardware entropy
└── firefly-1/                     # External QRNG with API key auth
```

---

## Statistical analysis

qr-sampler includes statistical tests (in `tests/test_statistical_properties.py`, requires `scipy`) that validate the mathematical properties of the sampling pipeline:

- **KS-test for u-value uniformity**: Under the null hypothesis (no bias), amplified `u` values should be uniformly distributed on (0, 1). The test runs a Kolmogorov-Smirnov test against a uniform reference distribution.
- **Bias detection**: Verifies that introducing a small per-byte mean shift (e.g., `mean=128.0` instead of `127.5`) produces a statistically detectable shift in the `u` distribution — confirming the amplification system is sensitive enough for consciousness-research experiments.
- **EDT monotonicity**: Validates that the entropy-dependent temperature strategy produces higher temperatures for higher-entropy logit distributions, as designed.

These tests run as part of the standard test suite:

```bash
pytest tests/test_statistical_properties.py -v
```

---

## Development

```bash
# Clone and install
git clone https://github.com/ereid7/Quantum-random-vLLM-sampler.git
cd Quantum-random-vLLM-sampler
pip install -e ".[dev]"

# Run tests (537 tests)
pytest tests/ -v

# Lint and format
ruff check src/ tests/
ruff format src/ tests/

# Type check
mypy --strict src/

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full development guide.

---

## Acknowledgements

This project is forked from [alchemystack/Quantum-random-vLLM-sampler](https://github.com/alchemystack/Quantum-random-vLLM-sampler). The original project by [alchemystack](https://github.com/alchemystack) established the core architecture: entropy source abstraction, signal amplification via z-score statistics, CDF-based token selection, and the gRPC transport layer.

This fork adds the pipeline-as-stages architecture, injection methods (M1/M2/M3, Min-P, XTC, Adaptive Injection), OpenEntropy integration, experiment presets, and expanded test coverage.

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
