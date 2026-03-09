# entropick

**Plug any physical randomness source into LLM token sampling.**

entropick replaces the pseudorandom number generator in LLM token sampling with true randomness from external sources — quantum random number generators, hardware entropy via [OpenEntropy](https://github.com/amenti-labs/openentropy), CPU timing jitter, or any device you connect via gRPC. A composable pipeline of 13 stages filters candidates, modulates temperature, perturbs logits, and selects tokens — all driven by physical randomness instead of software PRNGs.

Works with **vLLM** (via entry point), **Hugging Face Transformers**, **llama.cpp**, and **SGLang**.

```
pip install entropick
```

## Start Here

Pick the path that matches what you are trying to do:

| If you want to... | Use this path | Why |
|-------------------|---------------|-----|
| Prove the repo works end to end with minimal setup | [`deployments/urandom/`](deployments/urandom/) | Fastest working demo of vLLM + entropick + gRPC entropy |
| Use machine-local hardware noise directly | [`deployments/openentropy/`](deployments/openentropy/) | Native OpenEntropy setup with no network hop |
| Connect your own entropy server | [`deployments/_template/`](deployments/_template/) | Clean starting point for custom gRPC backends |
| Run locally without deployment profiles | [repo-level `.env` examples](#quick-start) | Smallest manual setup for direct `vllm serve` use |

If you are new to the repo, start with `deployments/urandom/`.

No matter which path you choose, the end result is the same:

1. Start a model server with entropick enabled.
2. Send normal OpenAI-compatible completion/chat requests to `http://localhost:8000`.
3. Tune only a small set of sampling knobs unless you are running a specific experiment.

> **Research context:** entropick was built for consciousness-research experiments studying whether statistical anomalies appear in quantum-random-driven token selection. The signal amplification system converts thousands of random bytes into a single token choice, making even tiny statistical biases (e.g., 0.1% shift in byte means) observable in downstream token distributions. All entropy is generated **just-in-time** — the physical measurement happens *after* logits are computed, never before. This is a measurement and sampling tool. It does not steer, constrain, or control model outputs.

> Forked from [alchemystack/Quantum-random-vLLM-sampler](https://github.com/alchemystack/Quantum-random-vLLM-sampler). See [Acknowledgements](#acknowledgements).

---

## Quick start

### Fastest working path: Docker + `urandom`

If you just want to see the whole stack working, use the `urandom` deployment profile first.

| Profile | Entropy source | Description |
|---------|---------------|-------------|
| [`urandom/`](deployments/urandom/) | `os.urandom()` via gRPC | Local gRPC server for testing. **Start here.** |
| [`openentropy/`](deployments/openentropy/) | Local hardware noise | Native-only profile for machine-local entropy sources. |
| [`_template/`](deployments/_template/) | Your hardware | Copy and customize. |

```bash
cd deployments/urandom
cp .env.example .env          # edit .env — set HF_TOKEN if using a gated model
docker compose up --build
```

### Native path: direct `vllm serve`

Use this when you want to run without Docker or when you want local OpenEntropy on the host machine.

For native installs, start from one of the repo-level `.env` examples:

| File | Use when |
|------|----------|
| [`.env.example`](.env.example) | You want the simplest local setup with system entropy. |
| [`.env.grpc.example`](.env.grpc.example) | Your entropy source is a gRPC server. |
| [`.env.openentropy.example`](.env.openentropy.example) | You want local OpenEntropy hardware noise. |

```bash
cp .env.example .env         # or .env.grpc.example / .env.openentropy.example
set -a; source .env; set +a  # export QR_* vars from the file
pip install entropick
# If you chose .env.openentropy.example, also run: pip install openentropy

vllm serve "$HF_MODEL" \
  --logits-processors qr_sampler \
  --dtype half --max-model-len 8096 --gpu-memory-utilization 0.80
```

For a native OpenEntropy path with profile-specific docs and launcher script, use [`deployments/openentropy/`](deployments/openentropy/).

### Once the server is running

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "prompt": "The nature of consciousness is",
    "max_tokens": 100
  }'
```

To use a gRPC entropy server, switch to `.env.grpc.example` or set:

```bash
export QR_ENTROPY_SOURCE_TYPE=quantum_grpc
export QR_GRPC_SERVER_ADDRESS=localhost:50051
```

For OpenEntropy, switch to `.env.openentropy.example` or set:

```bash
export QR_ENTROPY_SOURCE_TYPE=openentropy
export QR_OE_SOURCES=clock_jitter          # recommended fast single source
export QR_OE_CONDITIONING=sha256           # recommended default conditioning
export QR_FALLBACK_MODE=system
```

---

## Configuration In 30 Seconds

entropick has two configuration layers:

1. **Infrastructure**: where entropy comes from and how to reach it. Set these once in `.env`. They are process-wide.
2. **Sampling**: how token selection behaves. Set defaults in `.env`, then override per request with `extra_args` only when you need an experiment.

Most users only need these knobs:

| Variable | Why you would touch it |
|----------|------------------------|
| `QR_ENTROPY_SOURCE_TYPE` | Pick `system`, `quantum_grpc`, `openentropy`, `mock_uniform`, or `sham_qrng`. |
| `QR_GRPC_SERVER_ADDRESS` | Point at your gRPC entropy server when using `quantum_grpc`. |
| `QR_OE_SOURCES` | Pick a specific OpenEntropy source such as `clock_jitter`. |
| `QR_FALLBACK_MODE` | Decide whether failures fall back to `system`, `mock_uniform`, or `error`. |
| `QR_TEMPERATURE_STRATEGY` / `QR_FIXED_TEMPERATURE` | Set the baseline temperature behavior. |
| `QR_TOP_K` / `QR_TOP_P` | Set the candidate set before final token selection. |
| `QR_SAMPLE_COUNT` | Increase only if you are deliberately amplifying tiny entropy biases. |

Everything else in the full reference is optional and can stay at defaults until you need a specific experiment.

## How it works

entropick runs a **pipeline of 13 stages** on each token's logits. With default config, only 3 stages do actual work — the rest are disabled no-ops with zero overhead.

### Default path (always runs)

```
Logits from the model
  │
  ├─ Temperature ──────── Apply temperature scaling (fixed or entropy-dependent)
  │
  ├─ Entropy Fetch ────── Fetch 20,480 random bytes → z-score → uniform u ∈ (0,1)
  │
  └─ Selection ────────── top-k → softmax → top-p → CDF → pick token
       └─ Force one-hot: row = -inf everywhere, 0.0 at selected token
```

### Full pipeline

Enable any stage by setting its control parameter to a non-zero value:

```
  ┌─ PRE-TEMPERATURE ──────────────────────────────────────────────────────┐
  │  1. Adaptive Injection ── Scale injection by model entropy      (off) │
  │  2. Logit Perturbation ── Per-logit quantum Gaussian noise      (off) │
  │  3. DRY ─────────────── N-gram repetition penalty               (off) │
  │  4. Top-N-Sigma ─────── Keep logits within N sigma of max       (off) │
  └────────────────────────────────────────────────────────────────────────┘

  ── 5. Temperature ──────── ALWAYS RUNS ──────────────────────────────────

  ┌─ POST-TEMPERATURE ─────────────────────────────────────────────────────┐
  │  6. Temp Modulation ──── Quantum temperature modulation         (off) │
  │  7. Min-P ────────────── Dynamic probability floor              (off) │
  │  8. XTC ──────────────── Quantum coin-flip top-token exclusion  (off) │
  └────────────────────────────────────────────────────────────────────────┘

  ── 9. Entropy Fetch ────── ALWAYS RUNS ──────────────────────────────────

  ┌─ POST-ENTROPY ─────────────────────────────────────────────────────────┐
  │ 10. Selection Drift ──── Drift u with temporal memory           (off) │
  └────────────────────────────────────────────────────────────────────────┘

  ┌─ SELECTION (mutually exclusive — only one runs) ───────────────────────┐
  │ 11. Mirostat ────────── Adaptive perplexity control             (off) │
  │ 12. Gumbel Selection ── Gumbel-Max with quantum noise           (off) │
  │ 13. Selection ────────── CDF-based token selection          (default) │
  └────────────────────────────────────────────────────────────────────────┘
```

Stages 11-13 are **mutually exclusive**. If Mirostat or Gumbel is enabled, the default CDF Selection is skipped.

The processor registers via Python entry points — no framework source code modifications needed.

---

## Sampling stages

All stages below are **disabled by default**. Enable any by setting its control parameter. They compose freely.

### Logit Perturbation

Adds per-logit Gaussian noise derived from quantum entropy. Fetches `vocab_size x 4` bytes, maps to zero-mean noise via the probit transform, scales by `alpha x sigma`.

| Parameter | Default | Effect |
|-----------|---------|--------|
| `QR_LOGIT_PERTURBATION_ALPHA` | `0.0` | Noise magnitude (0 = disabled) |
| `QR_LOGIT_PERTURBATION_SIGMA` | `1.0` | Gaussian std before alpha scaling |

### Temperature Modulation

Modulates temperature per-token using quantum entropy: `T_new = T x (1 + beta x (u - 0.5))`.

| Parameter | Default | Effect |
|-----------|---------|--------|
| `QR_TEMP_MODULATION_BETA` | `0.0` | Modulation magnitude (0 = disabled) |

### Selection Drift

Maintains a per-request drift position that replaces the amplified `u` value. Creates temporal correlations across tokens within a request.

| Parameter | Default | Effect |
|-----------|---------|--------|
| `QR_DRIFT_STEP` | `0.0` | Step size (0 = disabled) |
| `QR_DRIFT_INITIAL_POSITION` | `0.5` | Starting position in [0, 1) |

### Min-P

Dynamic probability floor — removes tokens where `p < min_p x max(p)`. Adapts to model confidence automatically.

| Parameter | Default | Effect |
|-----------|---------|--------|
| `QR_MIN_P` | `0.0` | Threshold (0 = disabled, 1.0 = top token only) |

### XTC (Exclude Top Choices)

Probabilistically excludes top tokens using quantum coin flips. Each token above the threshold gets an independent quantum decision.

| Parameter | Default | Effect |
|-----------|---------|--------|
| `QR_XTC_PROBABILITY` | `0.0` | Exclusion probability per token (0 = disabled) |
| `QR_XTC_THRESHOLD` | `0.1` | Min probability to be an exclusion candidate |

### Adaptive Injection

Scales all injection methods by the Shannon entropy H of the logit distribution. Low model confidence = full injection; high confidence = suppressed.

| Parameter | Default | Effect |
|-----------|---------|--------|
| `QR_ADAPTIVE_INJECTION` | `false` | Enable/disable |
| `QR_ADAPTIVE_INJECTION_LOW_H` | `1.0` | H below this = scale 0 (nats) |
| `QR_ADAPTIVE_INJECTION_HIGH_H` | `3.0` | H above this = scale 1 (nats) |

### DRY (Don't Repeat Yourself)

Penalizes repeated n-gram sequences with an exponential penalty.

| Parameter | Default | Effect |
|-----------|---------|--------|
| `QR_DRY_MULTIPLIER` | `0.0` | Penalty multiplier (0 = disabled) |
| `QR_DRY_BASE` | `1.75` | Exponential penalty base |
| `QR_DRY_ALLOWED_LENGTH` | `2` | Min sequence length to penalize |
| `QR_DRY_PENALTY_LAST_N` | `-1` | Lookback window (-1 = full context) |

### Top-N-Sigma

Pre-softmax filter — keeps only tokens whose logits are within N standard deviations of the maximum.

| Parameter | Default | Effect |
|-----------|---------|--------|
| `QR_TOP_N_SIGMA` | `0.0` | Standard deviations (0 = disabled) |

### Mirostat v2

Adaptive perplexity control — adjusts the candidate set to maintain a target surprise rate tau.

| Parameter | Default | Effect |
|-----------|---------|--------|
| `QR_MIROSTAT_MODE` | `0` | 0 = disabled, 2 = mirostat v2 |
| `QR_MIROSTAT_TAU` | `5.0` | Target surprise rate (nats) |
| `QR_MIROSTAT_ETA` | `0.1` | Learning rate |

### Gumbel-Max Selection

Adds Gumbel noise (from quantum entropy) to log-probabilities and selects via argmax. Mathematically equivalent to categorical sampling but structurally different.

| Parameter | Default | Effect |
|-----------|---------|--------|
| `QR_GUMBEL_SELECTION` | `false` | Enable Gumbel-Max selection |

---

## Entropy sources

### Built-in

| Source | `QR_ENTROPY_SOURCE_TYPE` | Description |
|--------|--------------------------|-------------|
| **System** | `system` | `os.urandom()` — OS cryptographic RNG (default) |
| **Quantum gRPC** | `quantum_grpc` | Remote entropy server via gRPC |
| **OpenEntropy** | `openentropy` | Platform-dependent hardware noise sources — local, no network |
| **Mock uniform** | `mock_uniform` | Configurable test source with seed/bias |
| **Sham QRNG** | `sham_qrng` | `os.urandom()` + simulated latency for double-blind controls |

### OpenEntropy

[OpenEntropy](https://github.com/amenti-labs/openentropy) harvests entropy from platform-dependent hardware noise sources on the local machine — thermal sensors, CPU timing jitter, memory timing, dispatch queues, and more. No network, no API keys.

```bash
pip install openentropy

export QR_ENTROPY_SOURCE_TYPE=openentropy
export QR_OE_SOURCES=clock_jitter          # omit to auto-discover all available sources
export QR_OE_CONDITIONING=sha256           # raw | vonneumann | sha256
```

Production: use `clock_jitter` with `sha256` conditioning (~1-2ms/call).

### Fallback

`FallbackEntropySource` wraps a primary source with automatic failover. Only catches `EntropyUnavailableError` — other exceptions propagate.

| `QR_FALLBACK_MODE` | Behavior |
|---------------------|----------|
| `system` | Fall back to `os.urandom()` (default) |
| `mock_uniform` | Fall back to mock source |
| `error` | Raise immediately, no fallback |

### Third-party sources

Register via Python entry points:

```toml
# In your package's pyproject.toml
[project.entry-points."qr_sampler.entropy_sources"]
lava_lamp = "my_package:LavaLampEntropySource"
```

The source is auto-discovered when entropick starts.

---

## Multi-framework support

entropick started as a vLLM plugin but includes adapters for other inference frameworks:

| Framework | Adapter | Integration |
|-----------|---------|-------------|
| **vLLM** | `QRSamplerLogitsProcessor` | Auto-registered via entry point — zero config |
| **Transformers** | `QRSamplerLogitsProcessorHF` | Pass to `model.generate(logits_processor=[...])` |
| **llama.cpp** | `QRSamplerCallback` | Custom sampler callback |
| **SGLang** | `QRSamplerCustomLogitProcessor` | Custom logit processor |

```python
from qr_sampler.adapters import QRSamplerLogitsProcessorHF

processor = QRSamplerLogitsProcessorHF()
outputs = model.generate(input_ids, logits_processor=[processor])
```

All adapters share the same pipeline, config system, entropy sources, and logging.

---

## Experiment design

### Presets

Pre-configured experiment files in `experiments/`:

```
experiments/
├── baseline.yaml              # No injection — control condition
├── logit_perturbation.yaml    # Logit perturbation at multiple alpha values
├── temp_modulation.yaml       # Temperature modulation at multiple beta values
├── selection_drift.yaml       # Selection drift at multiple step sizes
├── min_p_filtering.yaml       # Min-P at multiple thresholds
├── xtc_quantum.yaml           # XTC at multiple probabilities
├── adaptive_injection.yaml    # Adaptive with different H bands
└── combined.yaml              # All methods active
```

These YAML files are human-readable environment bundles. They are not auto-loaded by entropick; copy the `env:` block into your shell, `.env` file, or request defaults. See [`experiments/README.md`](experiments/README.md).

### Double-blind controls

The `sham_qrng` entropy source provides `os.urandom()` bytes with configurable simulated latency (`QR_SHAM_QRNG_LATENCY_MS`). It is indistinguishable from a real QRNG in timing and output characteristics, enabling double-blind experimental designs where neither the operator nor the analysis pipeline knows which source was used.

### Analysis tools

entropick includes a statistical analysis module for post-experiment data analysis:

**Persistence** — save and load per-token diagnostic records:

```python
from qr_sampler.analysis import save_records, load_records

save_records(records, "session_001.jsonl")
data = load_records("session_001.jsonl")
```

**Statistical tests** (9 tests, requires `scipy`):
autocorrelation, runs test, serial correlation, Hurst exponent, approximate entropy, chi-square rank test, cumulative deviation, entropy rate, Bayesian sequential analysis.

**Session comparison** — compare experimental vs. control sessions:

```python
from qr_sampler.analysis import compare_sessions, effect_size_report

result = compare_sessions(baseline, experimental, field="u_value")
# Returns Mann-Whitney U, KS test, Welch's t-test results

effect = effect_size_report(baseline, experimental, field="u_value")
# Returns Cohen's d
```

---

## Configuration reference

All configuration uses the `QR_` environment variable prefix. Per-request overrides use `qr_` in `extra_args`.

> **Note:** The `QR_` prefix and `qr_sampler` Python import paths are inherited from the upstream project and will be migrated in a future release.

If you are just getting started, use the repo-level `.env` examples above and skip to this section only when you need a specific advanced knob.

### Per-request overrides

Override sampling parameters on individual requests:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "prompt": "The nature of consciousness is",
    "max_tokens": 100,
    "extra_args": {
      "qr_min_p": 0.1,
      "qr_logit_perturbation_alpha": 0.3,
      "qr_adaptive_injection": true
    }
  }'
```

Or set process-wide via environment variables:

```bash
export QR_LOGIT_PERTURBATION_ALPHA=0.3
export QR_TEMP_MODULATION_BETA=0.5
export QR_MIN_P=0.1
```

### How most users actually use config

Use the config in this order:

1. Pick an entropy backend in `.env` by setting `QR_ENTROPY_SOURCE_TYPE`.
2. Set only the backend-specific connection fields you need (`QR_GRPC_SERVER_ADDRESS` or `QR_OE_SOURCES`).
3. Leave the rest of the infrastructure defaults alone unless you are debugging reliability, security, or latency.
4. Tune sampling defaults only if you want a different generation style across all requests.
5. Use `extra_args` for one-off experiments, not for ordinary serving.

The reason the split exists is simple: infrastructure fields change how the server is wired, while sampling fields change how a token is chosen after logits are already available.

### Common setup recipes

If you ignore everything else, start from one of these three setups.

#### Simplest local setup

Use this when you just want entropick running with no extra infrastructure.

```env
QR_ENTROPY_SOURCE_TYPE=system
QR_FALLBACK_MODE=error
```

#### gRPC entropy server

Use this when entropy comes from a separate process, local hardware daemon, or another machine.

```env
QR_ENTROPY_SOURCE_TYPE=quantum_grpc
QR_GRPC_SERVER_ADDRESS=localhost:50051
QR_FALLBACK_MODE=system
```

Leave TLS, retries, method paths, and circuit breaker settings at their defaults unless you know you need them.

#### OpenEntropy

Use this when you want machine-local hardware noise with no network hop.

```env
QR_ENTROPY_SOURCE_TYPE=openentropy
QR_OE_SOURCES=clock_jitter
QR_OE_CONDITIONING=sha256
QR_FALLBACK_MODE=system
```

### Common sampling knobs

These are the sampling settings most people actually touch.

| Variable | `extra_args` key | What it changes |
|----------|-----------------|-----------------|
| `QR_TEMPERATURE_STRATEGY` | `qr_temperature_strategy` | Whether temperature is fixed or entropy-dependent (`edt`). |
| `QR_FIXED_TEMPERATURE` | `qr_fixed_temperature` | The baseline creativity / sharpness level for ordinary fixed-temperature sampling. |
| `QR_TOP_K` | `qr_top_k` | Hard cap on how many candidate tokens remain. |
| `QR_TOP_P` | `qr_top_p` | Nucleus cutoff for the candidate set. |
| `QR_MIN_P` | `qr_min_p` | Relative probability floor for dropping weak candidates. |
| `QR_SAMPLE_COUNT` | `qr_sample_count` | How much entropy is consumed per token. Higher means stronger amplification and more latency. |
| `QR_SIGNAL_AMPLIFIER_TYPE` | `qr_signal_amplifier_type` | How entropy bytes are mapped into the final uniform draw. |
| `QR_MIROSTAT_MODE` | `qr_mirostat_mode` | Switches to Mirostat v2 if you want surprise-rate control instead of ordinary CDF sampling. |

### Full reference

The exhaustive grouped variable list now lives in [docs/config-reference.md](docs/config-reference.md).

Use that file when you need:

1. Every supported env var and default.
2. The full `QR_` to `extra_args` mapping.
3. Advanced gRPC, TLS, calibration, repetition, and injection controls.

---

## gRPC transport

| Mode | `QR_GRPC_MODE` | Latency | Best for |
|------|----------------|---------|----------|
| Unary | `unary` | ~1-2ms overhead | Simplicity, debugging |
| Server streaming | `server_streaming` | ~0.5-1ms | Middle ground |
| Bidirectional | `bidi_streaming` | ~50-100us (same machine) | Production, lowest latency |

For co-located hardware, use Unix domain sockets:

```bash
export QR_GRPC_SERVER_ADDRESS=unix:///var/run/qrng.sock
export QR_GRPC_MODE=bidi_streaming
```

### Circuit breaker

The gRPC client includes an adaptive circuit breaker:

- Tracks rolling P99 latency (configurable window, default 100 requests)
- Adaptive timeout: `max(5ms, P99 x 1.5)` or configured timeout, whichever is lower
- Opens after 3 consecutive failures, half-open retry after 10s
- Falls back to `QR_FALLBACK_MODE` when the circuit is open
- All thresholds configurable via `QR_CB_*` environment variables

---

## Signal amplification

Converts raw entropy bytes into a uniform float `u` in (0, 1) for token selection:

1. Raw bytes interpreted as uint8 values
2. Compute sample mean M
3. Z-score: `z = (M - population_mean) / (population_std / sqrt(N))`
4. Normal CDF: `u = 0.5 x (1 + erf(z / sqrt(2)))`
5. Clamp to `(epsilon, 1 - epsilon)`

Under the null hypothesis (no bias), `u` is uniformly distributed on (0, 1). A small per-byte bias accumulates over 20,480 samples, producing a detectable shift in the CDF position — this is what makes consciousness-research experiments possible.

An alternative **ECDF amplifier** (`QR_SIGNAL_AMPLIFIER_TYPE=ecdf`) uses empirical calibration instead of distributional assumptions.

---

## Web UI

entropick works with [Open WebUI](https://github.com/open-webui/open-webui). Add `--profile ui` to any deployment:

```bash
cd deployments/urandom
docker compose --profile ui up --build
# Open http://localhost:3000
```

A pre-built [filter function](examples/open-webui/) injects sampling parameters into every chat message via the Open WebUI Valves system. See [`examples/open-webui/README.md`](examples/open-webui/README.md).

---

## Connecting your own entropy source

### Approach A: gRPC server (recommended)

Copy the template and implement your hardware's `generate()` method:

```python
class QRNGHardware:
    def generate(self, n_bytes: int) -> bytes:
        # CRITICAL: generate entropy NOW, not from a buffer
        return self._device.read(n_bytes)
```

```bash
cp examples/servers/qrng_template_server.py my_qrng_server.py
# Implement your hardware, then:
python3 my_qrng_server.py --port 50051
```

The template handles all gRPC boilerplate (unary + bidi streaming, health checks, graceful shutdown). The gRPC protocol is minimal — field 1 carries byte count (request) and random bytes (response). Any language with gRPC support works.

See [deployments/README.md](deployments/README.md) for Docker deployment.

### Approach B: Python plugin (in-process)

```python
from qr_sampler.entropy.base import EntropySource
from qr_sampler.entropy.registry import register_entropy_source

@register_entropy_source("my_source")
class MySource(EntropySource):
    @property
    def name(self) -> str: return "my_source"

    @property
    def is_available(self) -> bool: return True

    def get_random_bytes(self, n: int) -> bytes:
        return my_hardware.read(n)

    def close(self) -> None:
        my_hardware.disconnect()
```

Register via entry point:

```toml
[project.entry-points."qr_sampler.entropy_sources"]
my_source = "my_package:MySource"
```

Then set `QR_ENTROPY_SOURCE_TYPE=my_source`.

---

## Custom pipelines

Stages are registered via `@StageRegistry.register("name")` and auto-discovered via entry points. Build custom pipelines by selecting and ordering stages:

```python
from qr_sampler.pipeline.registry import StageRegistry

pipeline = [
    StageRegistry.get("temperature")(),
    StageRegistry.get("entropy_fetch")(),
    StageRegistry.get("selection")(),
]
```

To add a new stage, create a class with `name: str` and `__call__(self, ctx: SamplingContext) -> None`, register it, and add to `build_default_pipeline()`. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide.

---

## Development

```bash
git clone https://github.com/ereid7/entropick.git
cd entropick
pip install -e ".[dev]"

pytest tests/ -v                              # full suite
ruff check src/ tests/                        # lint
ruff format --check src/ tests/               # format check
mypy --strict src/                            # type check
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for coding conventions and component guides.

---

## Acknowledgements

Forked from [alchemystack/Quantum-random-vLLM-sampler](https://github.com/alchemystack/Quantum-random-vLLM-sampler). The original project by [alchemystack](https://github.com/alchemystack) established the core architecture: entropy source abstraction, z-score signal amplification, CDF-based token selection, and the gRPC transport layer.

This fork adds the pipeline-as-stages architecture, multi-framework adapters (Transformers, llama.cpp, SGLang), injection methods (Logit Perturbation, Temperature Modulation, Selection Drift, Min-P, XTC, Adaptive Injection, DRY, Top-N-Sigma, Mirostat, Gumbel-Max), OpenEntropy integration, statistical analysis tools, experiment presets, deployment profiles, and expanded test coverage.

## License

Apache 2.0 — see [LICENSE](LICENSE).
