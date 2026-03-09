# entropick

**Plug physical randomness into LLM token sampling.**

entropick replaces the software PRNG used during token selection with entropy fetched just in time from system randomness, gRPC entropy services, or OpenEntropy. It is primarily designed for vLLM-based research and engineering workflows that want external entropy in the token-selection path.

Works with **vLLM** first, with adapters for **Hugging Face Transformers** and **llama.cpp**.

## Primary use case

The main intended setup for this repo is:

- **vLLM**
- **OpenEntropy running on the host machine**
- **a Crypta Labs QCICADA QRNG device exposed through OpenEntropy**

If that is your target, start with [`deployments/openentropy/`](deployments/openentropy/). The `urandom` profile is still the fastest proof that the stack works, but it is a demo path, not the primary hardware-backed use case.

## Who this is for

- Researchers running controlled entropy experiments against LLM token selection.
- Engineers who want vLLM to draw sampling randomness from an external source.
- Teams integrating custom hardware or remote entropy services over gRPC.

If you only need ordinary temperature or top-p sampling with a software PRNG, this repo is probably more than you need.

## Start Here

Pick the path that matches what you are trying to do:

| If you want to... | Use this path | Why |
|-------------------|---------------|-----|
| Prove the repo works end to end with minimal setup | [`deployments/urandom/`](deployments/urandom/) | Fastest working demo of vLLM + entropick + gRPC entropy |
| Use the main target setup: QCICADA via OpenEntropy | [`deployments/openentropy/`](deployments/openentropy/) | Native host setup for machine-local hardware noise, including QCICADA-style OpenEntropy sources |
| Connect your own entropy server | [`deployments/_template/`](deployments/_template/) | Clean starting point for custom gRPC backends |
| Run locally without deployment profiles | [`docs/getting-started.md`](docs/getting-started.md) | Step-by-step native setup with repo-level `.env` examples |

If you are new to the repo and your goal is the intended hardware-backed path, start with `deployments/openentropy/`. If you just want to verify the stack quickly, start with `deployments/urandom/`.

No matter which path you choose, the end result is the same:

1. Start a model server with entropick enabled.
2. Send normal OpenAI-compatible completion or chat requests to `http://localhost:8000`.
3. Tune only a small set of sampling knobs unless you are running a specific experiment.

## Quick start

### Main path: QCICADA/OpenEntropy on the host

Use this when the goal is to run entropick the way the repo is primarily intended to be used: vLLM on the host, OpenEntropy on the host, and a QCICADA device feeding entropy through OpenEntropy.

```bash
cd deployments/openentropy
cp .env.example .env
./run-local.sh
```

See [`deployments/openentropy/README.md`](deployments/openentropy/README.md) for OpenEntropy-specific setup notes and source selection.

### Fastest working path: Docker + `urandom`

If you just want to see the whole stack working before you move to real hardware, use the `urandom` deployment profile first.

```bash
cd deployments/urandom
cp .env.example .env          # edit .env — set HF_TOKEN if using a gated model
docker compose up --build
```

### Native path: direct `vllm serve`

Use this when you want to run without Docker or when you want to use OpenEntropy on the host machine.

| File | Use when |
|------|----------|
| [`.env.example`](.env.example) | You want the simplest local setup with system entropy. |
| [`.env.grpc.example`](.env.grpc.example) | Your entropy source is a gRPC server. |
| [`.env.openentropy.example`](.env.openentropy.example) | You want local OpenEntropy hardware noise. |

```bash
cp .env.example .env         # or .env.grpc.example / .env.openentropy.example
set -a; source .env; set +a  # export QR_* vars from the file
pip install entropick vllm
# If you chose .env.openentropy.example, also run: pip install openentropy

vllm serve "$HF_MODEL" \
  --logits-processors qr_sampler \
  --dtype half --max-model-len 8096 --gpu-memory-utilization 0.80
```

For native gRPC and OpenEntropy variations, see [`docs/getting-started.md`](docs/getting-started.md) and [`deployments/openentropy/`](deployments/openentropy/).

### First request

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "prompt": "The nature of consciousness is",
    "max_tokens": 100
  }'
```

## What entropick changes

1. The model still computes logits normally.
2. entropick fetches entropy after logits are available.
3. That entropy is amplified into a uniform draw `u` and used in token selection.
4. The chosen token is forced back into the runtime so the framework emits that token.

See [`docs/how-it-works.md`](docs/how-it-works.md) for the default path, the full stage pipeline, gRPC transport modes, and signal amplification details.

## Runtime support

Support status in this repo means:

- `First-class`: onboarding, deployment docs, and per-request behavior are optimized for this runtime.
- `Supported`: adapter exists and is tested, but the repo is not primarily organized around that runtime.

| Runtime | Status | How you use it | Notes |
|---------|--------|----------------|-------|
| **vLLM** | First-class | Entry point plus `--logits-processors qr_sampler` | Recommended path and deployment profiles |
| **Transformers** | Supported | `QRSamplerLogitsProcessorHF` | Use inside `model.generate(...)` |
| **llama.cpp** | Supported | `QRSamplerCallback` | Use with `llama-cpp-python` logits processors |

See [`docs/framework-support.md`](docs/framework-support.md) for examples and runtime-specific caveats.

## Configuration in 30 seconds

entropick has two configuration layers:

1. **Infrastructure**: where entropy comes from and how to reach it. Set these once in `.env`. They are process-wide.
2. **Sampling**: how token selection behaves. Set defaults in `.env`, then override per request only when you need an experiment.

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

Everything else can usually stay at defaults until you need a specific experiment. The exhaustive variable matrix lives in [`docs/config-reference.md`](docs/config-reference.md).

## Docs map

| Read this | Use when |
|-----------|----------|
| [`docs/getting-started.md`](docs/getting-started.md) | You want a step-by-step first successful run. |
| [`deployments/README.md`](deployments/README.md) | You want to choose the right deployment profile. |
| [`docs/framework-support.md`](docs/framework-support.md) | You are integrating with vLLM, Transformers, or llama.cpp. |
| [`docs/config-reference.md`](docs/config-reference.md) | You need every env var, default, and per-request override name. |
| [`docs/how-it-works.md`](docs/how-it-works.md) | You want the pipeline, transport, and amplification internals. |
| [`docs/entropy-sources.md`](docs/entropy-sources.md) | You want to understand built-in entropy backends and fallback behavior. |
| [`docs/experiments.md`](docs/experiments.md) | You are running controlled experiments, sham controls, or post-run analysis. |
| [`examples/open-webui/README.md`](examples/open-webui/README.md) | You want Open WebUI integration and valve-controlled sampling. |
| [`docs/extending.md`](docs/extending.md) | You want to add a custom entropy source or custom pipeline stages. |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | You want development setup, tests, and contributor guidance. |

## Research context

entropick was built for consciousness-research experiments looking for statistical anomalies in quantum-random-driven token selection. It is a measurement and sampling tool, not a steering or control layer.

For experiment presets, sham controls, and analysis helpers, see [`docs/experiments.md`](docs/experiments.md) and [`experiments/README.md`](experiments/README.md).

## Development

For local setup, tests, linting, type checking, and contributor guidelines, see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Acknowledgements

Forked from [alchemystack/Quantum-random-vLLM-sampler](https://github.com/alchemystack/Quantum-random-vLLM-sampler). The original project by [alchemystack](https://github.com/alchemystack) established the core architecture: entropy source abstraction, z-score signal amplification, CDF-based token selection, and the gRPC transport layer.

This fork adds the pipeline-as-stages architecture, multi-framework adapters, injection methods, OpenEntropy integration, statistical analysis tools, experiment presets, deployment profiles, and expanded test coverage.

## License

Apache 2.0 — see [LICENSE](LICENSE).
