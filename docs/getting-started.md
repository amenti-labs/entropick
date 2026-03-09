# Getting Started

This is the step-by-step path for getting entropick to a first successful request.

## Choose your path

| If you want to... | Start here |
|-------------------|------------|
| Prove the full stack works with the least effort | [`deployments/urandom/`](../deployments/urandom/) |
| Run natively with host-local OpenEntropy | [`deployments/openentropy/`](../deployments/openentropy/) |
| Run natively without deployment profiles | The native `.env` path below |
| Connect your own entropy server | [`deployments/_template/`](../deployments/_template/) |

If you are unsure, start with `deployments/urandom/`.

## Fastest working path: `deployments/urandom/`

```bash
cd deployments/urandom
cp .env.example .env
docker compose up --build
```

Then send a request:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "prompt": "The nature of consciousness is",
    "max_tokens": 50
  }'
```

This path proves all of the following in one shot:

1. vLLM is serving.
2. entropick is active.
3. The gRPC entropy fetch path works end to end.

## Native path: direct `vllm serve`

Use this path when you want to run outside Docker.

### 1. Pick a starter `.env`

| File | Use when |
|------|----------|
| [`../.env.example`](../.env.example) | Simplest local setup with system entropy |
| [`../.env.grpc.example`](../.env.grpc.example) | Your entropy source is a gRPC server |
| [`../.env.openentropy.example`](../.env.openentropy.example) | You want local OpenEntropy hardware noise |

### 2. Export the environment and install dependencies

```bash
cp .env.example .env         # or .env.grpc.example / .env.openentropy.example
set -a; source .env; set +a
pip install entropick vllm
```

If you chose the OpenEntropy path, also install:

```bash
pip install openentropy
```

### 3. Start vLLM with entropick enabled

```bash
vllm serve "$HF_MODEL" \
  --logits-processors qr_sampler \
  --dtype half --max-model-len 8096 --gpu-memory-utilization 0.80
```

## Native gRPC variant

Use this when your entropy source is a separate local or remote gRPC service.

```bash
export QR_ENTROPY_SOURCE_TYPE=quantum_grpc
export QR_GRPC_SERVER_ADDRESS=localhost:50051
```

Start with unary mode. Only change transport, retries, TLS, or method paths when you have a concrete reason.

## Native OpenEntropy variant

Use this when you want machine-local hardware noise on the host.

```bash
export QR_ENTROPY_SOURCE_TYPE=openentropy
export QR_OE_SOURCES=clock_jitter
export QR_OE_CONDITIONING=sha256
export QR_FALLBACK_MODE=system
```

For the native launcher script and OpenEntropy-specific notes, see [`../deployments/openentropy/README.md`](../deployments/openentropy/README.md).

## First request

Once the server is running, use the normal OpenAI-compatible API:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "prompt": "The nature of consciousness is",
    "max_tokens": 100
  }'
```

## What to change first

Most users only touch these settings:

- `QR_ENTROPY_SOURCE_TYPE`
- `QR_GRPC_SERVER_ADDRESS`
- `QR_OE_SOURCES`
- `QR_FALLBACK_MODE`
- `QR_TEMPERATURE_STRATEGY`
- `QR_FIXED_TEMPERATURE`
- `QR_TOP_K`
- `QR_TOP_P`

Leave the rest alone until you need a specific experiment or transport behavior.

## Next steps

- Use [`config-reference.md`](config-reference.md) when you need the full variable matrix.
- Use [`how-it-works.md`](how-it-works.md) if you want the stage pipeline and transport internals.
- Use [`experiments.md`](experiments.md) if you are doing controlled research runs.
- Use [`../examples/open-webui/README.md`](../examples/open-webui/README.md) if you want an admin UI for valve-controlled sampling.
