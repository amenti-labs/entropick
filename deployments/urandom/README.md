# urandom Profile

Use this profile when you want the easiest possible proof that the full gRPC
entropy path works. It runs a tiny entropy server backed by `os.urandom()`
next to vLLM inside Docker.

## Shortest path

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

## What you usually change

| Variable | Why you would change it |
|----------|--------------------------|
| `HF_MODEL` | Serve a different model. |
| `HF_TOKEN` | Required only for gated models. |
| `QR_GRPC_MODE` | Switch to `bidi_streaming` if you want lower latency. |
| `QR_FALLBACK_MODE` | Choose whether failures fall back or hard-fail. |

You do not need to edit method paths, API key headers, or circuit breaker settings for normal use.

## What this profile is good for

- Validating that entropick is wired into vLLM correctly.
- Testing the full gRPC fetch path without real hardware.
- Measuring the overhead of gRPC-based entropy relative to simpler local baselines.

## Common upgrade

If you want lower latency, set:

```env
QR_GRPC_MODE=bidi_streaming
```

The bundled urandom server supports `unary`, `server_streaming`, and `bidi_streaming`.

## What this runs

- `entropy-server`: lightweight gRPC server container running `simple_urandom_server.py`
- `vllm`: vLLM with entropick enabled, fetching entropy from `entropy-server:50051`

The entropy server starts first and vLLM waits on it via `depends_on`.

## Optional Web UI

To also start [Open WebUI](https://github.com/open-webui/open-webui):

```bash
docker compose --profile ui up --build
```

Open http://localhost:3000.

If you want UI-level control over sampling parameters, import [`qr_sampler_filter.json`](../../examples/open-webui/qr_sampler_filter.json) and follow [examples/open-webui/README.md](../../examples/open-webui/README.md).

| Setting | `.env` variable | Default |
|---------|----------------|---------|
| Port | `OPEN_WEBUI_PORT` | `3000` |
| Authentication | `OPEN_WEBUI_AUTH` | `false` |

Set `OPEN_WEBUI_AUTH=true` if the server is accessible by others.

## Next step after this profile

If this works and you want a real entropy source:

1. Use [`_template/`](../_template/) for your own gRPC server.
2. Use [`openentropy/`](../openentropy/) for native machine-local noise sources.
