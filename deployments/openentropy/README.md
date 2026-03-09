# OpenEntropy Profile

Use this profile when you want machine-local hardware entropy with no network
hop. It runs vLLM natively on the host and uses OpenEntropy directly.

This profile is native-only. OpenEntropy needs direct host access to hardware
timing and platform-specific noise sources, which is why this setup is not
containerized.

## Shortest path

Install the dependencies:

```bash
pip install entropick vllm openentropy
# or, from a repo checkout:
# pip install -e /path/to/entropick
# pip install vllm openentropy
```

Then configure and run:

```bash
cd deployments/openentropy
cp .env.example .env
./run-local.sh
```

To pass extra vLLM flags:

```bash
./run-local.sh --dtype half --max-model-len 8096 --gpu-memory-utilization 0.80
```

`run-local.sh` exports the `.env` file correctly before launching vLLM.

## What you usually change

| Variable | Why you would change it |
|----------|--------------------------|
| `HF_MODEL` | Serve a different model. |
| `HF_TOKEN` | Required only for gated models. |
| `QR_OE_SOURCES` | Pick a specific hardware source such as `clock_jitter` or `counter_beat`. |
| `QR_OE_CONDITIONING` | Usually keep `sha256`; switch only if you need rawer or differently conditioned output. |
| `QR_FALLBACK_MODE` | Decide whether failures fall back or hard-fail. |

You usually do not need to touch parallel collection or timeout settings on day one.

## Good starting defaults

The profile example already uses a sane first-run setup:

```env
QR_ENTROPY_SOURCE_TYPE=openentropy
QR_OE_SOURCES=clock_jitter
QR_OE_CONDITIONING=sha256
QR_FALLBACK_MODE=system
```

`clock_jitter` is a good fast starting point for local runs. `sha256` is the safest default for general use.

## Finding available sources

Available source count depends on the machine and platform. To list what your hardware exposes:

```bash
python3 -c "from openentropy import detect_available_sources; print([s['name'] for s in detect_available_sources()])"
```

For the broader catalog and source descriptions, see the [OpenEntropy Source Catalog](https://github.com/amenti-labs/openentropy/blob/master/docs/SOURCES.md).

## Conditioning modes

| Mode | Use it when | Properties |
|------|-------------|------------|
| `sha256` | General use | Safest default; hashes the collected entropy |
| `vonneumann` | Debiasing experiments | Slower, more uniform |
| `raw` | Research | Minimal processing; preserves the signal most directly |

OpenEntropy settings are process-wide infrastructure and are not per-request overridable through `extra_args`.

## Advanced notes

### Why not Docker?

Docker containers cannot access the same native GPU and hardware-noise surfaces on macOS. OpenEntropy depends on host-level timing and platform behavior, so this profile runs directly on the machine.

### Parallel collection

`QR_OE_PARALLEL=true` collects from multiple sources simultaneously. Leave it on unless you have a specific reason to trade throughput for simpler sequential collection.

## Optional Web UI

If you want [Open WebUI](https://github.com/open-webui/open-webui), run it separately and point it at your local vLLM server on `localhost:8000`:

```bash
docker run -d -p 3000:3000 --name open-webui ghcr.io/open-webui/open-webui:latest
```

If you want UI-level control over entropick parameters, use the filter in [examples/open-webui/](../../examples/open-webui/).

## Next step after this profile

If this works, the next sensible comparisons are:

1. Try another OpenEntropy source via `QR_OE_SOURCES`.
2. Compare against [`urandom/`](../urandom/) or plain system entropy.
3. Open [`config-reference.md`](../../docs/config-reference.md) only if you need the advanced knobs.
