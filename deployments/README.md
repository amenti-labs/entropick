# Deployment Profiles

Each deployment profile is a self-contained setup for one entropy backend.
Use this directory when you want the shortest path to a working server instead
of wiring everything manually.

## Choose a profile

| Profile | Use it when | Runtime |
|---------|-------------|---------|
| [`urandom/`](urandom/) | You want the easiest end-to-end test of the gRPC entropy path. | Docker |
| [`openentropy/`](openentropy/) | You want machine-local hardware noise with no network hop. | Native only |
| [`_template/`](_template/) | You already have your own entropy server and need a starting point. | Usually Docker |

## What you usually edit

Most users only change a few variables:

| Variable | Why you change it |
|----------|-------------------|
| `HF_MODEL` | Pick the model to serve. |
| `HF_TOKEN` | Required only for gated Hugging Face models. |
| `QR_ENTROPY_SOURCE_TYPE` | Usually already set correctly by the profile. Change only if you are repurposing it. |
| `QR_GRPC_SERVER_ADDRESS` | Needed for custom or remote gRPC servers. |
| `QR_OE_SOURCES` | Pick a specific OpenEntropy source such as `clock_jitter`. |
| `QR_FALLBACK_MODE` | Decide whether failures should fall back or hard-fail. |

Everything else is advanced tuning. For the full variable matrix, see [docs/config-reference.md](../docs/config-reference.md).

## Shortest path

### Docker profile

Use this for [`urandom/`](urandom/) and most custom gRPC profiles:

```bash
cd deployments/<profile>
cp .env.example .env
docker compose up --build
```

### Native profile

Use this for [`openentropy/`](openentropy/):

```bash
cd deployments/openentropy
cp .env.example .env
./run-local.sh
```

## Profile-specific guidance

### `urandom/`

Start here if you want the lowest-friction proof that:

1. vLLM is loading.
2. entropick is active.
3. The gRPC entropy path works end to end.

See [`urandom/README.md`](urandom/README.md).

### `openentropy/`

Use this when you want native machine-local entropy sources. This profile is
not containerized because OpenEntropy needs direct host access.

See [`openentropy/README.md`](openentropy/README.md).

### `_template/`

Use this when you already have your own gRPC entropy server and want a clean
starting point rather than editing `urandom/`.

See [`_template/README.md`](_template/README.md).

## Optional Web UI

Docker profiles can also launch [Open WebUI](https://github.com/open-webui/open-webui):

```bash
docker compose --profile ui up --build
```

Then open http://localhost:3000.

If you want UI-level control over sampling parameters, import the filter from [examples/open-webui/](../examples/open-webui/).

## Security notes

- Profile `.env` files may contain API keys or other credentials.
- The `_template/` and `urandom/` profiles contain no secrets and are safe to commit as shipped.
- If your profile contains credentials you do not want in version control, add its folder name to `deployments/.gitignore`.
- entropick never logs the `QR_GRPC_API_KEY` value. Health checks report only whether authentication is enabled.
