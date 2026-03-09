# Deployment Profile Template

Use this template when you already have your own entropy server and want a clean
starting point. If you just want a working example, start from [`../urandom/`](../urandom/) instead.

## Shortest path

```bash
cp -r deployments/_template deployments/my-server
cd deployments/my-server
cp .env.example .env
docker compose up --build
```

## What you usually change

| Variable | Why you change it |
|----------|-------------------|
| `QR_GRPC_SERVER_ADDRESS` | Point vLLM at your entropy server. |
| `QR_GRPC_METHOD_PATH` | Match your unary RPC method. |
| `QR_GRPC_STREAM_METHOD_PATH` | Match your streaming RPC method, if you support streaming. |
| `QR_GRPC_API_KEY` | Add auth when your server requires it. |
| `HF_MODEL` | Pick the model to serve. |
| `HF_TOKEN` | Required only for gated models. |

If you do not know your method paths yet, keep reading. That is the one part people usually need to look up.

## If your server is co-located in Docker

Uncomment the `entropy-server` service block in `docker-compose.yml` and point it at your image or Dockerfile. See [`../urandom/docker-compose.yml`](../urandom/docker-compose.yml) for a working example.

When the server is co-located, use the Docker service name for `QR_GRPC_SERVER_ADDRESS`, such as `entropy-server:50051`, not `localhost`.

## Finding the right gRPC method path

The method path format is:

```text
/<package>.<Service>/<Method>
```

For example:

```protobuf
package qr_entropy;
service EntropyService {
  rpc GetEntropy (...) ...;
}
```

Produces:

```text
/qr_entropy.EntropyService/GetEntropy
```

The compatibility rule is simple:

- request message: byte count must be protobuf field `1`
- response message: random bytes must be protobuf field `1`

Any proto definition that follows that convention works without code changes.

## Optional Web UI

To also launch [Open WebUI](https://github.com/open-webui/open-webui):

```bash
docker compose --profile ui up --build
```

Open http://localhost:3000.

If you want UI-level control over entropick sampling parameters, use the filter in [examples/open-webui/](../../examples/open-webui/).

| Setting | `.env` variable | Default |
|---------|----------------|---------|
| Port | `OPEN_WEBUI_PORT` | `3000` |
| Authentication | `OPEN_WEBUI_AUTH` | `false` |

Set `OPEN_WEBUI_AUTH=true` if the server is accessible by others.
