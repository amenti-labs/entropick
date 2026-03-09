# Open WebUI Integration

[Open WebUI](https://github.com/open-webui/open-webui) provides a ChatGPT-style
web interface for chatting with models served by vLLM. Every entropick
deployment profile includes it as an optional Docker Compose service.

This directory contains a **filter function** that lets you control the common
entropick per-request parameters (temperature, top-k, top-p, sample count, etc.)
directly from the Open WebUI admin panel — no manual API editing needed.

## Starting Open WebUI

From any deployment profile directory, add `--profile ui`:

```bash
cd deployments/urandom          # or _template, your-profile
cp .env.example .env
docker compose --profile ui up --build
```

Open http://localhost:3000. Without `--profile ui`, Open WebUI does not start
and the deployment behaves exactly as before.

## Installing the filter function

The filter function ships as two files:

| File | Purpose |
|------|---------|
| `qr_sampler_filter.py` | Human-readable source code |
| `qr_sampler_filter.json` | Open WebUI importable JSON |

### Import steps

1. Open http://localhost:3000 and log in (first user becomes admin).
2. Go to **Admin Panel > Functions** (or **Workspace > Functions**).
3. Click **Import** (the upload icon).
4. Select `qr_sampler_filter.json` from this directory.
5. Toggle the imported function to **Global** so it applies to all models.

The filter is now active. Every chat message will include entropick parameters
in requests sent to vLLM.

### Alternative: paste the source

If you prefer not to use the JSON import:

1. Go to **Admin Panel > Functions** and click **Create a new function**.
2. Set the type to **Filter**.
3. Copy the contents of `qr_sampler_filter.py` into the code editor.
4. Save and toggle to **Global**.

## Configuring parameters (Valves)

After importing the filter, click the **gear icon** next to it to open the
Valves panel. Each Valve maps to an entropick per-request parameter:

### Filter control

| Valve | Default | Description |
|-------|---------|-------------|
| `priority` | `0` | Filter execution order (lower runs first). |
| `enable_qr_sampling` | `true` | Master switch. Set to `false` to pass requests through unmodified. |

### Token selection

| Valve | Default | Maps to | Description |
|-------|---------|---------|-------------|
| `top_k` | `0` | `qr_top_k` | Keep only the k most probable tokens (0 disables). |
| `top_p` | `1.0` | `qr_top_p` | Nucleus sampling threshold (1.0 disables). |

### Temperature

| Valve | Default | Maps to | Description |
|-------|---------|---------|-------------|
| `temperature_strategy` | `fixed` | `qr_temperature_strategy` | `fixed` or `edt` (entropy-dependent). |
| `fixed_temperature` | `0.7` | `qr_fixed_temperature` | Constant temperature (fixed strategy). |
| `edt_base_temp` | `0.8` | `qr_edt_base_temp` | Base coefficient for EDT. |
| `edt_exponent` | `0.5` | `qr_edt_exponent` | Power-law exponent for EDT. |
| `edt_min_temp` | `0.1` | `qr_edt_min_temp` | EDT temperature floor. |
| `edt_max_temp` | `2.0` | `qr_edt_max_temp` | EDT temperature ceiling. |

### Signal amplification

| Valve | Default | Maps to | Description |
|-------|---------|---------|-------------|
| `signal_amplifier_type` | `zscore_mean` | `qr_signal_amplifier_type` | Amplification algorithm. |
| `sample_count` | `20480` | `qr_sample_count` | Entropy bytes fetched per token. |

The filter intentionally does **not** expose every possible `qr_*` field.
Less common experimental knobs can still be sent directly via API calls if you
need them.

## How it works

```
User types message in Open WebUI
  |
  +-> Open WebUI sends request to vLLM (/v1/chat/completions)
  |
  +-> Filter inlet() runs BEFORE the request reaches vLLM:
  |     - Reads current Valve values
  |     - Adds qr_top_k, qr_top_p, qr_temperature_strategy, etc.
  |       as top-level keys in the request body
  |
  +-> vLLM receives the request:
  |     - Unknown top-level keys become SamplingParams.extra_args
  |     - entropick's resolve_config() reads qr_* from extra_args
  |     - Token sampling uses the parameters from the Valves
  |
  +-> Response streams back through Open WebUI to the user
```

Infrastructure settings (gRPC server address, fallback mode, OpenEntropy
settings, logging, diagnostic capture, etc.) are **not** exposed as Valves.
They cannot change per-request and are controlled by environment variables on
the vLLM container.

## What is NOT controlled by the filter

The filter only manages per-request sampling parameters. These settings are
configured via environment variables in your `.env` file and apply to all
requests:

- Entropy source type and gRPC server address
- gRPC transport mode, timeout, and retry count
- Fallback mode
- Circuit breaker thresholds
- API key authentication

See the [configuration reference](../../docs/config-reference.md) for the full list.

## Disabling the filter

To stop injecting entropick parameters without removing the filter:

1. Open the Valves panel (gear icon).
2. Set `enable_qr_sampling` to `false`.

Requests will pass through to vLLM unmodified, and entropick will use its
default configuration from environment variables.

## Customizing the UI port

Set `OPEN_WEBUI_PORT` in your `.env` file:

```
OPEN_WEBUI_PORT=8080
```

Then access Open WebUI at http://localhost:8080.

## Authentication

By default, Open WebUI runs without authentication (`OPEN_WEBUI_AUTH=false`).
This is convenient for local development. For shared or public servers, enable
authentication:

```
OPEN_WEBUI_AUTH=true
```

The first user to sign up becomes the admin.
