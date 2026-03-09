# Experiment Presets

The YAML files in this directory are human-readable presets for common
entropick experiments. Each file contains:

- `name`: short identifier for the preset
- `description`: what the preset is trying to test
- `env`: environment variables to apply for that condition

These files are not auto-loaded by entropick. Use them as templates:

1. Copy the `env:` block into a local `.env` file for a process-wide run.
2. Export selected values in your shell before launching a server.
3. Translate per-request-safe fields to `extra_args` with the `qr_` prefix.

Infrastructure settings remain process-wide. In particular, entropy source
selection and transport settings are not per-request overridable.

Recommended workflow:

1. Start from [`baseline.yaml`](baseline.yaml) for a control condition.
2. Copy one preset and adjust a single mechanism at a time.
3. Record the exact env block used for each run so comparisons stay reproducible.
