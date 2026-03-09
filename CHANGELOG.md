# Changelog

All notable changes to entropick will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Rebranded from qr-sampler to entropick

### Added

- vLLM V1 LogitsProcessor plugin (`QRSamplerLogitsProcessor`) with batch-level processing
- Pydantic-settings configuration system with `QR_` env prefix and per-request overrides
- Entropy source subsystem with ABC, auto-discovery registry, and entry-point support
  - `QuantumGrpcSource`: gRPC client with unary, server-streaming, and bidirectional transport modes
  - `SystemEntropySource`: `os.urandom()` wrapper
  - `MockUniformSource`: configurable test source with seed and bias control
  - `FallbackEntropySource`: automatic failover wrapper
- Adaptive circuit breaker for gRPC source (rolling P99, half-open recovery)
- Z-score mean signal amplifier (`zscore_mean`) for bias-preserving entropy-to-uniform mapping
- Temperature strategies: fixed and entropy-dependent (EDT) with Shannon entropy computation
- CDF-based token selector with top-k, top-p (nucleus) filtering
- Diagnostic logging subsystem with three verbosity levels and in-memory record storage
- gRPC proto definition and hand-written stubs for `EntropyService`
- Reference entropy servers: `simple_urandom_server.py`, `qrng_template_server.py`
- Docker and docker-compose deployment templates
- systemd service unit and environment file
- Apache 2.0 license
- Pre-commit configuration with ruff, mypy, bandit, and standard hooks
- Comprehensive test suite with statistical validation tests
