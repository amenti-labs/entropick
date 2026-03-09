# Entropy Sources

entropick chooses tokens using entropy fetched after logits are available. This document explains the built-in backends and when to use each one.

## Built-in sources

| Source | `QR_ENTROPY_SOURCE_TYPE` | Use it when |
|--------|--------------------------|-------------|
| **System** | `system` | You want the simplest local baseline using `os.urandom()` |
| **Quantum gRPC** | `quantum_grpc` | Entropy comes from a separate local or remote gRPC service |
| **OpenEntropy** | `openentropy` | You want machine-local hardware noise on the host |
| **Mock uniform** | `mock_uniform` | You need a deterministic or controllable test source |
| **Sham QRNG** | `sham_qrng` | You want a control condition with simulated device latency |

## Which source should most users start with?

- Start with `system` if you want the simplest local baseline.
- Start with `quantum_grpc` if your real entropy source lives in another process, another container, or another machine.
- Start with `openentropy` if your machine itself is the entropy source.
- Use `mock_uniform` only for testing.
- Use `sham_qrng` when you need a double-blind control that looks like a hardware source in timing terms.

## OpenEntropy

[OpenEntropy](https://github.com/amenti-labs/openentropy) harvests entropy from platform-dependent hardware noise sources on the local machine.

Typical setup:

```bash
pip install openentropy

export QR_ENTROPY_SOURCE_TYPE=openentropy
export QR_OE_SOURCES=clock_jitter
export QR_OE_CONDITIONING=sha256
```

Practical guidance:

- `clock_jitter` is a good fast default for local runs
- `sha256` is the safest default conditioning mode for general use
- available sources depend on the hardware and operating system

List available sources on the current machine:

```bash
python3 -c "from openentropy import detect_available_sources; print([s['name'] for s in detect_available_sources()])"
```

For the native launcher and OpenEntropy-specific deployment notes, see [`../deployments/openentropy/README.md`](../deployments/openentropy/README.md).

## Fallback behavior

`FallbackEntropySource` wraps a primary source with automatic failover.

| `QR_FALLBACK_MODE` | Behavior |
|--------------------|----------|
| `system` | Fall back to `os.urandom()` |
| `mock_uniform` | Fall back to the mock source |
| `error` | Raise immediately with no fallback |

Recommendation:

- use `system` when you want resilience
- use `error` when you need strict experimental or operational failure semantics

## Custom sources

If the built-in backends are not enough, add your own source via:

- a custom gRPC server
- an in-process Python entropy source plugin

See [`extending.md`](extending.md).
