# Extending entropick

Use this document when the built-in entropy backends and default pipeline are not enough.

## Choose an extension path

| If you want to... | Use this approach |
|-------------------|------------------|
| Connect external hardware or a service in another process | Custom gRPC server |
| Add an in-process Python entropy source | Python entropy plugin |
| Change stage ordering or add new stages | Custom pipeline or pipeline stage |

## Approach A: custom gRPC server

This is the recommended path when entropy comes from hardware, firmware, or a service you want to keep outside the inference process.

Copy the template:

```bash
cp examples/servers/qrng_template_server.py my_qrng_server.py
python3 my_qrng_server.py --port 50051
```

Implement the hardware read in `generate()`:

```python
class QRNGHardware:
    def generate(self, n_bytes: int) -> bytes:
        return self._device.read(n_bytes)
```

Protocol compatibility rule:

- request message: byte count must be protobuf field `1`
- response message: random bytes must be protobuf field `1`

Any language with gRPC support works if it follows that wire-format convention.

For a deployment-oriented starting point, use [`../deployments/_template/`](../deployments/_template/).

## Approach B: Python entropy source plugin

Use this when the source is available directly from Python in the same process.

```python
from qr_sampler.entropy.base import EntropySource
from qr_sampler.entropy.registry import register_entropy_source

@register_entropy_source("my_source")
class MySource(EntropySource):
    @property
    def name(self) -> str:
        return "my_source"

    @property
    def is_available(self) -> bool:
        return True

    def get_random_bytes(self, n: int) -> bytes:
        return my_hardware.read(n)

    def close(self) -> None:
        my_hardware.disconnect()
```

Register the source through Python entry points:

```toml
[project.entry-points."qr_sampler.entropy_sources"]
my_source = "my_package:MySource"
```

Then set:

```bash
export QR_ENTROPY_SOURCE_TYPE=my_source
```

## Custom pipelines

Stages are registered and can be composed into custom pipelines.

```python
from qr_sampler.pipeline.registry import StageRegistry

pipeline = [
    StageRegistry.get("temperature")(),
    StageRegistry.get("entropy_fetch")(),
    StageRegistry.get("selection")(),
]
```

Use a custom pipeline when you want a different ordering or a subset of stages.

If you are adding new stages to the project itself, see [`../CONTRIBUTING.md`](../CONTRIBUTING.md) for the full contributor workflow.

## What belongs in CONTRIBUTING

This document is for extension strategy. [`../CONTRIBUTING.md`](../CONTRIBUTING.md) is the place for:

- development setup
- test commands
- lint and type-check expectations
- adding new config fields
- adding new amplifiers, temperature strategies, or stages
- architectural invariants
