# Framework Support

entropick is organized around vLLM first. Other runtimes are supported through adapter classes that reuse the same core pipeline.

## Support tiers

| Status | Meaning |
|--------|---------|
| `First-class` | The repo's onboarding, deployment docs, and per-request configuration model are optimized for this runtime. |
| `Supported` | An adapter exists and is tested, but the repo is not primarily organized around this runtime. |

## Support matrix

| Runtime | Status | Entry point | Notes |
|---------|--------|-------------|-------|
| **vLLM** | First-class | `qr_sampler.processor:QRSamplerLogitsProcessor` | Recommended path for most users |
| **Transformers** | Supported | `qr_sampler.adapters.QRSamplerLogitsProcessorHF` | Use inside `model.generate(...)` |
| **llama.cpp** | Supported | `qr_sampler.adapters.QRSamplerCallback` | Use with `llama-cpp-python` logits processors |

## vLLM

vLLM is the primary integration target for this repo.

Why it is first-class:

- deployment profiles are built around it
- `qr_*` per-request overrides are documented around vLLM `extra_args`
- the processor is registered via Python entry points

Typical launch:

```bash
vllm serve "$HF_MODEL" \
  --logits-processors qr_sampler \
  --dtype half --max-model-len 8096 --gpu-memory-utilization 0.80
```

Per-request overrides are passed through the normal vLLM request payload:

```json
{
  "extra_args": {
    "qr_top_p": 0.9,
    "qr_min_p": 0.05
  }
}
```

## Hugging Face Transformers

Use the Transformers adapter when you want the same entropick pipeline inside `model.generate(...)`.

```python
from qr_sampler.adapters import QRSamplerLogitsProcessorHF

processor = QRSamplerLogitsProcessorHF()
outputs = model.generate(
    input_ids,
    logits_processor=[processor],
    do_sample=True,
)
```

Notes:

- install `transformers` and `torch` separately; entropick does not vendor them
- adapter instances keep per-session stage state such as selection drift
- configuration can come from the environment or from explicit constructor arguments

## llama.cpp

Use the llama.cpp adapter with `llama-cpp-python` logits processors.

```python
from llama_cpp import Llama, LogitsProcessorList
from qr_sampler.adapters import QRSamplerCallback

llm = Llama(model_path="model.gguf")
callback = QRSamplerCallback()
output = llm.create_completion(
    "Once upon a time",
    logits_processor=LogitsProcessorList([callback]),
)
```

Notes:

- install `llama-cpp-python` separately
- the adapter works on the flat score list passed by `llama-cpp-python`
- configuration can come from the environment or explicit constructor arguments

## Shared behavior across runtimes

All runtimes use the same core components:

- entropy sources
- amplification logic
- temperature strategies
- pipeline stages
- logging and diagnostic records

What differs is how configuration enters the system:

- **vLLM**: process-wide env vars plus per-request `extra_args`
- **Transformers / llama.cpp**: env vars or constructor-level overrides in your Python code

For the full config surface, see [`config-reference.md`](config-reference.md).
