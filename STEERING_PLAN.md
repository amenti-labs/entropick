# qr-steering: Planning Document (v4)

## Vision

A comprehensive, generic open-source LLM activation steering library — with first-class
QRNG integration for consciousness research. Useful to anyone doing activation steering
(researchers, alignment engineers, hobbyists). Our unique contribution is quantum-random
modulation of the steering process.

**Relationship to qr-sampler**: qr-sampler is "output control" (logits-level). qr-steering
is "state control" (hidden-state-level). Together they form a dual-surface quantum-influence
pipeline — two independent intervention points in a single forward pass. qr-steering imports
qr-sampler as a dependency for entropy sources, amplification, and analysis infrastructure.

**Runtime**: HuggingFace Transformers + PyTorch hooks, with first-class **nnsight**
interoperability for MI researchers. Optional vLLM integration via forked engine (Phase 7).

**Design philosophy**: "It just works." Researchers should go from `pip install` to steered
generation in under 10 lines of code. The library should feel like a natural extension of
the MI researcher's existing toolkit (nnsight, SAELens, Neuronpedia), not a parallel universe.

---

## Why This Project is Novel

**Nobody has combined activation steering with quantum entropy for consciousness research.**

The existing landscape:
- **Consciousness-RNG research** (Global Consciousness Project, PEAR, IONS) uses simple
  bit-distribution analysis on QRNG output. 17-year GCP experiment: Z=7.31 across 500
  pre-registered events. Holmberg (2025): Bayesian entropy framework with TrueRNG,
  t=4.347, p<0.001. Key finding: "the source of randomness matters" — quantum vs
  pseudo-random distinction is fundamental to detecting mind-matter effects.
- **Activation steering research** (Golden Gate Claude, EasySteer, AUSteer, PID Steering)
  uses fixed vectors and fixed coefficients. No stochastic modulation. No quantum entropy.
- **Goodfire** ($1.25B, the commercial leader) deprecated their SAE API and pivoted to
  closed-source "intentional model design." The open-source steering space is wide open.

**Our project bridges these two worlds**: instead of analyzing bits from a QRNG, we use
LLM hidden states as a high-dimensional "detector" for consciousness-quantum interactions.
The model's 4096-dimensional activation space is a far richer signal than binary bit streams.

---

## Researcher Ergonomics (v4 — new section)

### The MI Researcher's Existing Toolkit

MI researchers in 2026 work primarily in **Jupyter notebooks / Google Colab**. Their
standard imports are:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nnsight import LanguageModel          # intervention API
from sae_lens import SAE                    # SAE loading
import neuronpedia                          # feature discovery
```

The typical workflow is: load model → find features → apply interventions → generate →
compare outputs. Everything happens interactively in a notebook.

### How qr-steering Fits In

qr-steering should feel like a natural **fourth tool** alongside nnsight, SAELens, and
Neuronpedia. Not a replacement — an enhancement. This means:

1. **Works alongside nnsight** — can be used inside `with model.trace()` contexts, or
   standalone with its own hook management. Researchers don't have to choose.
2. **Loads SAEs via SAELens** — `sae_lens.SAE.from_pretrained()` format is the standard.
   We use their weight format, don't reinvent it.
3. **Uses familiar naming** — TransformerLens hook names (`resid_pre`, `resid_post`,
   `mlp_out`, `attn_out`) are universally understood. We adopt them.
4. **Notebook-first documentation** — every feature ships with a Colab notebook.
   README has "Open in Colab" badges. Examples are runnable, not just readable.
5. **Minimal boilerplate** — 5 lines to steer, not 50.

### The 5-Line Steering Promise

```python
from qr_steering import SteeringEngine

engine = SteeringEngine.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
engine.steer(concept="honesty", source="neuronpedia", strength=8.0)
print(engine.generate("Tell me about yourself", max_new_tokens=128))
```

### Context Manager Pattern (researcher expectation)

MI researchers universally expect a context manager for temporary steering
(established by `steering-vectors`, `repeng`, and `llm_steer` libraries):

```python
# Temporary steering — automatically removed on exit
with engine.steer(concept="honesty", source="neuronpedia", strength=8.0):
    steered = engine.generate("Tell me about yourself", max_new_tokens=128)
baseline = engine.generate("Tell me about yourself", max_new_tokens=128)  # unsteered
```

Both persistent (`engine.steer(...)` + `engine.clear()`) and context manager modes are
supported. Negative layer indexing works: `layer=-5` means 5th from last.

### The 3-Line QRNG Promise

```python
engine.enable_quantum(entropy_source="system", modulator="alpha", beta=0.2)
# Every generation now uses quantum-modulated steering strength
print(engine.generate("Tell me about yourself", max_new_tokens=128))
```

### nnsight Interop

For researchers already using nnsight, qr-steering integrates seamlessly:

```python
from nnsight import LanguageModel
from qr_steering.nnsight import SteeringIntervention

model = LanguageModel("meta-llama/Llama-3.1-8B-Instruct")
intervention = SteeringIntervention.from_neuronpedia("honesty", model_id="llama-3.1-8b")

with model.trace("Tell me about yourself", max_new_tokens=128) as tracer:
    # Apply steering inside nnsight's tracing context
    intervention.apply(tracer, layer=15, strength=8.0)
    output = model.output.save()

print(model.tokenizer.decode(output))
```

---

## Existing Ecosystem (build vs. depend)

| Library | Stars | What it does | Our relationship |
|---------|-------|-------------|-----------------|
| **nnsight** (NDIF) | ~800 | Next-gen MI library. Wraps any PyTorch model. Tracer context for interventions. Remote execution via NDIF. **v0.6.2 (Mar 2026).** First-class vLLM support with async streaming. 2.4-3.9x faster than 0.5. | **Primary interop target.** Provide `SteeringIntervention` class that works inside nnsight traces. vLLM path via nnsight (not forked engine). |
| **nnterp** | ~200 | Standardized wrapper around nnsight. Unified interface across 50+ model variants, 16 architecture families. | **Key reference for architecture detection.** Study their model family mapping. Possibly depend on for auto-detection. |
| **SAELens** | 1.1K | SAE training + loading + analysis. **v6.37.6 (Feb 2026).** Neuronpedia integration built-in. safetensors + cfg.json format. Covers Gemma 2/3, Llama 3/3.1/3.3, GPT-2, Qwen, Mistral, Pythia. | **Required dependency** for SAE vector extraction. Use `SAE.from_pretrained()` + `sae.W_dec[feature_idx]`. |
| **EasySteer** (ZJU) | 158 | Steering on vLLM. Forked engine with hook integration. 10-22x faster than HF. | **Key reference for Phase 7** (vLLM path). Study their approach. |
| **TransformerLens** | 2.1K | MI library. Reimplements architectures. v3 alpha (Sep 2025). Best <=9B. | **Naming conventions only.** Use their hook name vocabulary (`resid_pre`, etc.) for familiarity. Don't depend on it. |
| **pyvene** (Stanford) | 868 | Declarative intervention framework. Used in AxBench. | Monitor. Too abstract for our needs. |
| **neuronpedia-python** | 5 | Python API client for Neuronpedia. | **Direct dependency** for feature discovery. |
| **steering-vectors** | 26 | Clean `train_steering_vector()` / `sv.apply(model)` context manager API. Works on native HF models. | Good Phase 1 code reference. Context manager pattern is the standard researchers expect. |
| **repeng** | ~200 | `ControlVector.train()` in <60 seconds. `ControlModel.set_control(vec, strength)`. | Reference for fast vector training UX. |
| **Dialz** (Cardiff NLP) | ~50 | PCA + mean-difference vector computation with visualization. ACL 2025 demo. | Reference for vector computation methods. |
| **RISER** (Jan 2026) | N/A | Dynamic vector composition: K=6 reasoning primitives + Router MLP. Gumbel-Sigmoid selection, GRPO training. 3.4-6.5% zero-shot improvement. | Key reference for Q5 (feature walk) and multi-concept composition. Dynamic selection is closely related to our quantum feature selection. |
| **Goodfire** | $1.25B | Commercial. Deprecated SAE API. Going closed. | Our open-source positioning is complementary. |

**Key v4 decision**: Build on **raw PyTorch hooks** for maximum control and vLLM
compatibility, but provide first-class **nnsight interop** so MI researchers can use
qr-steering inside their existing nnsight workflows. Use **SAELens** for SAE loading
(don't reinvent). Use **nnterp-style** architecture auto-detection (eliminate manual
ModelAdapter classes).

---

## The Steering Landscape (as of 2026)

### What is activation steering?

Instead of modifying prompts or fine-tuning weights, you modify the model's **hidden states
during the forward pass**. A hook on transformer layer `l` intercepts the activation vector
`x^l` and transforms it — biasing the model's "thinking" toward or away from a concept.

### Vector Discovery Methods

| Method | How it works | Phase |
|--------|-------------|-------|
| **Custom Vectors** | Load from file (.pt/.npy/.safetensors). | 1 |
| **SAE Decoder** | Extract column from SAE decoder matrix via SAELens. Each column = one interpretable feature. 131K+ features/layer. Browsable on Neuronpedia. | 2 |
| **Contrastive Activation Addition (CAA)** | Mean activation diff between positive/negative prompt pairs. (Turner et al. 2023) | 2 |
| **SAE-Targeted Steering (SAE-TS)** | Optimize vector to target specific SAE features while minimizing side effects. (Chalnev et al. 2024) | 5 |
| **AUSteer Discriminative AUs** | Identify discriminative atomic units via activation momenta on contrastive samples. Steer fewer dimensions, achieve more. (Feng et al. ICLR 2026) | 6 |

### Steering Application Methods

| Method | Formula | Key insight | Phase |
|--------|---------|------------|-------|
| **Additive** | `x = x + α·v` | Simplest. α ≈ half layer activation magnitude. | 1 |
| **Clamping** | `x = x - (x·v)v + α·v` | Removes existing projection first. More stable. Anthropic's method. | 2 |
| **Token-wise Decay** | `α_t = α_0 · decay^t` | Prevents degenerate repetition. Top-1 latent > top-k. (Xie 2025) | 3 |
| **PID Steering** | P: `α·v`, I: accumulated error across layers, D: counteracts rapid changes | Control theory. Standard steering = P-only controller. PID adds stability guarantees. (Nguyen et al. ICLR 2026) | 3 |
| **Fine-Grained (AUSteer)** | Per-AU adaptive strength on discriminative dimensions only | "Steering less achieves more." Different dimensions control different token distributions. (Feng et al. ICLR 2026) | 6 |
| **Conceptors** | Soft projection `x = Cx + α·v`. Boolean AND/OR/NOT. (Abreu et al. 2025) | Compositional multi-concept. | 5 |

### Layer Strategy

- **Early layers (0–25%)**: Input token features. Low-level manipulation.
- **Middle layers (25–75%)**: Semantic/conceptual features. **Best for concept steering.**
- **Late layers (75–100%)**: Output token predictions. Direct output bias.
- **Multi-layer**: `α_l = α̂ · l` (scale by depth). PID I-term handles this automatically.
- **Llama 3.1 8B** (32 layers): Layer 15 sweet spot. SAEs at layers 3, 7, 11, 15, 19, 23, 27.

### Key Research Findings

- **AxBench (ICML 2025)**: Prompting > finetuning > ReFT-r1 > DiffMean > SAEs for steering.
- **Eiffel Tower Llama (2025)**: Clamping > addition. α ≈ 7–9 for Llama 8B L15. <5 no effect, >10 gibberish.
- **AUSteer (ICLR 2026)**: Block-level activations entangle beneficial + harmful features. Decompose to atomic units (individual dimensions). Steer fewer, targeted AUs → better results.
- **PID Steering (ICLR 2026)**: Standard `x + α·v` = proportional-only controller. Adding integral (cross-layer accumulation) and derivative (overshoot prevention) terms gives stability guarantees and consistent outperformance.
- **Token-wise Decay (2025)**: Constant steering → degenerate repetition. Decay fixes it. Top-1 SAE latent > top-k.
- **EasySteer (2025)**: Steering works on vLLM with forked engine. 10-22x faster than HF. Continuous batching compatible.
- **Steering Awareness (2026)**: Models can detect steering (95.5%) but detection → *more* susceptible.
- **Consciousness-RNG literature**: GCP Z=7.31 (17 years), Holmberg t=4.347 p<0.001. Source of randomness (quantum vs pseudo) is central.

---

## QRNG Injection Points (our unique contribution)

Everywhere current steering uses a **fixed scalar or fixed vector**, we substitute a
**quantum-modulated value**. This creates a quantum-influence surface at the hidden-state
level — independent of qr-sampler's logits-level surface.

### Q1: α Modulation (steering strength) — Phase 3
```
α_effective = α_base + β · qrng_uniform()
```
Per-token quantum-varied steering intensity. Highest-priority injection point.

### Q2: PID Gain Modulation — Phase 4
Modulate the P, I, or D gains of PID steering with QRNG. The integral term's
accumulated error becomes a quantum-influenced trajectory through activation space.

### Q3: AU-Level Quantum Selection — Phase 6
Instead of modulating the full vector, use QRNG to select *which* discriminative AUs
receive steering on each token. Combines AUSteer's "steer less" insight with quantum
feature selection.

### Q4: Directional Perturbation — Phase 3
```
v_perturbed = normalize(v + σ · qrng_noise_vector)
```
Quantum noise in the steering direction. Explores nearby concept representations.

### Q5: Feature Walk (stochastic feature selection) — Phase 4
Multiple SAE features map to one concept. QRNG selects which feature(s) activate
per token. Quantum walk through concept space.

### Q6: Layer Targeting — Phase 4
QRNG varies which layer(s) receive the steering hook per token.

### Q7: Vectorless Quantum Noise Injection — Phase 3
No concept vector. QRNG-sourced noise directly into the residual stream. Pure
quantum perturbation. Most direct consciousness-research test.

### Q8: Decay Rate Modulation — Phase 4
```
decay_t = decay_base + δ · qrng_uniform()
```
Quantum-varied steering fade. Non-deterministic persistence patterns.

---

## Steering Without a Concept Vector

Approaches for consciousness research where we test quantum influence without semantic bias:

1. **Pure Quantum Noise (Q7)** — Random vectors from QRNG into residual stream. No target. Model's response to unstructured quantum perturbation is the experimental signal.
2. **Activation Magnitude Scaling** — `x = x * (1 + β · qrng_uniform())`. Amplifies/dampens current "thinking."
3. **Attention Head Gating** — QRNG stochastically gates heads on/off.
4. **AU-Level Noise** — Apply QRNG noise only to dimensions identified as discriminative (via AUSteer methodology). Targeted perturbation in the model's decision-relevant subspace.
5. **Entropy-Adaptive Steering** — Compute hidden-state entropy. High uncertainty → stronger quantum influence. Low → back off.

---

## Architecture Design

### Guiding Principles

1. **nnsight-compatible, not nnsight-dependent.** Raw PyTorch hooks as the core. nnsight interop as a first-class integration layer. Researchers can use either.
2. **nnterp-style architecture detection.** Auto-detect model family from `model.config.model_type`. No per-architecture adapter classes to maintain. One mapping dict.
3. **SAELens for SAE loading.** Don't reinvent weight loading. Use their format, their API.
4. **Registry pattern** for everything (mirrors qr-sampler).
5. **Generic first, QRNG second.** Useful without QRNG.
6. **PID steering as first-class method**, not an afterthought.
7. **Notebook-first development.** Every feature ships with a runnable Colab notebook.

### Package Structure

```
qr-steering/
├── src/qr_steering/
│   ├── __init__.py
│   ├── config.py                    # SteeringConfig (pydantic-settings, QRS_ prefix)
│   ├── exceptions.py                # SteeringError hierarchy
│   │
│   ├── vectors/                     # Vector Discovery
│   │   ├── __init__.py
│   │   ├── base.py                  # VectorSource ABC
│   │   ├── registry.py              # VectorSourceRegistry
│   │   ├── file_source.py           # Load from .pt / .safetensors / .npy
│   │   ├── sae_source.py            # SAELens integration: SAE.from_pretrained() → decoder column
│   │   ├── contrastive.py           # CAA from prompt pairs
│   │   └── neuronpedia.py           # Neuronpedia API: search, download
│   │
│   ├── methods/                     # Application Methods
│   │   ├── __init__.py
│   │   ├── base.py                  # SteeringMethod ABC
│   │   ├── registry.py              # SteeringMethodRegistry
│   │   ├── additive.py              # x = x + α·v
│   │   ├── clamping.py              # Remove projection + add target
│   │   ├── decaying.py              # Token-wise decaying strength
│   │   └── pid.py                   # PID controller (P + I + D terms)
│   │
│   ├── quantum/                     # QRNG Modulation (optional, requires qr-sampler)
│   │   ├── __init__.py
│   │   ├── base.py                  # QuantumModulator ABC
│   │   ├── registry.py              # ModulatorRegistry
│   │   ├── alpha_mod.py             # Q1: strength modulation
│   │   ├── directional.py           # Q4: vector perturbation
│   │   ├── noise_injection.py       # Q7: vectorless residual stream noise
│   │   ├── pid_mod.py               # Q2: PID gain modulation
│   │   ├── feature_walk.py          # Q5: stochastic feature selection
│   │   └── au_selection.py          # Q3: AU-level quantum selection
│   │
│   ├── hooks/                       # PyTorch Hook Management (core intervention layer)
│   │   ├── __init__.py
│   │   ├── manager.py               # HookManager: register/remove/lifecycle
│   │   ├── context.py               # SteeringContext: per-generation mutable state
│   │   └── models.py                # Model family detection (nnterp-style, single dict)
│   │
│   ├── nnsight/                     # nnsight Interop Layer (optional dependency)
│   │   ├── __init__.py
│   │   └── intervention.py          # SteeringIntervention: works inside model.trace()
│   │
│   ├── engine.py                    # SteeringEngine: top-level API
│   │
│   └── evaluation/                  # Steering Quality Measurement
│       ├── __init__.py
│       ├── metrics.py               # Repetition, perplexity, KL divergence
│       └── sweep.py                 # α sweep / grid search
│
├── tests/
│   ├── conftest.py
│   ├── helpers.py                   # Mock models, test vectors
│   ├── test_config.py
│   ├── test_hooks/
│   ├── test_vectors/
│   ├── test_methods/
│   │   ├── test_additive.py
│   │   ├── test_clamping.py
│   │   ├── test_pid.py
│   │   └── test_decaying.py
│   ├── test_quantum/
│   ├── test_nnsight/                # nnsight interop tests (require nnsight)
│   └── test_engine.py
│
├── notebooks/                       # Colab-ready Jupyter notebooks (primary docs)
│   ├── 01_quickstart.ipynb          # 5 minutes: install, load, steer, generate
│   ├── 02_neuronpedia_discovery.ipynb # Find features, browse, download
│   ├── 03_contrastive_vectors.ipynb # Build CAA vectors from prompt pairs
│   ├── 04_sae_steering.ipynb        # SAELens → steering vectors
│   ├── 05_pid_steering.ipynb        # PID controller demo + comparison with additive
│   ├── 06_qrng_modulation.ipynb     # QRNG-modulated steering + consciousness protocol
│   ├── 07_vectorless_noise.ipynb    # Pure quantum noise injection
│   ├── 08_nnsight_interop.ipynb     # Using qr-steering inside nnsight workflows
│   └── 09_dual_surface.ipynb        # qr-steering + qr-sampler combined experiment
│
├── examples/                        # Standalone scripts (for non-notebook users)
│   ├── basic_steering.py
│   ├── pid_steering.py
│   ├── qrng_alpha_mod.py
│   └── dual_surface.py
│
├── experiments/                     # YAML experiment presets
│   ├── baseline.yaml
│   ├── sae_additive.yaml
│   ├── pid_steering.yaml
│   ├── qrng_alpha.yaml
│   ├── qrng_noise.yaml
│   └── dual_surface.yaml
│
├── pyproject.toml
├── CLAUDE.md
└── README.md
```

### Model Architecture Detection (nnterp-style)

Instead of maintaining per-model `ModelAdapter` classes (v3), use a single mapping dict
that maps `model.config.model_type` to layer access patterns:

```python
# hooks/models.py
MODEL_FAMILIES: dict[str, ModelConfig] = {
    "llama":  ModelConfig(layers="model.layers", resid="input_layernorm", hidden="hidden_size"),
    "gemma":  ModelConfig(layers="model.layers", resid="input_layernorm", hidden="hidden_size"),
    "gemma2": ModelConfig(layers="model.layers", resid="input_layernorm", hidden="hidden_size"),
    "gpt2":   ModelConfig(layers="transformer.h", resid="ln_1", hidden="n_embd"),
    "qwen2":  ModelConfig(layers="model.layers", resid="input_layernorm", hidden="hidden_size"),
    "mistral": ModelConfig(layers="model.layers", resid="input_layernorm", hidden="hidden_size"),
    "phi3":   ModelConfig(layers="model.layers", resid="input_layernorm", hidden="hidden_size"),
}

def detect_model(model) -> ModelConfig:
    """Auto-detect model family from config. Falls back to generic Llama-like layout."""
    model_type = getattr(model.config, "model_type", "")
    return MODEL_FAMILIES.get(model_type, MODEL_FAMILIES["llama"])
```

This approach:
- Covers 90%+ of models researchers use (Llama, Gemma, GPT-2, Qwen, Mistral, Phi)
- Adding a new model = one dict entry, not a whole class
- nnterp supports 16 families — we can add more as needed
- Falls back gracefully to Llama-like layout (most modern models use this)

### Key ABCs

```python
class VectorSource(ABC):
    """Discovers/produces steering vectors."""
    name: str
    @abstractmethod
    def get_vectors(self, model, concept: str, **kwargs) -> list[SteeringVector]: ...

@dataclass(frozen=True)
class SteeringVector:
    """A single steering direction with metadata."""
    vector: torch.Tensor          # Normalized, shape [hidden_dim]
    layer: int                    # Target layer index
    source_name: str              # Which VectorSource produced this
    feature_id: str | None = None # Neuronpedia/SAE feature ID
    description: str = ""


class SteeringMethod(ABC):
    """Applies steering to hidden states. Stateful (may track token count, PID state)."""
    name: str
    @abstractmethod
    def apply(self, hidden_states: Tensor, vector: Tensor, strength: float,
              token_idx: int = 0, **kwargs) -> Tensor: ...
    def reset(self) -> None: ...  # Reset state between generations


class QuantumModulator(ABC):
    """Modulates a steering parameter using QRNG."""
    name: str
    @abstractmethod
    def modulate(self, base_value: float, entropy_source: EntropySource,
                 config: SteeringConfig) -> float: ...
```

### Top-Level API: SteeringEngine

```python
from qr_steering import SteeringEngine

engine = SteeringEngine.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Simple additive steering (5 lines total)
engine.steer(vector_file="vectors/honesty_l15.pt", method="additive", strength=8.0)
output = engine.generate("Tell me about yourself", max_new_tokens=128)
engine.clear()

# Neuronpedia concept search → steer (auto-downloads SAE feature)
engine.steer(concept="honesty", source="neuronpedia", method="clamping", strength=8.0)

# PID steering (more stable)
engine.steer(
    concept="honesty", source="neuronpedia",
    method="pid", strength=8.0,
    pid_ki=0.3, pid_kd=0.1,
)

# QRNG-modulated (consciousness research)
engine.enable_quantum(
    entropy_source="system",      # or "openentropy", "quantum" (gRPC)
    modulator="alpha",
    beta=0.2,                     # modulation magnitude
)
# Now every generate() call uses quantum-modulated steering

# Dual-surface (qr-steering + qr-sampler)
engine.enable_dual_surface(qr_sampler_config={
    "entropy_source_type": "openentropy",
    "oe_sources": "clock_jitter",
})
# Both hidden states AND logits are quantum-influenced
```

### nnsight Interop API

```python
from nnsight import LanguageModel
from qr_steering.nnsight import SteeringIntervention

model = LanguageModel("meta-llama/Llama-3.1-8B-Instruct")

# Create intervention from Neuronpedia feature
intervention = SteeringIntervention.from_neuronpedia(
    concept="honesty",
    model_id="llama-3.1-8b",
    method="clamping",
    strength=8.0,
)

# Use inside nnsight's tracing context — works with nnsight's deferred execution
with model.trace("Tell me about yourself", max_new_tokens=128) as tracer:
    intervention.apply(tracer, layer=15)
    output = model.output.save()

# With QRNG modulation inside nnsight
intervention.enable_quantum(entropy_source="system", modulator="alpha", beta=0.2)
with model.trace("Tell me about yourself", max_new_tokens=128) as tracer:
    intervention.apply(tracer, layer=15)
    output = model.output.save()

# vLLM mode (nnsight 0.6+ — async streaming with interventions)
from nnsight.modeling.vllm import VLLM
vllm_model = VLLM("meta-llama/Llama-3.1-8B-Instruct", mode="async")
with vllm_model.trace(prompt, max_tokens=256) as tracer:
    intervention.apply(tracer, layer=15)
async for output in tracer.backend():
    print(output.outputs[0].text, end="", flush=True)
```

### Hook Flow (per token)

```
Forward pass reaches hooked layer l
  │
  ├─ HookManager intercepts layer output (hidden_states, ...)
  │
  ├─ For each SteeringVector targeting this layer:
  │     │
  │     ├─ If quantum modulation:
  │     │     ├─ Fetch entropy from EntropySource
  │     │     ├─ Modulator.modulate(α_base) → α_eff
  │     │     └─ (optional) Modulator.perturb(v) → v_eff
  │     │
  │     ├─ SteeringMethod.apply(hidden_states, v, α, token_idx)
  │     │     ├─ Additive:  h = h + α·v
  │     │     ├─ Clamping:  h = h - (h·v)v + α·v
  │     │     ├─ Decaying:  h = h + (α · decay^t)·v
  │     │     └─ PID:       h = h + (Kp·e + Ki·∫e + Kd·de/dt)·v
  │     │          where e = target_activation - current_projection
  │     │
  │     └─ Log: α used, activation magnitude, cosine shift
  │
  └─ Return modified hidden_states
```

---

## Consciousness Research Framework

### Scientific Foundation

The mind-matter interaction literature provides the empirical basis:
- **PEAR Lab** (Princeton, 1979-2007): Millions of trials, shifts of ~10⁻⁴ against odds of billions-to-one.
- **Global Consciousness Project** (1998-2015): Z=7.31 across 500 pre-registered events using worldwide RNG network.
- **Holmberg (2025)**: Bayesian entropy framework + 2-year TrueRNG experiment, t=4.347, p<0.001. Modeled consciousness as an "informational variable" biasing probabilistic outcomes.
- **Key principle**: "The source of randomness matters." Quantum-origin randomness shows effects that pseudo-random does not.

### Why LLM Hidden States Are a Better Detector

Traditional consciousness-RNG experiments measure simple bit distributions (binary, 1D).
LLM hidden states offer a far richer signal space:
- **4096 dimensions** of continuous-valued activations (vs. 1 bit)
- **Semantic structure**: perturbations map to interpretable concept directions
- **Amplification**: small activation changes cascade through 32 layers → large output effects
- **Natural language output**: human-readable results, not statistical tables
- **Double-blind capable**: QRNG vs os.urandom, identical code path, different entropy source

### Dual-Surface Architecture

```
                      ┌─────────────────────────────────────────────────┐
                      │               Forward Pass                      │
                      │                                                 │
Input ──────────────> │  L0 ── ... ── L15 ── ... ── L31 ──────> Logits │
                      │                 │                          │    │
                      │        ┌────────┴────────┐       ┌────────┴────────┐
                      │        │  qr-steering     │       │  qr-sampler      │
                      │        │  STATE CONTROL   │       │  OUTPUT CONTROL  │
                      │        │                  │       │                  │
                      │        │  QRNG → α mod    │       │  QRNG → u value  │
                      │        │  PID: h + Σ·v    │       │  CDF selection   │
                      │        └──────────────────┘       └──────────────────┘
                      │                                                 │
                      │    Surface 1: Hidden states    Surface 2: Token │
                      └─────────────────────────────────────────────────┘
```

Two independent QRNG-influence surfaces. Can be combined or used separately.
Combined mode maximizes the "surface area" available for consciousness experiments.

### Experimental Protocol

```
Condition A (Quantum):   Real QRNG → modulates steering + selection
Condition B (Sham):      os.urandom() → same code path (double-blind control)
Condition C (Fixed):     No modulation (baseline)

Protocol:
1. Select concept vector (e.g., "honesty", "creativity")
2. Generate N responses per condition with identical prompts
3. Log per-token: α used, activation shift, token selected, entropy source
4. Statistical comparison: Mann-Whitney, KS, Welch's t, Cohen's d
5. Analysis: do quantum-modulated runs show detectable structure?
```

### Analysis Pipeline
- Reuse qr-sampler's analysis module (JSONL persistence, statistics, two-sample comparison)
- Add activation-space metrics: cosine drift, magnitude tracking, AU-level statistics
- LLM-judge evaluation for steered output quality
- Bayesian entropy analysis (Holmberg methodology) on activation perturbation patterns

---

## Implementation Phases

### Phase 1: Foundation (MVP — "Hello World" of Steering)

**Goal**: Steer a model with a pre-saved vector. No SAE, no QRNG, no Neuronpedia.
Ships with a Colab quickstart notebook.

**Deliverables**:
- [ ] Repo scaffolding: pyproject.toml, CI (ruff, mypy, pytest), LICENSE, README
- [ ] `SteeringConfig` (pydantic-settings, `QRS_` env prefix)
- [ ] `exceptions.py` — `SteeringError` hierarchy
- [ ] `hooks/models.py` — nnterp-style model family auto-detection (single mapping dict)
- [ ] `HookManager` — register/remove forward hooks, context manager lifecycle
- [ ] `SteeringContext` — per-generation mutable state (token count, logged activations)
- [ ] `SteeringMethod` ABC + `AdditiveMethod` (`x = x + α·v`)
- [ ] `VectorSource` ABC + `FileSource` (load .pt / .npy / .safetensors vectors)
- [ ] `SteeringVector` frozen dataclass
- [ ] `SteeringEngine` — load model, steer, generate, clear
  - Both persistent mode (`engine.steer(...)` + `engine.clear()`) and context manager (`with engine.steer(...):`)
  - Negative layer indexing (`layer=-5` = 5th from last)
  - Scalar strength/multiplier parameter
- [ ] Tests: hook lifecycle, additive math, engine round-trip (using mock model)
- [ ] `notebooks/01_quickstart.ipynb` — Colab-ready, "Open in Colab" badge

**Deps**: torch, transformers, pydantic-settings
**Test on**: Gemma 2 2B (5 GB VRAM) and Llama 3.1 8B (16 GB VRAM)

**Exit criteria**: `engine.steer(vector_file="...", strength=8.0)` → concept-steered text.
Tests pass, ruff clean, mypy strict clean. Notebook runs on Colab T4.

---

### Phase 2: Vector Discovery (Find Your Own Concepts)

**Goal**: Discover and build steering vectors without pre-extracted files.
SAELens integration for SAE loading. Neuronpedia for concept search.

**Deliverables**:
- [ ] `NeuronpediaSource` — API client:
  - Semantic search by concept description
  - Inference search (run text, find top features)
  - Download SAE decoder column as `SteeringVector`
  - Feature metadata (description, density, max activation)
- [ ] `SAESource` — SAELens integration:
  - `SAE.from_pretrained(release, sae_id)` → `sae.W_dec[feature_idx]` as SteeringVector
  - Supports all SAELens v6 releases: Gemma 2/3, Llama 3/3.1/3.3, GPT-2, Qwen, Mistral, Pythia
  - Neuronpedia ID auto-resolved from SAE metadata (`sae.cfg.metadata.neuronpedia_id`)
  - Graceful fallback to raw `torch.load` if SAELens not installed
- [ ] `ContrastiveSource` — CAA vector computation:
  - Accept positive/negative prompt pair lists
  - Collect activations at target layer, compute mean difference, normalize
- [ ] `ClampingMethod` — `x = x - (x·v)v + α·v`
- [ ] Tests: Neuronpedia (mocked API), SAE extraction, contrastive math, clamping
- [ ] `notebooks/02_neuronpedia_discovery.ipynb`
- [ ] `notebooks/03_contrastive_vectors.ipynb`
- [ ] `notebooks/04_sae_steering.ipynb`

**New deps**: neuronpedia (optional), sae-lens (optional)

**Exit criteria**: `engine.steer(concept="honesty", source="neuronpedia")` works.
CAA from prompt pairs works. Clamping method verified against Eiffel Tower Llama results.

---

### Phase 3: PID Steering + QRNG Integration (The Differentiators)

**Goal**: Implement PID steering (principled control theory) and quantum modulation
(our unique contribution). These two features together are what no other library offers.

**Deliverables**:
- [ ] `PIDMethod`:
  - Proportional: `Kp · e · v` (standard additive = P-only with Kp=1)
  - Integral: accumulate projection error across layers → persistent correction
  - Derivative: counteract rapid activation changes → prevent overshoot
  - Configurable gains: Kp, Ki, Kd
  - `reset()` clears state between generations
- [ ] `DecayingMethod` — token-wise decay: `α_t = α_0 · decay^t`
- [ ] `QuantumModulator` ABC + `ModulatorRegistry`
- [ ] `AlphaModulator` (Q1) — `α_eff = α_base + β · qrng_uniform()`
  - Uses qr-sampler's EntropySource
  - Configurable β, clamping bounds
- [ ] `DirectionalModulator` (Q4) — `v_eff = normalize(v + σ · qrng_noise)`
- [ ] `NoiseInjector` (Q7) — vectorless residual stream quantum noise
- [ ] Integration with qr-sampler entropy sources (system, mock, quantum, openentropy)
- [ ] Steering diagnostic logging (per-token: α, vector norm, activation shift, PID terms)
- [ ] Tests: PID math (known-value), modulator bounds, noise injection, entropy integration
- [ ] `notebooks/05_pid_steering.ipynb`
- [ ] `notebooks/06_qrng_modulation.ipynb`
- [ ] `notebooks/07_vectorless_noise.ipynb`

**New deps**: qr-sampler (optional)

**Exit criteria**: PID steering demonstrably more stable than additive on α sweep.
QRNG modulation produces per-token varied steering. Vectorless noise works without
concept vector. All modulators deterministically testable with MockUniformSource.

---

### Phase 4: nnsight Interop + Multi-Concept

**Goal**: First-class nnsight integration for MI researchers. Multi-vector steering.

**Deliverables**:
- [ ] `nnsight/intervention.py` — `SteeringIntervention` class:
  - Works inside `with model.trace()` context
  - Supports all SteeringMethods (additive, clamping, PID, decaying)
  - Supports QuantumModulators
  - Compatible with nnsight's deferred execution model
- [ ] Multi-vector steering: multiple SteeringVectors on same/different layers
- [ ] Composition strategies: sum, sequential application
- [ ] `PIDGainModulator` (Q2) — QRNG-varied P/I/D gains
- [ ] `FeatureWalkModulator` (Q5) — QRNG selects among concept features per token
- [ ] `LayerTargetModulator` (Q6) — QRNG varies active layers per token
- [ ] `DecayModulator` (Q8) — QRNG-varied decay rate
- [ ] `notebooks/08_nnsight_interop.ipynb`
- [ ] Tests: nnsight interop (require nnsight as test dep), multi-vector

**New deps**: nnsight (optional)

**Exit criteria**: `SteeringIntervention` works inside nnsight trace context.
Multi-concept steering works. All 8 quantum modulators functional.

---

### Phase 5: Evaluation + Experiment Runner

**Goal**: Comprehensive evaluation pipeline and experiment infrastructure for
consciousness research.

**Deliverables**:
- [ ] `AlphaSweep` — grid search over α with automated metrics
- [ ] Steering quality metrics: n-gram repetition, perplexity delta, KL divergence
- [ ] LLM-judge evaluation (optional API):
  - Concept inclusion, instruction following, fluency (0–2 scale each)
  - Harmonic mean aggregate (Eiffel Tower paper methodology)
- [ ] Activation logging: per-token stats → JSONL (compatible with qr-sampler analysis)
- [ ] Quantum vs Classical comparison framework:
  - Paired experiment runner (same prompts, QRNG vs urandom vs fixed)
  - Statistical battery (Mann-Whitney, KS, Welch's t, Cohen's d)
  - Bayesian entropy analysis (Holmberg methodology)
- [ ] `ExperimentRunner` — load YAML, batch experiments, save results
- [ ] `notebooks/09_dual_surface.ipynb`
- [ ] Consciousness experiment protocol documentation
- [ ] Multi-model validation: Gemma 2B, Gemma 9B, Llama 8B, GPT-2 Small

**Exit criteria**: `python -m qr_steering.experiment experiments/dual_surface.yaml`
produces statistical comparison. Validated on 3+ model architectures.

---

### Phase 6: Fine-Grained Steering (AUSteer + SAE-TS)

**Goal**: Implement state-of-the-art fine-grained steering methodologies.

**Deliverables**:
- [ ] AU-level analysis:
  - Compute activation momenta on contrastive samples
  - Identify discriminative AUs (dimensions that matter)
  - Per-AU adaptive steering strengths
- [ ] `AUQuantumModulator` (Q3) — QRNG selects which discriminative AUs get steered
- [ ] SAE-Targeted Steering (SAE-TS) — minimize side effects
- [ ] Conceptors — Boolean composition (AND/OR/NOT) for multi-concept
- [ ] `Qwen` and `Phi` model family entries

**Exit criteria**: AU-level steering outperforms block-level on evaluation suite.

---

### Phase 7: vLLM Integration + Production Path

**Goal**: Bring steering into vLLM for production-speed deployment. Enable unified
dual-surface (qr-steering + qr-sampler) in a single vLLM process.

**Key insight (v4)**: nnsight 0.6 has **first-class vLLM support** with async streaming.
This may eliminate the need to fork vLLM's engine (EasySteer's approach). The nnsight
`VLLM` class wraps vLLM and provides the same `model.trace()` intervention API:

```python
from nnsight.modeling.vllm import VLLM
model = VLLM("meta-llama/Llama-3.1-8B-Instruct", mode="async")
with model.trace(prompt, max_tokens=256) as tracer:
    model.model.layers[16].output[0][:] += steering_vector
async for output in tracer.backend():
    print(output.outputs[0].text, end="", flush=True)
```

**Deliverables**:
- [ ] Evaluate nnsight's vLLM integration for steering (primary path)
- [ ] If nnsight vLLM is sufficient: adapt SteeringIntervention for vLLM mode
- [ ] If not: study EasySteer's fork architecture as fallback
- [ ] Continuous batching support with per-request steering vectors
- [ ] Unified deployment: qr-sampler (logits) + qr-steering (activations) in one vLLM config
- [ ] Performance benchmarking: HF vs vLLM steering latency
- [ ] Documentation: deployment guide

**Exit criteria**: Steering works on vLLM with comparable quality to HF path.
Dual-surface experiment runs at production speed.

---

### Phase 8: Documentation, Polish, and Community (Ongoing)

- [ ] README with architecture diagrams, quick start, API reference
- [ ] CLAUDE.md for agent-assisted development
- [ ] All notebooks with "Open in Colab" badges
- [ ] Published to PyPI
- [ ] Blog post: "Quantum-Modulated LLM Steering for Consciousness Research"
- [ ] Experiment results write-up

---

## Hardware Requirements

| Model | VRAM (fp16) | With SAE | Notes |
|-------|------------|---------|-------|
| GPT-2 Small (124M) | ~0.5 GB | ~1 GB | Best for unit testing. Full SAE coverage. |
| Gemma 2 2B | ~5 GB | ~7 GB | Best for development. GemmaScope SAEs. Consumer GPU. |
| Llama 3.1 8B | ~16 GB | ~20 GB | Primary research target. RTX 4090 / A100. |
| Gemma 2 9B | ~18 GB | ~22 GB | Second research target. GemmaScope SAEs. |
| Llama 3.3 70B | ~140 GB | ~160 GB | Multi-GPU or NDIF remote execution. |
| Llama 3.1 405B | ~810 GB | N/A | NDIF only. Pre-loaded on NDIF cluster. |

**Strategy**: Build on GPT-2/Gemma 2B (fast iteration). Validate on Llama 8B. Publish results on 8B.
**NDIF remote execution** via nnsight enables experiments on 70B+ and 405B models without
local GPU — the intervention graph is serialized and sent to NDIF's pre-loaded model shards.
For 3B+ parameter models, remote execution is faster than local HPC including network round-trip.

---

## Competitive Positioning

| | Goodfire | EasySteer | AISteer360 | nnsight | **qr-steering** |
|---|---------|-----------|-----------|---------|----------------|
| Open source | Deprecated API | Yes | Yes | Yes | **Yes** |
| Runtime | Proprietary | vLLM (forked) | HuggingFace | HuggingFace | **HF + vLLM (P7)** |
| nnsight interop | No | No | No | N/A | **Yes (P4)** |
| PID steering | No | No | No | No | **Yes (P3)** |
| AUSteer | No | No | No | No | **Yes (P6)** |
| QRNG integration | No | No | No | No | **Yes (P3)** |
| Consciousness research | No | No | No | No | **Yes** |
| Neuronpedia integration | No | No | No | No | **Yes (P2)** |
| Dual-surface (state+output) | No | No | Partial | No | **Yes (P5-7)** |
| SAELens integration | No | Pre-computed | No | Via nnterp | **Yes (P2)** |
| Notebook tutorials | No | No | No | Basic | **Yes (9 notebooks)** |
| Remote execution (NDIF) | No | No | No | Yes | **Via nnsight (P4)** |

**Our unique differentiators**:
1. Only library with QRNG-modulated steering
2. Only library with PID steering (control-theoretic foundation)
3. Only project connecting activation steering to consciousness research
4. Dual-surface architecture (hidden states + logits) for maximum experimental surface area
5. Open-source with Neuronpedia/SAE integration (Goodfire went closed)
6. nnsight interop — works inside existing MI researcher workflows
7. Notebook-first documentation (9 Colab-ready notebooks)

---

## Key References

**Steering Methods**:
1. Eiffel Tower Llama (Louapre, HuggingFace 2025) — SAE steering, clamping, α optimization
2. Golden Gate Claude (Templeton et al., Anthropic 2024) — Original SAE clamping
3. ActAdd / CAA (Turner et al. 2023) — Contrastive activation addition
4. SAE-TS (Chalnev & Conmy 2024) — Minimize steering side effects
5. FGAA (Soo et al., ICLR 2025) — Feature Guided Activation Additions
6. SADI (Wang et al., ICLR 2025) — Semantics-Adaptive Dynamic Intervention
7. AUSteer (Feng et al., ICLR 2026) — Fine-grained AU-level steering
8. PID Steering (Nguyen et al., ICLR 2026) — Control-theoretic foundation
9. Token-wise Decay (Xie, ETH Zurich 2025) — Decaying SAE steering
10. Conceptors (Abreu et al., NeurIPS 2025) — Boolean compositional steering
11. K-Steering (Thoughtworks 2026) — Non-linear steering
12. RISER (Jan 2026) — Dynamic vector composition with Router MLP, K=6 reasoning primitives
13. PAS (Cui et al., Yale 2025) — Painless Activation Steering, automated vector training from labeled data
14. Dialz (Cardiff NLP, ACL 2025) — PCA/mean-diff steering with visualization

**Benchmarks & Analysis**:
15. AxBench (Wu et al., Stanford, ICML 2025) — Steering benchmark
16. Steering Awareness (Fonseca Rivera & Africa 2026) — Models detect steering
17. SAE decomposition pitfall (Mayne et al. 2024) — SAEs can't reliably decompose steering vectors

**Frameworks & Tools**:
18. EasySteer (Xu et al., ZJU 2025) — vLLM-integrated steering, 10-22x speedup
19. pyvene (Wu et al., Stanford, NAACL 2024) — Intervention library
20. SAELens (Bloom et al.) — SAE training/loading, standard weight format (v6.37.6)
21. AISteer360 (IBM 2025) — 4-category steering taxonomy
22. Conditional Activation Steering (IBM, ICLR 2025) — Context-dependent steering
23. Neuronpedia — Interpretability platform, 30+ models, 50M+ features
24. nnsight (Fiotto-Kaufman et al., NDIF) — Next-gen MI library, native HF, vLLM, remote (v0.6.2)
25. nnterp — Standardized nnsight wrapper, 50+ model variants, 16 architecture families
26. baukit (Bau, MIT) — Predecessor to nnsight, hook context managers
27. steering-vectors — Context manager pattern, contrastive pair training
28. repeng — Fast vector training (<60s), clean control API

**Consciousness & QRNG**:
29. Global Consciousness Project (Nelson 2024) — 17-year RNG experiment, Z=7.31
30. Holmberg (2025) — Bayesian entropy framework, TrueRNG t=4.347 p<0.001
31. IONS / Radin (2025) — Observer influence on quantum interference
32. Quantum Randomness in Psi Experiments (Hong 2025) — "Source of randomness matters"
