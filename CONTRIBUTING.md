# Contributing to entropick

## Development setup

```bash
git clone https://github.com/ereid7/entropick.git
cd entropick
pip install -e ".[dev]"
pre-commit install
```

This installs the package in editable mode with all dev dependencies (pytest, ruff, mypy,
scipy, pre-commit, bandit, gRPC) and sets up pre-commit hooks.

## Running tests

```bash
# Full suite
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src/qr_sampler --cov-report=term-missing

# Specific modules
pytest tests/test_config.py -v
pytest tests/test_amplification/ -v
pytest tests/test_temperature/ -v
pytest tests/test_selection/ -v
pytest tests/test_logging/ -v
pytest tests/test_entropy/ -v
pytest tests/test_injection/ -v
pytest tests/test_pipeline/ -v
pytest tests/test_analysis/ -v
pytest tests/test_adapters/ -v
pytest tests/test_processor.py -v
pytest tests/test_contracts.py -v
pytest tests/test_statistical_properties.py -v
```

No real QRNG server or GPU is needed. Tests use `MockUniformSource` and numpy arrays.
Statistical tests in `test_statistical_properties.py` require `scipy` (included in dev deps).

## Linting and type checking

All three must pass before merging:

```bash
ruff check src/ tests/
ruff format --check src/ tests/
mypy --strict src/
```

Fix formatting automatically with `ruff format src/ tests/`.
Fix auto-fixable lint issues with `ruff check --fix src/ tests/`.

## Code conventions

- **Python 3.10+** -- use `X | Y` union syntax, not `Union[X, Y]`
- **Type hints** on all function signatures and return types
- **Google-style docstrings** on every public class and method
- **100-character line length** (configured in `pyproject.toml`)
- **Imports** -- stdlib first, third-party second, local third; no wildcard imports
- **Custom exceptions** rooted in `QRSamplerError` (see `exceptions.py`); never catch bare `Exception`
- **No `print()`** -- use `logging.getLogger("qr_sampler")`
- **No global mutable state** outside processor instances
- **Frozen dataclasses** with `__slots__` for all result types
- **`QR_` prefix** for environment variables, `qr_` prefix for `extra_args` keys

## Adding new components

See `CLAUDE.md` for the full file map, architecture invariants, and data flow diagrams.

### New pipeline stage

1. Create a class in `src/qr_sampler/stages/` with `name: str` and `__call__(self, ctx: SamplingContext) -> None`
2. Register with `@StageRegistry.register("my_stage")`
3. Add config fields to `QRSamplerConfig` and `_PER_REQUEST_FIELDS` if needed
4. Add to `build_default_pipeline()` in `src/qr_sampler/stages/__init__.py` at the appropriate position
5. Add entry point in `pyproject.toml` under `[project.entry-points."qr_sampler.pipeline_stages"]`
6. Add tests in `tests/test_pipeline/`

### New entropy source

1. Create a class in `src/qr_sampler/entropy/` subclassing `EntropySource`
2. Implement `name`, `is_available`, `get_random_bytes(n)`, `close()`
3. Raise `EntropyUnavailableError` from `get_random_bytes()` on failure
4. Register with `@register_entropy_source("my_name")`
5. Add entry point in `pyproject.toml` under `[project.entry-points."qr_sampler.entropy_sources"]`
6. Add tests in `tests/test_entropy/`

### New signal amplifier

1. Create a class in `src/qr_sampler/amplification/` subclassing `SignalAmplifier`
2. Implement `amplify(raw_bytes) -> AmplificationResult`
3. Constructor takes `config: QRSamplerConfig` as first arg
4. Register with `@AmplifierRegistry.register("my_name")`
5. Add tests in `tests/test_amplification/`

### New temperature strategy

1. Create a class in `src/qr_sampler/temperature/` subclassing `TemperatureStrategy`
2. Implement `compute_temperature(logits, config) -> TemperatureResult`
3. Always compute and return `shannon_entropy` (logging depends on it)
4. If the constructor needs `vocab_size`, accept it as first positional arg
5. Register with `@TemperatureStrategyRegistry.register("my_name")`
6. Add tests in `tests/test_temperature/`

### New injection method

Injection methods are stateless utility classes (not registered via the registry pattern).
See the "New injection method" section in `CLAUDE.md` for the full walkthrough.

### New config field

1. Add the field to `QRSamplerConfig` in `config.py` with `Field(default=..., description=...)`
2. If per-request overridable, add to `_PER_REQUEST_FIELDS` frozenset
3. Env var `QR_{FIELD_NAME_UPPER}` and extra_args key `qr_{field_name}` are auto-supported
4. Add tests in `tests/test_config.py`

## Architecture invariants

Do not break these (see `CLAUDE.md` for the full list):

- **No hardcoded values** -- every constant traces to a named field in `QRSamplerConfig`
- **Registry pattern** for amplifiers, temperature strategies, and entropy sources
- **Frozen dataclasses** for all result types
- **Per-request config** never mutates the default config instance
- **SEM is derived** (`population_std / sqrt(N)`), never stored as a config field
- **One-hot forcing** -- after selection, all logits are `-inf` except the chosen token (`0.0`)
- **Just-in-time entropy** -- no pre-buffering or caching of random bytes

## Pull request guidelines

1. **Tests required** -- every PR must include tests for new functionality
2. **Lint must pass** -- `ruff check` and `ruff format --check` with zero warnings
3. **Types must pass** -- `mypy --strict src/` with zero errors
4. **Keep PRs focused** -- one feature or fix per PR
5. **Describe the change** -- explain *why*, not just *what*
6. **Update CLAUDE.md** if you change the file map, architecture, or public API
7. Run the full quality gate before submitting:
   ```bash
   ruff check src/ tests/ && ruff format --check src/ tests/ && mypy --strict src/ && pytest tests/ -v
   ```

## Pre-commit hooks

```bash
pre-commit run --all-files
```

Hooks run ruff (lint + format), mypy, and bandit automatically on staged files.

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
