"""Shared test mock objects and helpers for entropick tests."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from qr_sampler.processor import QRSamplerLogitsProcessor

# ---------------------------------------------------------------------------
# Mock objects simulating vLLM's batch management types
# ---------------------------------------------------------------------------


@dataclass
class MockVllmConfig:
    """Simulates vLLM's VllmConfig with vocab_size access."""

    vocab_size: int = 10


@dataclass
class MockModelConfig:
    """Simulates vLLM's model config nested structure."""

    hf_text_config: Any = None


@dataclass
class MockHfTextConfig:
    """Simulates the HuggingFace text config with vocab_size."""

    vocab_size: int = 10


@dataclass
class MockSamplingParams:
    """Simulates vLLM's SamplingParams."""

    extra_args: dict[str, Any] | None = None


@dataclass
class MockAddedRequest:
    """Simulates a BatchUpdate added request."""

    req_index: int
    sampling_params: MockSamplingParams | None = None


@dataclass
class MockMovedRequest:
    """Simulates a BatchUpdate moved request."""

    src_index: int
    dst_index: int


@dataclass
class MockBatchUpdate:
    """Simulates vLLM's BatchUpdate dataclass."""

    removed: list[int] | None = None
    moved: list[MockMovedRequest] | None = None
    added: list[MockAddedRequest] | None = None

    def __post_init__(self) -> None:
        if self.removed is None:
            self.removed = []
        if self.moved is None:
            self.moved = []
        if self.added is None:
            self.added = []


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

VOCAB_SIZE = 10
SAMPLE_LOGITS = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]


def make_processor(
    vocab_size: int = VOCAB_SIZE,
    entropy_source_type: str = "mock_uniform",
    fallback_mode: str = "error",
    **config_overrides: Any,
) -> QRSamplerLogitsProcessor:
    """Create a processor with mock entropy and optional config overrides.

    Sets environment variables to configure, then instantiates. Cleans up
    environment after construction.
    """
    env_vars = {
        "QR_ENTROPY_SOURCE_TYPE": entropy_source_type,
        "QR_FALLBACK_MODE": fallback_mode,
        "QR_LOG_LEVEL": "none",
    }
    for key, value in config_overrides.items():
        env_vars[f"QR_{key.upper()}"] = str(value)

    old_env: dict[str, str | None] = {}
    for key, value in env_vars.items():
        old_env[key] = os.environ.get(key)
        os.environ[key] = value

    try:
        vllm_config = MockVllmConfig(vocab_size=vocab_size)
        proc = QRSamplerLogitsProcessor(
            vllm_config=vllm_config,
            device=None,
            is_pin_memory=False,
        )
    finally:
        for key, original in old_env.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original

    return proc


def assert_onehot(row: np.ndarray) -> None:
    """Assert a logit row is one-hot: exactly one 0.0, rest -inf."""
    n_zero = int(np.sum(row == 0.0))
    n_neginf = int(np.sum(np.isneginf(row)))
    assert n_zero == 1, f"Expected 1 zero, got {n_zero}"
    assert n_neginf == len(row) - 1, f"Expected {len(row) - 1} -inf, got {n_neginf}"


def register_request(
    proc: QRSamplerLogitsProcessor,
    req_index: int = 0,
    extra_args: dict[str, Any] | None = None,
) -> None:
    """Register a single request with optional extra_args."""
    params = MockSamplingParams(extra_args=extra_args)
    batch = MockBatchUpdate(added=[MockAddedRequest(req_index=req_index, sampling_params=params)])
    proc.update_state(batch)
