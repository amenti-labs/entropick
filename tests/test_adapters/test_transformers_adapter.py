"""Tests for the Hugging Face Transformers adapter.

Tests exercise the adapter using numpy arrays and mock torch tensors,
so no actual torch installation is required for the test suite.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from qr_sampler.config import QRSamplerConfig

# ---------------------------------------------------------------------------
# Helpers: lightweight mock tensor that behaves like torch.Tensor
# ---------------------------------------------------------------------------


class MockTensor:
    """Minimal mock of torch.Tensor for testing the HF adapter.

    Wraps a numpy array and provides the torch.Tensor interface
    methods used by the adapter: ``dim()``, ``shape``, ``detach()``,
    ``numpy()``, ``fill_()``, ``__getitem__``, ``__setitem__``,
    ``is_cuda``.
    """

    def __init__(
        self,
        data: np.ndarray[Any, np.dtype[np.floating[Any]]],
        _view: bool = False,
    ) -> None:
        self._data = data if _view else data.astype(np.float32)
        self.shape = self._data.shape
        self.is_cuda = False

    def dim(self) -> int:
        """Return number of dimensions."""
        return len(self._data.shape)

    def detach(self) -> MockTensor:
        """Return self (no autograd in mock)."""
        return self

    def numpy(self) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
        """Return underlying numpy array."""
        return self._data

    def fill_(self, value: float) -> MockTensor:
        """Fill array with value in-place."""
        self._data[:] = value
        return self

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int) and self._data.ndim >= 2:
            # Return a view that shares memory with the parent.
            return MockTensor(self._data[key], _view=True)
        return self._data[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        self._data[key] = value


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_config(**overrides: Any) -> QRSamplerConfig:
    """Create a test config with mock entropy and logging disabled."""
    defaults: dict[str, Any] = {
        "entropy_source_type": "mock_uniform",
        "fallback_mode": "error",
        "log_level": "none",
    }
    defaults.update(overrides)
    return QRSamplerConfig(_env_file=None, **defaults)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBasicSelection:
    """Test that the adapter produces valid one-hot output."""

    def test_basic_selection_with_mock_tensor(self) -> None:
        """Adapter processes mock tensor and produces one-hot output."""
        from qr_sampler.adapters.transformers import QRSamplerLogitsProcessorHF

        config = _make_config()
        processor = QRSamplerLogitsProcessorHF(config=config, vocab_size=10)

        scores = MockTensor(np.array([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]]))
        input_ids = MockTensor(np.array([[1, 2, 3]]))

        # Patch torch import inside the adapter.
        import sys

        mock_torch = MagicMock()
        mock_torch.__name__ = "torch"
        original_torch = sys.modules.get("torch")
        sys.modules["torch"] = mock_torch
        try:
            result = processor(input_ids, scores)
        finally:
            if original_torch is not None:
                sys.modules["torch"] = original_torch
            else:
                sys.modules.pop("torch", None)

        assert result is scores
        row = result._data[0]
        assert np.sum(row == 0.0) == 1, "Expected exactly one 0.0 in one-hot output"
        assert np.sum(np.isneginf(row)) == 9, "Expected 9 -inf values in one-hot output"

    def test_dominant_token_selected(self) -> None:
        """When one logit overwhelmingly dominates, it is always selected."""
        from qr_sampler.adapters.transformers import QRSamplerLogitsProcessorHF

        config = _make_config()
        processor = QRSamplerLogitsProcessorHF(config=config, vocab_size=10)

        logits = np.array(
            [
                [
                    -100.0,
                    -100.0,
                    -100.0,
                    100.0,
                    -100.0,
                    -100.0,
                    -100.0,
                    -100.0,
                    -100.0,
                    -100.0,
                ]
            ]
        )
        scores = MockTensor(logits)
        input_ids = MockTensor(np.array([[1]]))

        import sys

        mock_torch = MagicMock()
        original_torch = sys.modules.get("torch")
        sys.modules["torch"] = mock_torch
        try:
            processor(input_ids, scores)
        finally:
            if original_torch is not None:
                sys.modules["torch"] = original_torch
            else:
                sys.modules.pop("torch", None)

        assert scores._data[0, 3] == 0.0, "Token 3 should be selected"


class TestConfigOverride:
    """Test that custom config and overrides are applied."""

    def test_config_passed_through(self) -> None:
        """Config object is used by the adapter."""
        from qr_sampler.adapters.transformers import QRSamplerLogitsProcessorHF

        config = _make_config(top_k=1)
        processor = QRSamplerLogitsProcessorHF(config=config, vocab_size=10)

        scores = MockTensor(np.array([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]]))
        input_ids = MockTensor(np.array([[1]]))

        import sys

        mock_torch = MagicMock()
        original_torch = sys.modules.get("torch")
        sys.modules["torch"] = mock_torch
        try:
            processor(input_ids, scores)
        finally:
            if original_torch is not None:
                sys.modules["torch"] = original_torch
            else:
                sys.modules.pop("torch", None)

        # With top_k=1, only the highest logit (index 0) should be selected.
        assert scores._data[0, 0] == 0.0, "Token 0 should be selected with top_k=1"

    def test_kwargs_override(self) -> None:
        """Keyword overrides are applied to config."""
        from qr_sampler.adapters.transformers import QRSamplerLogitsProcessorHF

        config = _make_config()
        processor = QRSamplerLogitsProcessorHF(config=config, vocab_size=10, top_k=1)

        scores = MockTensor(np.array([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]]))
        input_ids = MockTensor(np.array([[1]]))

        import sys

        mock_torch = MagicMock()
        original_torch = sys.modules.get("torch")
        sys.modules["torch"] = mock_torch
        try:
            processor(input_ids, scores)
        finally:
            if original_torch is not None:
                sys.modules["torch"] = original_torch
            else:
                sys.modules.pop("torch", None)

        assert scores._data[0, 0] == 0.0, "Token 0 should be selected with top_k=1 override"


class TestPipelineExecution:
    """Test that all pipeline stages execute."""

    def test_diagnostic_records_produced(self) -> None:
        """With diagnostic_mode, records are stored."""
        from qr_sampler.adapters.transformers import QRSamplerLogitsProcessorHF

        config = _make_config(diagnostic_mode=True)
        processor = QRSamplerLogitsProcessorHF(config=config, vocab_size=10)

        scores = MockTensor(np.array([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]]))
        input_ids = MockTensor(np.array([[1]]))

        import sys

        mock_torch = MagicMock()
        original_torch = sys.modules.get("torch")
        sys.modules["torch"] = mock_torch
        try:
            processor(input_ids, scores)
        finally:
            if original_torch is not None:
                sys.modules["torch"] = original_torch
            else:
                sys.modules.pop("torch", None)

        records = processor.sampling_logger.get_diagnostic_data()
        assert len(records) == 1

        record = records[0]
        assert record.token_id >= 0
        assert record.token_id < 10
        assert 0.0 < record.u_value < 1.0
        assert record.token_rank >= 0
        assert record.token_prob > 0.0
        assert record.num_candidates > 0
        assert record.total_sampling_ms > 0.0
        assert record.temperature_used > 0.0

    def test_multiple_calls_accumulate_records(self) -> None:
        """Multiple calls produce multiple diagnostic records."""
        from qr_sampler.adapters.transformers import QRSamplerLogitsProcessorHF

        config = _make_config(diagnostic_mode=True)
        processor = QRSamplerLogitsProcessorHF(config=config, vocab_size=10)

        import sys

        mock_torch = MagicMock()
        original_torch = sys.modules.get("torch")
        sys.modules["torch"] = mock_torch
        try:
            for _ in range(3):
                scores = MockTensor(
                    np.array([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]])
                )
                input_ids = MockTensor(np.array([[1]]))
                processor(input_ids, scores)
        finally:
            if original_torch is not None:
                sys.modules["torch"] = original_torch
            else:
                sys.modules.pop("torch", None)

        records = processor.sampling_logger.get_diagnostic_data()
        assert len(records) == 3


class TestLazyInit:
    """Test lazy initialization behavior."""

    def test_vocab_size_inferred(self) -> None:
        """When vocab_size=0, it is inferred from the scores tensor."""
        from qr_sampler.adapters.transformers import QRSamplerLogitsProcessorHF

        config = _make_config()
        processor = QRSamplerLogitsProcessorHF(config=config)  # vocab_size=0

        scores = MockTensor(np.array([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]]))
        input_ids = MockTensor(np.array([[1]]))

        import sys

        mock_torch = MagicMock()
        original_torch = sys.modules.get("torch")
        sys.modules["torch"] = mock_torch
        try:
            processor(input_ids, scores)
        finally:
            if original_torch is not None:
                sys.modules["torch"] = original_torch
            else:
                sys.modules.pop("torch", None)

        # Should have inferred vocab_size=10 from scores.
        assert processor._components is not None
        assert processor._components.vocab_size == 10

    def test_config_property_before_init(self) -> None:
        """Config property works before first call."""
        from qr_sampler.adapters.transformers import QRSamplerLogitsProcessorHF

        config = _make_config()
        processor = QRSamplerLogitsProcessorHF(config=config)
        assert processor.config is config

    def test_config_property_after_init(self) -> None:
        """Config property returns resolved config after first call."""
        from qr_sampler.adapters.transformers import QRSamplerLogitsProcessorHF

        config = _make_config()
        processor = QRSamplerLogitsProcessorHF(config=config, vocab_size=10)

        scores = MockTensor(np.array([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]]))
        input_ids = MockTensor(np.array([[1]]))

        import sys

        mock_torch = MagicMock()
        original_torch = sys.modules.get("torch")
        sys.modules["torch"] = mock_torch
        try:
            processor(input_ids, scores)
        finally:
            if original_torch is not None:
                sys.modules["torch"] = original_torch
            else:
                sys.modules.pop("torch", None)

        assert processor.config is not None
        assert isinstance(processor.config, QRSamplerConfig)


class TestClose:
    """Test resource cleanup."""

    def test_close_before_init(self) -> None:
        """Close before first call is a no-op."""
        from qr_sampler.adapters.transformers import QRSamplerLogitsProcessorHF

        config = _make_config()
        processor = QRSamplerLogitsProcessorHF(config=config)
        processor.close()  # Should not raise.

    def test_close_after_init(self) -> None:
        """Close after use releases resources."""
        from qr_sampler.adapters.transformers import QRSamplerLogitsProcessorHF

        config = _make_config()
        processor = QRSamplerLogitsProcessorHF(config=config, vocab_size=10)

        scores = MockTensor(np.array([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]]))
        input_ids = MockTensor(np.array([[1]]))

        import sys

        mock_torch = MagicMock()
        original_torch = sys.modules.get("torch")
        sys.modules["torch"] = mock_torch
        try:
            processor(input_ids, scores)
        finally:
            if original_torch is not None:
                sys.modules["torch"] = original_torch
            else:
                sys.modules.pop("torch", None)

        processor.close()  # Should not raise.
        processor.close()  # Idempotent.


class TestLlamaCppAdapter:
    """Test the llama-cpp-python adapter."""

    def test_basic_list_processing(self) -> None:
        """Callback processes a flat list of scores and returns one-hot list."""
        from qr_sampler.adapters.llamacpp import QRSamplerCallback

        config = _make_config()
        callback = QRSamplerCallback(config=config, vocab_size=10)

        input_ids = [1, 2, 3]
        scores = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]

        result = callback(input_ids, scores)

        # Result should be a list with exactly one 0.0, rest -inf.
        assert isinstance(result, list)
        assert len(result) == 10
        assert result.count(0.0) == 1
        assert sum(1 for v in result if v == float("-inf")) == 9

    def test_dominant_token_selected(self) -> None:
        """When one logit overwhelmingly dominates, it is always selected."""
        from qr_sampler.adapters.llamacpp import QRSamplerCallback

        config = _make_config()
        callback = QRSamplerCallback(config=config, vocab_size=10)

        input_ids = [1]
        scores = [-100.0, -100.0, -100.0, 100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0]

        result = callback(input_ids, scores)
        assert result[3] == 0.0, "Token 3 should be selected"

    def test_diagnostic_records(self) -> None:
        """Diagnostic records are produced with diagnostic_mode=True."""
        from qr_sampler.adapters.llamacpp import QRSamplerCallback

        config = _make_config(diagnostic_mode=True)
        callback = QRSamplerCallback(config=config, vocab_size=10)

        scores = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]
        callback([], scores)

        records = callback.sampling_logger.get_diagnostic_data()
        assert len(records) == 1
        assert records[0].token_id >= 0
        assert records[0].token_id < 10

    def test_close(self) -> None:
        """Callback close works correctly."""
        from qr_sampler.adapters.llamacpp import QRSamplerCallback

        config = _make_config()
        callback = QRSamplerCallback(config=config)
        callback.close()  # Before init -- no-op.


class TestAdapterImports:
    """Test the adapter package imports."""

    def test_lazy_import_transformers(self) -> None:
        """Lazy import of QRSamplerLogitsProcessorHF works."""
        from qr_sampler.adapters import QRSamplerLogitsProcessorHF
        from qr_sampler.adapters.transformers import (
            QRSamplerLogitsProcessorHF as Direct,
        )

        assert QRSamplerLogitsProcessorHF is Direct

    def test_lazy_import_llamacpp(self) -> None:
        """Lazy import of QRSamplerCallback works."""
        from qr_sampler.adapters import QRSamplerCallback
        from qr_sampler.adapters.llamacpp import QRSamplerCallback as Direct

        assert QRSamplerCallback is Direct

    def test_unknown_attr_raises(self) -> None:
        """Accessing unknown attribute on adapters package raises AttributeError."""
        import qr_sampler.adapters

        with pytest.raises(AttributeError):
            _ = qr_sampler.adapters.NonExistentClass  # type: ignore[attr-defined]
