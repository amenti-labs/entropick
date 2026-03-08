"""Tests for SelectionDrift injection method."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from qr_sampler.config import QRSamplerConfig
from qr_sampler.entropy.mock import MockUniformSource
from qr_sampler.exceptions import EntropyUnavailableError
from qr_sampler.injection.selection_drift import SelectionDrift


@pytest.fixture()
def source() -> MockUniformSource:
    """Seeded mock entropy source for reproducible tests."""
    return MockUniformSource(seed=42)


class TestSelectionDrift:
    """Tests for SelectionDrift.step()."""

    def test_step_drifts_from_initial_position(
        self,
        source: MockUniformSource,
    ) -> None:
        """With drift_step=0.1, position should change from initial."""
        config = QRSamplerConfig(
            _env_file=None,  # type: ignore[call-arg]
            drift_step=0.1,
        )
        initial_u = 0.5
        initial_pos = 0.5
        new_u, new_pos = SelectionDrift.step(initial_u, source, config, initial_pos)
        assert new_pos != pytest.approx(initial_pos)
        # Both return values should be the drift position
        assert new_u == pytest.approx(new_pos)

    def test_step_noop_when_disabled(
        self,
        source: MockUniformSource,
    ) -> None:
        """With drift_step=0.0, position and u should be unchanged."""
        config = QRSamplerConfig(
            _env_file=None,  # type: ignore[call-arg]
            drift_step=0.0,
        )
        new_u, new_pos = SelectionDrift.step(0.42, source, config, 0.5)
        assert new_u == pytest.approx(0.42)
        assert new_pos == pytest.approx(0.5)

    def test_step_stays_in_bounds_over_many_steps(self) -> None:
        """Over 1000 steps, drift position must always stay in [0, 1)."""
        config = QRSamplerConfig(
            _env_file=None,  # type: ignore[call-arg]
            drift_step=0.1,
        )
        pos = 0.5
        u = 0.5
        for seed in range(1000):
            src = MockUniformSource(seed=seed)
            u, pos = SelectionDrift.step(u, src, config, pos)
            assert 0.0 <= pos < 1.0, f"step {seed}: position {pos} out of [0, 1)"
            assert 0.0 <= u < 1.0, f"step {seed}: u {u} out of [0, 1)"

    def test_step_returns_same_u_and_position(
        self,
        source: MockUniformSource,
    ) -> None:
        """Both return values (new_u, new_pos) should be equal."""
        config = QRSamplerConfig(
            _env_file=None,  # type: ignore[call-arg]
            drift_step=0.2,
        )
        new_u, new_pos = SelectionDrift.step(0.5, source, config, 0.5)
        assert new_u == pytest.approx(new_pos)

    def test_step_handles_entropy_unavailable(self) -> None:
        """When entropy source raises, returns (u, drift_position) unchanged."""
        config = QRSamplerConfig(
            _env_file=None,  # type: ignore[call-arg]
            drift_step=0.1,
        )
        failing_source = MagicMock()
        failing_source.get_random_bytes.side_effect = EntropyUnavailableError("test")
        new_u, new_pos = SelectionDrift.step(0.42, failing_source, config, 0.75)
        assert new_u == pytest.approx(0.42)
        assert new_pos == pytest.approx(0.75)

    def test_step_handles_empty_entropy_payload(self) -> None:
        """When entropy source returns empty bytes, returns unchanged values."""
        config = QRSamplerConfig(
            _env_file=None,  # type: ignore[call-arg]
            drift_step=0.1,
        )
        empty_source = MagicMock()
        empty_source.get_random_bytes.return_value = b""
        new_u, new_pos = SelectionDrift.step(0.42, empty_source, config, 0.75)
        assert new_u == pytest.approx(0.42)
        assert new_pos == pytest.approx(0.75)

    def test_step_wraps_at_boundary(self) -> None:
        """Position near 1.0 with positive drift should wrap via modulo."""
        config = QRSamplerConfig(
            _env_file=None,  # type: ignore[call-arg]
            drift_step=0.5,
        )
        # Run many seeds starting near the boundary to exercise wrapping.
        wrapped_count = 0
        for seed in range(100):
            src = MockUniformSource(seed=seed)
            _, pos = SelectionDrift.step(0.5, src, config, 0.95)
            assert 0.0 <= pos < 1.0
            if pos < 0.5:
                wrapped_count += 1
        # At least some should have wrapped around
        assert wrapped_count > 0, "No wrap-around observed in 100 trials"
