"""Tests for analysis.persistence -- JSONL save/load round-trip."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from qr_sampler.analysis.persistence import load_records, save_records
from qr_sampler.logging.types import TokenSamplingRecord


def _make_record(**overrides: object) -> TokenSamplingRecord:
    """Create a TokenSamplingRecord with sensible defaults."""
    defaults: dict[str, object] = {
        "timestamp_ns": 1_000_000_000,
        "entropy_fetch_ms": 1.5,
        "total_sampling_ms": 3.0,
        "entropy_source_used": "mock",
        "entropy_is_fallback": False,
        "sample_mean": 127.5,
        "z_score": 0.1,
        "u_value": 0.55,
        "temperature_strategy": "fixed",
        "shannon_entropy": 2.3,
        "temperature_used": 0.7,
        "token_id": 42,
        "token_rank": 0,
        "token_prob": 0.15,
        "num_candidates": 100,
        "config_hash": "abcdef1234567890",
    }
    defaults.update(overrides)
    return TokenSamplingRecord(**defaults)  # type: ignore[arg-type]


class TestSaveRecords:
    """Tests for save_records."""

    def test_save_creates_file(self, tmp_path: Path) -> None:
        """Saving records creates the JSONL file."""
        path = tmp_path / "out.jsonl"
        save_records([_make_record()], path)
        assert path.exists()

    def test_save_without_metadata(self, tmp_path: Path) -> None:
        """Without metadata, file contains only record lines."""
        path = tmp_path / "out.jsonl"
        records = [_make_record(token_id=1), _make_record(token_id=2)]
        save_records(records, path)

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            obj = json.loads(line)
            assert "_meta" not in obj

    def test_save_with_metadata(self, tmp_path: Path) -> None:
        """Metadata is written as the first line with _meta sentinel."""
        path = tmp_path / "out.jsonl"
        save_records(
            [_make_record()],
            path,
            metadata={"experiment": "test_1"},
        )
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2

        meta = json.loads(lines[0])
        assert meta["_meta"] is True
        assert meta["experiment"] == "test_1"
        assert "session_id" in meta
        assert "timestamp" in meta

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Intermediate directories are created automatically."""
        path = tmp_path / "sub" / "dir" / "out.jsonl"
        save_records([_make_record()], path)
        assert path.exists()

    def test_save_empty_list(self, tmp_path: Path) -> None:
        """Saving an empty list produces a valid file."""
        path = tmp_path / "empty.jsonl"
        save_records([], path)
        assert path.exists()
        assert path.read_text().strip() == ""


class TestLoadRecords:
    """Tests for load_records."""

    def test_round_trip_no_metadata(self, tmp_path: Path) -> None:
        """Records survive a save/load round trip."""
        path = tmp_path / "rt.jsonl"
        original = [_make_record(token_id=10), _make_record(token_id=20)]
        save_records(original, path)

        meta, loaded = load_records(path)
        assert meta == {}
        assert len(loaded) == 2
        assert loaded[0]["token_id"] == 10
        assert loaded[1]["token_id"] == 20

    def test_round_trip_with_metadata(self, tmp_path: Path) -> None:
        """Metadata and records both survive a round trip."""
        path = tmp_path / "rt_meta.jsonl"
        save_records(
            [_make_record()],
            path,
            metadata={"run": "alpha"},
        )

        meta, loaded = load_records(path)
        assert meta["_meta"] is True
        assert meta["run"] == "alpha"
        assert len(loaded) == 1

    def test_load_empty_file(self, tmp_path: Path) -> None:
        """Loading an empty file returns empty metadata and records."""
        path = tmp_path / "empty.jsonl"
        save_records([], path)
        meta, loaded = load_records(path)
        assert meta == {}
        assert loaded == []

    def test_all_fields_preserved(self, tmp_path: Path) -> None:
        """Every TokenSamplingRecord field is present in the loaded dict."""
        path = tmp_path / "fields.jsonl"
        record = _make_record(
            injection_alpha=0.05,
            injection_beta=0.1,
            injection_step=0.02,
            injection_scale=1.5,
        )
        save_records([record], path)

        _, loaded = load_records(path)
        rec = loaded[0]
        assert rec["u_value"] == pytest.approx(0.55)
        assert rec["injection_alpha"] == pytest.approx(0.05)
        assert rec["injection_beta"] == pytest.approx(0.1)
        assert rec["injection_step"] == pytest.approx(0.02)
        assert rec["injection_scale"] == pytest.approx(1.5)
        assert rec["config_hash"] == "abcdef1234567890"

    def test_string_path_accepted(self, tmp_path: Path) -> None:
        """Both str and Path work for the path argument."""
        path = str(tmp_path / "str.jsonl")
        save_records([_make_record()], path)
        _meta, loaded = load_records(path)
        assert len(loaded) == 1
