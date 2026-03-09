"""Tests for the Open WebUI example filter."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

from qr_sampler.config import validate_extra_args


def _load_filter_class() -> type:
    repo_root = Path(__file__).resolve().parents[1]
    filter_path = repo_root / "examples" / "open-webui" / "qr_sampler_filter.py"
    spec = importlib.util.spec_from_file_location("qr_sampler_filter_example", filter_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Filter


def test_open_webui_filter_only_injects_valid_per_request_fields() -> None:
    """The example filter should never inject infrastructure-only qr_* keys."""
    filter_cls = _load_filter_class()
    filter_instance = filter_cls()

    body = {"model": "demo-model"}
    updated = asyncio.run(filter_instance.inlet(body))
    qr_args = {k: v for k, v in updated.items() if k.startswith("qr_")}

    assert qr_args
    validate_extra_args(qr_args)
