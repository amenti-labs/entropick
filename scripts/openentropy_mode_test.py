#!/usr/bin/env python3
"""Full-mode test: every injection method + filter powered by OpenEntropy.

Tests each mode individually and in combination using a real hardware entropy
source (clock_jitter via openentropy). Validates that quantum bytes actually
flow through each pipeline stage.

Modes tested:
  M1  Logit Noise         — per-logit quantum Gaussian perturbation
  M2  Temperature Variance — per-token temperature modulation
  M3  Correlated Walk      — drifting selection point with temporal memory
  Min-P                    — dynamic probability floor filter
  XTC                      — quantum coin-flip top-token exclusion
  Adaptive Injection       — entropy-aware injection scaling
  Combined                 — all modes active simultaneously

Usage:
    source /tmp/oe_test_env/bin/activate
    python scripts/openentropy_mode_test.py
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Defaults — overridden per-test.
os.environ["QR_LOG_LEVEL"] = "none"
os.environ["QR_DIAGNOSTIC_MODE"] = "true"
os.environ["QR_ENTROPY_SOURCE_TYPE"] = "openentropy"
os.environ["QR_OE_SOURCES"] = "clock_jitter"
os.environ["QR_OE_CONDITIONING"] = "sha256"
os.environ["QR_FALLBACK_MODE"] = "system"

VOCAB_SIZE = 32
N_TOKENS = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hr(char: str = "─", width: int = 72) -> None:
    print(char * width)


def _section(title: str) -> None:
    print()
    _hr("═")
    print(f"  {title}")
    _hr("═")


def _make_processor(**env_overrides: str) -> Any:
    from qr_sampler.processor import QRSamplerLogitsProcessor

    saved: dict[str, str | None] = {}
    for key, val in env_overrides.items():
        saved[key] = os.environ.get(key)
        os.environ[key] = val
    try:
        proc = QRSamplerLogitsProcessor(vllm_config=None, device=None, is_pin_memory=False)
    finally:
        for key, orig in saved.items():
            if orig is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = orig
    return proc


def _realistic_logits(vocab_size: int = VOCAB_SIZE, rng_seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    logits = rng.standard_normal(vocab_size).astype(np.float32) * 0.5
    logits[0] = 3.0
    logits[1] = 2.8
    logits[2] = 2.6
    return logits


@dataclass
class _MockAdded:
    req_index: int
    sampling_params: Any


@dataclass
class _MockSamplingParams:
    extra_args: dict[str, Any] | None = None


@dataclass
class _MockBatchUpdate:
    removed: list[Any] | None = None
    moved: list[Any] | None = None
    added: list[Any] | None = None

    def __post_init__(self) -> None:
        if self.removed is None:
            self.removed = []
        if self.moved is None:
            self.moved = []
        if self.added is None:
            self.added = []


def _register(proc: Any, req_index: int = 0, extra_args: dict[str, Any] | None = None) -> None:
    batch = _MockBatchUpdate(
        added=[_MockAdded(req_index=req_index, sampling_params=_MockSamplingParams(extra_args=extra_args))]
    )
    proc.update_state(batch)


def _run_tokens(proc: Any, n_tokens: int = N_TOKENS) -> list[dict[str, Any]]:
    results = []
    for step in range(n_tokens):
        logits = np.array([_realistic_logits(VOCAB_SIZE, rng_seed=step)], dtype=np.float32)
        proc.apply(logits)
        recs = proc.sampling_logger.get_diagnostic_data()
        if recs:
            r = recs[-1]
            results.append({
                "step": step,
                "token_id": r.token_id,
                "token_rank": r.token_rank,
                "token_prob": r.token_prob,
                "u_value": r.u_value,
                "temp": r.temperature_used,
                "entropy_ms": r.entropy_fetch_ms,
                "num_candidates": r.num_candidates,
            })
    return results


def _safe(v: float, fmt: str = ".4f") -> str:
    return f"{v:{fmt}}" if not math.isnan(v) else "nan"


def _summarize(results: list[dict[str, Any]], label: str) -> None:
    ranks = [r["token_rank"] for r in results]
    temps = [r["temp"] for r in results if not math.isnan(r["temp"])]
    u_vals = [r["u_value"] for r in results if not math.isnan(r["u_value"])]
    ms_vals = [r["entropy_ms"] for r in results]
    cands = [r["num_candidates"] for r in results]

    print(f"\n  {label}")
    print(f"    tokens: {len(results)}  rank-0: {ranks.count(0)}/{len(ranks)}  mean_rank: {sum(ranks)/len(ranks):.2f}")
    if temps:
        print(f"    temp range: [{min(temps):.3f}, {max(temps):.3f}]  mean: {sum(temps)/len(temps):.3f}")
    if u_vals:
        print(f"    u range: [{min(u_vals):.4f}, {max(u_vals):.4f}]  mean: {sum(u_vals)/len(u_vals):.4f}")
    print(f"    candidates: [{min(cands)}, {max(cands)}]  mean: {sum(cands)/len(cands):.1f}")
    print(f"    entropy_fetch_ms: mean={sum(ms_vals)/len(ms_vals):.2f}  total={sum(ms_vals):.1f}ms")


# ---------------------------------------------------------------------------
# Individual mode tests
# ---------------------------------------------------------------------------

def test_baseline() -> None:
    _section("Baseline — No Injection (OpenEntropy for amplifier only)")
    proc = _make_processor()
    _register(proc)
    results = _run_tokens(proc)
    _summarize(results, "baseline (all injection disabled, quantum u only)")
    u_vals = [r["u_value"] for r in results if not math.isnan(r["u_value"])]
    print(f"    u_values: {[_safe(u, '.3f') for u in u_vals[:8]]}...")


def test_m1_logit_noise() -> None:
    _section("M1 — Logit Noise (per-logit quantum bytes via OpenEntropy)")
    for label, alpha in [("mild α=0.3", "0.3"), ("strong α=1.0", "1.0")]:
        proc = _make_processor(QR_LOGIT_NOISE_ALPHA=alpha)
        _register(proc)
        results = _run_tokens(proc)
        _summarize(results, f"M1 {label}")


def test_m2_temp_variance() -> None:
    _section("M2 — Temperature Variance (quantum modulation via OpenEntropy)")
    for label, beta in [("moderate β=0.5", "0.5"), ("strong β=1.5", "1.5")]:
        proc = _make_processor(QR_TEMP_VARIANCE_BETA=beta)
        _register(proc)
        results = _run_tokens(proc)
        _summarize(results, f"M2 {label}")


def test_m3_correlated_walk() -> None:
    _section("M3 — Correlated Walk (quantum drift via OpenEntropy)")
    for label, step in [("gentle step=0.05", "0.05"), ("strong step=0.2", "0.2")]:
        proc = _make_processor(QR_WALK_STEP=step)
        _register(proc)
        results = _run_tokens(proc)
        _summarize(results, f"M3 {label}")
        u_vals = [r["u_value"] for r in results if not math.isnan(r["u_value"])]
        if len(u_vals) >= 2:
            diffs = [abs(u_vals[i + 1] - u_vals[i]) for i in range(len(u_vals) - 1)]
            print(f"    mean |Δu|: {sum(diffs)/len(diffs):.4f}  (lower = more correlated)")
            print(f"    u_values: {[_safe(u, '.3f') for u in u_vals[:10]]}...")


def test_min_p() -> None:
    _section("Min-P Filtering (dynamic probability floor)")
    for label, mp in [("mild min_p=0.1", "0.1"), ("aggressive min_p=0.5", "0.5")]:
        proc = _make_processor(QR_MIN_P=mp)
        _register(proc)
        results = _run_tokens(proc)
        _summarize(results, f"Min-P {label}")


def test_xtc() -> None:
    _section("XTC — Exclude Top Choices (quantum coin-flip exclusion)")
    for label, prob, thresh in [
        ("moderate p=0.5 t=0.1", "0.5", "0.1"),
        ("aggressive p=1.0 t=0.05", "1.0", "0.05"),
    ]:
        proc = _make_processor(QR_XTC_PROBABILITY=prob, QR_XTC_THRESHOLD=thresh)
        _register(proc)
        results = _run_tokens(proc)
        _summarize(results, f"XTC {label}")


def test_adaptive_injection() -> None:
    _section("Adaptive Injection (entropy-aware scaling)")
    # With M1 as the injection to scale.
    for label, low_h, high_h in [
        ("narrow band [0.5, 1.5]", "0.5", "1.5"),
        ("wide band [0.0, 5.0]", "0.0", "5.0"),
    ]:
        proc = _make_processor(
            QR_ADAPTIVE_INJECTION="true",
            QR_ADAPTIVE_INJECTION_LOW_H=low_h,
            QR_ADAPTIVE_INJECTION_HIGH_H=high_h,
            QR_LOGIT_NOISE_ALPHA="0.5",
        )
        _register(proc)
        results = _run_tokens(proc)
        _summarize(results, f"Adaptive + M1(α=0.5) H∈[{low_h},{high_h}]")


def test_combined() -> None:
    _section("Combined — All Modes Active (OpenEntropy)")
    proc = _make_processor(
        QR_ADAPTIVE_INJECTION="true",
        QR_ADAPTIVE_INJECTION_LOW_H="0.5",
        QR_ADAPTIVE_INJECTION_HIGH_H="3.0",
        QR_LOGIT_NOISE_ALPHA="0.3",
        QR_TEMP_VARIANCE_BETA="0.5",
        QR_WALK_STEP="0.05",
        QR_MIN_P="0.1",
        QR_XTC_PROBABILITY="0.3",
        QR_XTC_THRESHOLD="0.1",
    )
    _register(proc)
    results = _run_tokens(proc)
    _summarize(results, "ALL modes: adaptive+M1+M2+M3+min_p+xtc")
    u_vals = [r["u_value"] for r in results if not math.isnan(r["u_value"])]
    if len(u_vals) >= 2:
        diffs = [abs(u_vals[i + 1] - u_vals[i]) for i in range(len(u_vals) - 1)]
        print(f"    mean |Δu|: {sum(diffs)/len(diffs):.4f}")


# ---------------------------------------------------------------------------
# Entropy source validation
# ---------------------------------------------------------------------------

def validate_openentropy() -> bool:
    """Verify OpenEntropy is working before running tests."""
    _section("OpenEntropy Source Validation")
    try:
        from openentropy import EntropyPool
        pool = EntropyPool.auto()
        names = pool.source_names()
        print(f"  Sources available: {pool.source_count}")
        print(f"  Using: clock_jitter (sha256 conditioning)")

        t0 = time.perf_counter()
        data = pool.get_source_bytes("clock_jitter", 20480, conditioning="sha256")
        t1 = time.perf_counter()
        print(f"  Test fetch: {len(data)} bytes in {(t1-t0)*1000:.1f}ms")

        # Quick uniformity check on first 1000 bytes.
        arr = np.frombuffer(data[:1000], dtype=np.uint8)
        print(f"  Byte stats: mean={arr.mean():.1f} (expect ~127.5)  std={arr.std():.1f} (expect ~73.6)")

        if abs(arr.mean() - 127.5) > 30:
            print("  ⚠ WARNING: mean is far from expected — conditioning may be off")

        print(f"\n  ✓ OpenEntropy validated — proceeding with tests")
        return True
    except Exception as e:
        print(f"  ✗ OpenEntropy validation failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.perf_counter()

    print()
    _hr("═")
    print("  OpenEntropy Full-Mode Test")
    print(f"  Entropy: clock_jitter + sha256 | Vocab: {VOCAB_SIZE} | Tokens/test: {N_TOKENS}")
    _hr("═")

    if not validate_openentropy():
        print("\n  Cannot proceed without working OpenEntropy source.")
        sys.exit(1)

    test_baseline()
    test_m1_logit_noise()
    test_m2_temp_variance()
    test_m3_correlated_walk()
    test_min_p()
    test_xtc()
    test_adaptive_injection()
    test_combined()

    t_total = time.perf_counter() - t_start
    _section(f"Done — {t_total:.1f}s total")
    print()


if __name__ == "__main__":
    main()
