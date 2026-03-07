#!/usr/bin/env python3
"""
Cross-mode injection test across benchmark prompts.

Tests all three injection modes against prompts from the Entropy Seeding Benchmark:
  M1 — Logit noise          (qr-sampler processor with simulated logits)
  M2 — Temperature variance (qr-sampler processor with simulated logits)
  M3 — Correlated walk      (qr-sampler processor with simulated logits)

All modes operate at the vLLM LogitsProcessor level and are simulated here using
a realistic logit distribution that mimics an LLM at a high-entropy decision point.

Usage:
    python scripts/injection_mode_test.py

Requirements:
    - qr-sampler installed (pip install -e .)
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Any

import numpy as np

# Ensure src/ is on path when run from repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

os.environ.setdefault("QR_LOG_LEVEL", "none")
os.environ.setdefault("QR_DIAGNOSTIC_MODE", "true")
os.environ.setdefault("QR_ENTROPY_SOURCE_TYPE", "system")
os.environ.setdefault("QR_FALLBACK_MODE", "system")

VOCAB_SIZE = 32
SEP = "─" * 70

# ---------------------------------------------------------------------------
# Benchmark prompts (subset from Entropy_Seeding_Benchmark.pdf)
# ---------------------------------------------------------------------------

PROMPTS: list[tuple[str, str, str]] = [
    (
        "creative",
        "[CREATIVE]",
        "The old lighthouse keeper had never seen anything like it.",
    ),
    (
        "philosophical",
        "[PHILOSOPHICAL]",
        "Think of a color you have never seen before. Describe it in detail.",
    ),
    (
        "colored_entropy",
        "[COLORED ENTROPY]",
        "If mathematics was discovered by an ancient civilization that understood "
        "it as a conscious entity, how would that change our understanding of "
        "consciousness itself?",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hr(char: str = "─", width: int = 70) -> None:
    print(char * width)


def _section(title: str) -> None:
    print()
    _hr("═")
    print(f"  {title}")
    _hr("═")


def _subsection(title: str) -> None:
    print()
    _hr()
    print(f"  {title}")
    _hr()


def _make_processor(**env_overrides: str) -> Any:
    """Build a QRSamplerLogitsProcessor with env-var config overrides."""
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
    """Produce a realistic-looking logit vector with genuine uncertainty.

    Models a high-entropy decision point: the top-3 tokens are close in
    probability (simulating a creative or ambiguous generation step), with
    a long tail of less-likely tokens.
    """
    rng = np.random.default_rng(rng_seed)
    # Sharp baseline: most tokens unlikely.
    logits = rng.standard_normal(vocab_size).astype(np.float32) * 0.5
    # Make top-3 competitive.
    logits[0] = 3.0
    logits[1] = 2.8
    logits[2] = 2.6
    return logits


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    shifted = logits / temperature - logits.max() / temperature
    exp = np.exp(shifted)
    return exp / exp.sum()


def _top3_str(logits: np.ndarray, temperature: float = 0.7) -> str:
    probs = _softmax(logits, temperature)
    top3 = np.argsort(probs)[::-1][:3]
    return "  ".join(f"tok{t}({probs[t]:.3f})" for t in top3)


def _run_processor_tokens(
    proc: Any,
    n_tokens: int = 8,
    vocab_size: int = VOCAB_SIZE,
) -> list[dict[str, Any]]:
    """Run processor for n_tokens using fresh logits each step. Returns records."""
    results = []
    for step in range(n_tokens):
        logits = np.array([_realistic_logits(vocab_size, rng_seed=step)], dtype=np.float32)
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
            })
    return results


def _print_processor_results(results: list[dict[str, Any]], label: str) -> None:
    print(f"\n  {label}")
    print(f"  {'step':>4}  {'token_id':>8}  {'rank':>4}  {'prob':>6}  {'u_value':>8}  {'temp':>6}")
    print(f"  {'----':>4}  {'--------':>8}  {'----':>4}  {'------':>6}  {'--------':>8}  {'------':>6}")
    for r in results:
        prob_str = f"{r['token_prob']:.4f}" if not math.isnan(r["token_prob"]) else "   nan"
        u_str = f"{r['u_value']:.6f}" if not math.isnan(r["u_value"]) else "     nan"
        temp_str = f"{r['temp']:.4f}" if not math.isnan(r["temp"]) else "   nan"
        print(f"  {r['step']:>4}  {r['token_id']:>8}  {r['token_rank']:>4}  {prob_str:>6}  {u_str:>8}  {temp_str:>6}")


# ---------------------------------------------------------------------------
# Test: M1 — Logit noise (direct per-logit quantum bytes)
# ---------------------------------------------------------------------------

def test_m1_logit_noise() -> None:
    _section("M1 — Logit Perturbation (Direct Per-Logit Quantum Bytes)")
    print("  alpha=0.0 (disabled) vs alpha=0.3 vs alpha=1.0")
    print("  Benchmark spec: logits += alpha * probit(entropy_bytes_to_float(source.get_bytes(vocab_size * 4)))")
    print("  Expected: higher alpha -> more rank 1/2 tokens selected (distribution broadens)")
    print()

    base_logits = _realistic_logits(VOCAB_SIZE)
    print(f"  Base logit top-3 at temp=0.7: {_top3_str(base_logits)}")
    print()

    N_TOKENS = 20

    for alpha_label, alpha_val in [("disabled (alpha=0.0)", "0.0"), ("mild (alpha=0.3)", "0.3"), ("strong (alpha=1.0)", "1.0")]:
        proc = _make_processor(
            QR_ENTROPY_SOURCE_TYPE="system",
            QR_FALLBACK_MODE="system",
            QR_LOG_LEVEL="none",
            QR_DIAGNOSTIC_MODE="true",
            QR_LOGIT_NOISE_ALPHA=alpha_val,
        )
        results = _run_processor_tokens(proc, n_tokens=N_TOKENS)
        ranks = [r["token_rank"] for r in results]
        rank0_count = ranks.count(0)
        mean_rank = sum(ranks) / len(ranks) if ranks else 0
        print(f"  {alpha_label}")
        print(f"    rank-0 selections: {rank0_count}/{N_TOKENS}  mean_rank: {mean_rank:.2f}")
        u_preview = [f"{r['u_value']:.3f}" for r in results[:6]]
        print(f"    u_values: {u_preview}...")
        print()


# ---------------------------------------------------------------------------
# Test: M2 — Temperature variance
# ---------------------------------------------------------------------------

def test_m2_temp_variance() -> None:
    _section("M2 — Temperature Modulation (Per-Token Variance)")
    print("  beta=0.0 (disabled) vs beta=0.5 vs beta=1.5")
    print("  Expected: higher beta -> temperature varies more -> rank variance increases")
    print()

    N_TOKENS = 20

    for beta_label, beta_val in [("disabled (beta=0.0)", "0.0"), ("moderate (beta=0.5)", "0.5"), ("strong (beta=1.5)", "1.5")]:
        proc = _make_processor(
            QR_ENTROPY_SOURCE_TYPE="system",
            QR_FALLBACK_MODE="system",
            QR_LOG_LEVEL="none",
            QR_DIAGNOSTIC_MODE="true",
            QR_TEMP_VARIANCE_BETA=beta_val,
            QR_FIXED_TEMPERATURE="0.7",
        )
        results = _run_processor_tokens(proc, n_tokens=N_TOKENS)
        temps = [r["temp"] for r in results if not math.isnan(r["temp"])]
        ranks = [r["token_rank"] for r in results]
        print(f"  {beta_label}")
        if temps:
            print(f"    temperature range: [{min(temps):.3f}, {max(temps):.3f}]  mean: {sum(temps)/len(temps):.3f}")
        print(f"    rank-0 selections: {ranks.count(0)}/{N_TOKENS}  mean_rank: {sum(ranks)/len(ranks):.2f}")
        print()


# ---------------------------------------------------------------------------
# Test: M3 — Correlated walk
# ---------------------------------------------------------------------------

def test_m3_correlated_walk() -> None:
    _section("M3 — Correlated Walk (Temporal Correlation Across Tokens)")
    print("  walk_step=0.0 (disabled) vs walk_step=0.05 vs walk_step=0.2")
    print("  Expected: higher walk_step -> u_values drift in correlated sequences")
    print()

    from dataclasses import dataclass

    @dataclass
    class _MockAdded:
        req_index: int
        sampling_params: Any

    @dataclass
    class _MockSamplingParams:
        extra_args: dict[str, Any] | None = None

    @dataclass
    class _MockBatchUpdate:
        removed: list = None  # type: ignore
        moved: list = None    # type: ignore
        added: list = None    # type: ignore
        def __post_init__(self) -> None:
            if self.removed is None: self.removed = []
            if self.moved is None: self.moved = []
            if self.added is None: self.added = []

    N_TOKENS = 16

    for step_label, step_val in [("disabled (step=0.0)", "0.0"), ("gentle (step=0.05)", "0.05"), ("strong (step=0.2)", "0.2")]:
        proc = _make_processor(
            QR_ENTROPY_SOURCE_TYPE="system",
            QR_FALLBACK_MODE="system",
            QR_LOG_LEVEL="none",
            QR_DIAGNOSTIC_MODE="true",
            QR_WALK_STEP=step_val,
        )
        batch = _MockBatchUpdate(
            added=[_MockAdded(req_index=0, sampling_params=_MockSamplingParams())]
        )
        proc.update_state(batch)

        results = _run_processor_tokens(proc, n_tokens=N_TOKENS)
        u_vals = [r["u_value"] for r in results if not math.isnan(r["u_value"])]

        if len(u_vals) >= 4:
            diffs = [abs(u_vals[i+1] - u_vals[i]) for i in range(len(u_vals)-1)]
            mean_step = sum(diffs) / len(diffs)
        else:
            mean_step = float("nan")

        print(f"  {step_label}")
        print(f"    u_values:  {[f'{u:.3f}' for u in u_vals]}")
        print(f"    mean |Δu|: {mean_step:.4f}  (walk coherence measure)")
        if hasattr(proc, "_request_states") and 0 in proc._request_states:
            final_pos = proc._request_states[0].walk_position
            print(f"    final walk_position: {final_pos:.6f}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    _hr("═")
    print("  Entropy Seeding Benchmark — Injection Mode Comparison")
    print(f"  Vocab (simulated): {VOCAB_SIZE}")
    _hr("═")

    print("\n  Modes tested:")
    print("    M1 Logit Noise      — direct per-logit quantum bytes (benchmark Method 1)")
    print("    M2 Temp Variance    — processor-level, simulated logits")
    print("    M3 Correlated Walk  — processor-level, simulated logits, per-request state")

    test_m1_logit_noise()
    test_m2_temp_variance()
    test_m3_correlated_walk()

    _section("Done")
    print("  All injection modes tested.")
    print()


if __name__ == "__main__":
    main()
