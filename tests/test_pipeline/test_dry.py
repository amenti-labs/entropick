"""Tests for the DRY (Don't Repeat Yourself) penalty stage."""

from __future__ import annotations

import numpy as np

from qr_sampler.config import QRSamplerConfig
from qr_sampler.entropy.mock import MockUniformSource
from qr_sampler.pipeline.context import SamplingContext
from qr_sampler.stages.dry import DRYPenaltyStage


def _make_config(**overrides: object) -> QRSamplerConfig:
    """Create a config with sensible DRY defaults and optional overrides."""
    defaults: dict[str, object] = {
        "dry_multiplier": 2.0,
        "dry_base": 1.75,
        "dry_allowed_length": 2,
        "dry_penalty_last_n": -1,
        "dry_sequence_breakers": "",
    }
    defaults.update(overrides)
    return QRSamplerConfig(_env_file=None, **defaults)  # type: ignore[call-arg]


def _make_ctx(
    logits: np.ndarray,
    token_history: list[int],
    config: QRSamplerConfig | None = None,
) -> SamplingContext:
    """Build a minimal SamplingContext for DRY stage testing."""
    if config is None:
        config = _make_config()
    return SamplingContext(
        row=logits,
        config=config,
        entropy_source=MockUniformSource(seed=42),
        amplifier=None,  # type: ignore[arg-type]
        temperature_strategy=None,  # type: ignore[arg-type]
        config_hash="test",
        stage_state={"token_history": token_history},
    )


class TestDRYPenaltyStage:
    """Unit tests for DRYPenaltyStage."""

    def test_noop_when_multiplier_zero(self) -> None:
        """No change to logits when dry_multiplier=0."""
        config = _make_config(dry_multiplier=0.0)
        logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float64)
        original = logits.copy()

        ctx = _make_ctx(logits, token_history=[0, 1, 0, 1], config=config)
        DRYPenaltyStage()(ctx)

        np.testing.assert_array_equal(ctx.row, original)

    def test_noop_when_no_history(self) -> None:
        """No change to logits when token_history is empty."""
        logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float64)
        original = logits.copy()

        ctx = _make_ctx(logits, token_history=[])
        DRYPenaltyStage()(ctx)

        np.testing.assert_array_equal(ctx.row, original)

    def test_noop_when_penalty_last_n_zero(self) -> None:
        """No change to logits when dry_penalty_last_n=0."""
        config = _make_config(dry_penalty_last_n=0)
        logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float64)
        original = logits.copy()

        ctx = _make_ctx(logits, token_history=[0, 1, 0, 1], config=config)
        DRYPenaltyStage()(ctx)

        np.testing.assert_array_equal(ctx.row, original)

    def test_penalizes_repeated_ngrams(self) -> None:
        """Tokens that would extend a repeated n-gram pattern get penalized.

        History: [3, 4, 3, 4]
        The suffix [3, 4] at the end matches the earlier [3, 4].
        The token that followed the earlier match (token 3 at index 2) would
        continue the repetition, so token 3's logit should decrease.
        """
        config = _make_config(
            dry_multiplier=2.0,
            dry_base=1.75,
            dry_allowed_length=2,
        )
        logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float64)
        original = logits.copy()

        ctx = _make_ctx(logits, token_history=[3, 4, 3, 4], config=config)
        DRYPenaltyStage()(ctx)

        # Token 3 should be penalized (logit decreased).
        assert ctx.row[3] < original[3]
        # Other tokens should remain unchanged.
        assert ctx.row[0] == original[0]
        assert ctx.row[1] == original[1]
        assert ctx.row[4] == original[4]

    def test_penalty_scales_with_base(self) -> None:
        """Longer matches receive exponentially higher penalties via dry_base.

        With base=2.0 and allowed_length=2:
          - match_length=2 -> penalty = multiplier * 2^(2-2) = multiplier * 1
          - match_length=3 -> penalty = multiplier * 2^(3-2) = multiplier * 2

        History: [1, 2, 3, 1, 2, 3]
        Suffix [2, 3] matches earlier [2, 3] at position 1 -> follow token=3, length=3.
        Suffix [1, 2, 3] matches earlier [1, 2, 3] -> follow token would be at index 3
        which is 1, length=4.
        Token 1 (longer match) should get a bigger penalty than token 3 (shorter match).
        """
        multiplier = 1.0
        base = 2.0
        config = _make_config(
            dry_multiplier=multiplier,
            dry_base=base,
            dry_allowed_length=2,
        )
        logits = np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float64)
        original = logits.copy()

        # History: [1, 2, 3, 1, 2, 3]
        # Suffix [3] at pos 5 matches [3] at pos 2 -> follow=1 at pos 3, match_len=2
        # Suffix [2, 3] at pos 4-5 matches [2, 3] at pos 1-2 -> follow=1 at pos 3, match_len=3
        # Suffix [1, 2, 3] at pos 3-5 matches [1, 2, 3] at pos 0-2 -> follow idx=3, which
        #   is the token at history[3]=1, match_len=4
        # So token 1 gets max penalty from the longest match.
        ctx = _make_ctx(logits, token_history=[1, 2, 3, 1, 2, 3], config=config)
        DRYPenaltyStage()(ctx)

        penalty_on_1 = original[1] - ctx.row[1]
        # Token 1 should be penalized.
        assert penalty_on_1 > 0
        # The penalty should reflect exponential scaling: multiplier * base^(match_len - allowed)
        # For the longest match (length 4): 1.0 * 2^(4-2) = 4.0
        assert penalty_on_1 == pytest.approx(multiplier * base ** (4 - 2), abs=1e-6)

    def test_sequence_breaker_resets(self) -> None:
        """A breaker token in history causes only tokens after the breaker
        to be considered for n-gram matching.

        History: [0, 1, 0, 999, 0, 1]
        Breaker=999. Effective window after last breaker: [0, 1].
        Without breaker, [0, 1] repeats and token 0 would be penalized.
        With breaker, the window [0, 1] only has one occurrence of any
        pattern suffix, so there is nothing to match against -- no penalty.
        """
        config = _make_config(
            dry_multiplier=2.0,
            dry_sequence_breakers="999",
        )
        logits = np.array([5.0, 4.0, 3.0], dtype=np.float64)
        original = logits.copy()

        ctx = _make_ctx(
            logits,
            token_history=[0, 1, 0, 999, 0, 1],
            config=config,
        )
        DRYPenaltyStage()(ctx)

        # After the breaker (999), only [0, 1] remains.
        # A 2-token window can match suffix of length 1 at most;
        # suffix [1] at pos 1 has no earlier occurrence in [0, 1] at a
        # position that would leave a follow token, so no penalty.
        np.testing.assert_array_equal(ctx.row, original)

        # Now verify that WITHOUT the breaker, the same history DOES produce a penalty.
        config_no_breaker = _make_config(
            dry_multiplier=2.0,
            dry_sequence_breakers="",
        )
        logits2 = np.array([5.0, 4.0, 3.0] + [0.0] * 997, dtype=np.float64)
        original2 = logits2.copy()

        ctx2 = _make_ctx(
            logits2,
            token_history=[0, 1, 0, 999, 0, 1],
            config=config_no_breaker,
        )
        DRYPenaltyStage()(ctx2)

        # Without breaker, 999 is just a normal token.
        # Suffix [0, 1] matches earlier [0, 1] at pos 0-1, follow=0, so token 0 penalized.
        assert ctx2.row[0] < original2[0]

    def test_penalty_last_n_limits_window(self) -> None:
        """Only the last N tokens from history are considered.

        History: [0, 1, 0, 1, 2, 3, 4]
        With penalty_last_n=3, window = [2, 3, 4] -- no repetition.
        With penalty_last_n=-1 (full), the [0, 1, 0, 1] pattern is visible.
        """
        # With a small window, the repetition is outside the window.
        config_small = _make_config(
            dry_multiplier=2.0,
            dry_penalty_last_n=3,
        )
        logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float64)
        original = logits.copy()

        history = [0, 1, 0, 1, 2, 3, 4]
        ctx = _make_ctx(logits, token_history=history, config=config_small)
        DRYPenaltyStage()(ctx)

        # Window is [2, 3, 4] -- all unique, no penalty.
        np.testing.assert_array_equal(ctx.row, original)

        # With full context, the [0, 1] repetition IS visible.
        config_full = _make_config(
            dry_multiplier=2.0,
            dry_penalty_last_n=-1,
        )
        logits2 = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float64)
        original2 = logits2.copy()

        ctx2 = _make_ctx(logits2, token_history=[0, 1, 0, 1], config=config_full)
        DRYPenaltyStage()(ctx2)

        # Token 0 should be penalized because suffix [0, 1] matches.
        assert ctx2.row[0] < original2[0]

    def test_does_not_modify_out_of_range_tokens(self) -> None:
        """Token IDs in history that are >= vocab_size are ignored
        when applying penalties (the guard ``0 <= token_id < len(ctx.row)``).
        """
        config = _make_config(
            dry_multiplier=2.0,
            dry_allowed_length=2,
        )
        # Vocab size = 5 (logits array has 5 elements).
        logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float64)
        original = logits.copy()

        # History tokens 99 and 100 are >= vocab_size=5.
        # Pattern [99, 100, 99, 100] repeats, but token 99 is out of range.
        ctx = _make_ctx(
            logits,
            token_history=[99, 100, 99, 100],
            config=config,
        )
        DRYPenaltyStage()(ctx)

        # Penalty target (token 99) is out of range, so logits should be unchanged.
        np.testing.assert_array_equal(ctx.row, original)


# pytest.approx is used in test_penalty_scales_with_base
import pytest  # noqa: E402
