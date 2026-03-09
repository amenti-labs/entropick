"""DRYPenaltyStage -- Don't Repeat Yourself n-gram repetition penalty."""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING

from qr_sampler.pipeline.registry import StageRegistry

if TYPE_CHECKING:
    from qr_sampler.pipeline.context import SamplingContext

_logger = logging.getLogger("qr_sampler")

_HISTORY_IDS_KEY = "history_ids"


@StageRegistry.register("dry")
class DRYPenaltyStage:
    """Penalize tokens that would extend repeated n-grams in the output.

    Scans the token history for n-gram suffixes matching the most recent
    tokens.  For each potential next token that would continue a repeated
    pattern of length >= ``dry_allowed_length``, applies an exponential
    penalty: ``dry_multiplier * dry_base ^ (match_length - dry_allowed_length)``.

    Operates on logits: subtracts the penalty from logit values.

    Pipeline position: after logit_perturbation, before temperature.

    No-ops when ``config.dry_multiplier <= 0`` or token history is empty.
    """

    name: str = "dry"

    def __call__(self, ctx: SamplingContext) -> None:
        if ctx.config.dry_multiplier <= 0.0:
            return

        # --- Get token history ---
        history: list[int] = ctx.stage_state.get(_HISTORY_IDS_KEY, [])
        if not history:
            return

        multiplier = ctx.config.dry_multiplier
        base = ctx.config.dry_base
        allowed_length = ctx.config.dry_allowed_length
        penalty_last_n = ctx.config.dry_penalty_last_n

        # --- Determine lookback window ---
        if penalty_last_n < 0:
            # -1 means full context.
            window = history
        elif penalty_last_n == 0:
            return
        else:
            window = history[-penalty_last_n:]

        if not window:
            return

        # --- Parse sequence breakers ---
        breaker_tokens = self._parse_breakers(ctx.config.dry_sequence_breakers)

        # --- Find the last breaker position in the window ---
        # Only look at the window for matching (after the last breaker).
        effective_start = 0
        for idx in range(len(window) - 1, -1, -1):
            if window[idx] in breaker_tokens:
                effective_start = idx + 1
                break

        effective_window = window[effective_start:]
        if not effective_window:
            return

        # --- Find longest n-gram matches ---
        # For each suffix of effective_window (from the end), find matching
        # prefixes earlier in the window to determine what token would
        # extend the repetition.
        penalties: dict[int, float] = {}
        n = len(effective_window)

        # The current suffix is effective_window itself. We look for matches
        # of the tail of effective_window in earlier positions.
        for suffix_len in range(1, n):
            # The suffix is the last `suffix_len` tokens.
            suffix = effective_window[n - suffix_len :]

            # Search for this suffix earlier in the window.
            for start in range(n - suffix_len):
                # Check if effective_window[start:start+suffix_len] == suffix.
                match = True
                for k in range(suffix_len):
                    if effective_window[start + k] != suffix[k]:
                        match = False
                        break

                if match:
                    # The token that followed this match in history would
                    # continue the repetition.
                    follow_idx = start + suffix_len
                    if follow_idx < n:
                        next_token = effective_window[follow_idx]
                        match_length = suffix_len + 1  # +1 for the follow token

                        if match_length >= allowed_length:
                            penalty = multiplier * (base ** (match_length - allowed_length))
                            # Keep the maximum penalty per token.
                            if next_token not in penalties or penalty > penalties[next_token]:
                                penalties[next_token] = penalty

        if not penalties:
            return

        # --- Apply penalties to logits ---
        for token_id, penalty in penalties.items():
            if 0 <= token_id < len(ctx.row):
                ctx.row[token_id] -= penalty

        if ctx.config.injection_verbose and penalties:
            _logger.debug(
                "DRY: penalized %d tokens (max_penalty=%.3f, window=%d)",
                len(penalties),
                max(penalties.values()),
                len(effective_window),
            )

    @staticmethod
    def _parse_breakers(breakers_str: str) -> set[int]:
        """Parse comma-separated sequence breaker string into token IDs.

        For simplicity, treats each comma-separated value as a potential
        integer token ID. Non-integer entries are silently skipped.

        Args:
            breakers_str: Comma-separated string of breaker tokens.

        Returns:
            Set of integer token IDs that act as sequence breakers.
        """
        result: set[int] = set()
        if not breakers_str:
            return result
        for part in breakers_str.split(","):
            stripped = part.strip()
            with contextlib.suppress(ValueError):
                result.add(int(stripped))
        return result
