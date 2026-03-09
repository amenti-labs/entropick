"""
title: QR-Sampler Parameters
author: entropick
author_url: https://github.com/your-org/entropick
version: 0.1.0
license: MIT
description: >
  Injects entropick per-request parameters into vLLM requests. Configure
  sampling parameters via Valves to control external-entropy-driven token
  selection.
"""

from pydantic import BaseModel, Field


class Filter:
    """Open WebUI filter that injects qr_* parameters into vLLM requests.

    When entropick is loaded as a vLLM logits processor, it reads per-request
    overrides from extra fields in the request body (prefixed with ``qr_``).
    This filter adds those fields via the ``inlet()`` hook so that Open WebUI
    users can control the common per-request sampling parameters through the
    Valves UI without manually editing API calls.

    Parameter flow::

        Open WebUI chat -> inlet() adds qr_* keys -> vLLM /v1/chat/completions
        -> SamplingParams.extra_args -> entropick resolve_config()

    Toggle ``enable_qr_sampling`` to False to disable parameter injection
    entirely (requests pass through unmodified).
    """

    class Valves(BaseModel):
        """Admin-configurable entropick parameters.

        Each field maps to a ``qr_*`` key that entropick's
        ``resolve_config()`` accepts. This filter intentionally exposes only
        the most useful per-request knobs. Infrastructure settings (gRPC
        address, fallback mode, logging, diagnostic capture, etc.) stay
        process-wide and are configured via environment variables on the
        vLLM container.
        """

        # --- Filter control ---
        priority: int = Field(
            default=0,
            description="Filter execution priority (lower runs first).",
        )
        enable_qr_sampling: bool = Field(
            default=True,
            description=(
                "Master switch. When False, no qr_* parameters are injected "
                "and requests pass through unmodified."
            ),
        )

        # --- Token selection ---
        top_k: int = Field(
            default=0,
            description="Top-k filtering: keep only the k most probable tokens (0 disables).",
        )
        top_p: float = Field(
            default=1.0,
            description="Nucleus sampling threshold (1.0 disables).",
        )

        # --- Temperature ---
        temperature_strategy: str = Field(
            default="fixed",
            description="Temperature strategy: 'fixed' or 'edt' (entropy-dependent).",
        )
        fixed_temperature: float = Field(
            default=0.7,
            description="Constant temperature when strategy is 'fixed'.",
        )
        edt_base_temp: float = Field(
            default=0.8,
            description="Base coefficient for EDT strategy.",
        )
        edt_exponent: float = Field(
            default=0.5,
            description="Power-law exponent for EDT strategy.",
        )
        edt_min_temp: float = Field(
            default=0.1,
            description="EDT temperature floor.",
        )
        edt_max_temp: float = Field(
            default=2.0,
            description="EDT temperature ceiling.",
        )

        # --- Signal amplification ---
        signal_amplifier_type: str = Field(
            default="zscore_mean",
            description="Signal amplification algorithm.",
        )
        sample_count: int = Field(
            default=20480,
            description="Number of entropy bytes to fetch per token.",
        )

    def __init__(self) -> None:
        self.valves = self.Valves()

    # Fields that are part of entropick config (everything except filter-
    # internal fields like ``priority`` and ``enable_qr_sampling``).
    _QR_FIELDS: frozenset = frozenset(
        {
            "signal_amplifier_type",
            "sample_count",
            "temperature_strategy",
            "fixed_temperature",
            "edt_base_temp",
            "edt_exponent",
            "edt_min_temp",
            "edt_max_temp",
            "top_k",
            "top_p",
        }
    )

    async def inlet(
        self,
        body: dict,
        __user__: dict | None = None,
    ) -> dict:
        """Inject qr_* parameters into the request body before it reaches vLLM.

        Each Valve value is added as a top-level ``qr_<field>`` key.  vLLM
        maps unknown top-level keys to ``SamplingParams.extra_args``, and
        entropick's ``resolve_config()`` picks them up from there.

        Args:
            body: The request body (messages, model, stream, etc.).
            __user__: Optional user information from Open WebUI.

        Returns:
            The request body with qr_* keys injected (or unmodified if
            sampling is disabled).
        """
        if not self.valves.enable_qr_sampling:
            return body

        valve_dict = self.valves.model_dump()
        for field_name in self._QR_FIELDS:
            body[f"qr_{field_name}"] = valve_dict[field_name]

        return body

    async def outlet(
        self,
        body: dict,
        __user__: dict | None = None,
    ) -> dict:
        """Pass-through — no post-processing needed.

        Args:
            body: The response body from vLLM.
            __user__: Optional user information from Open WebUI.

        Returns:
            The response body unmodified.
        """
        return body
