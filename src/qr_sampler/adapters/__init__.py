"""Framework adapters for qr-sampler.

Thin wrappers that integrate the qr-sampler pipeline with different
inference frameworks. Each adapter converts framework-specific input/output
formats while delegating all sampling logic to the shared pipeline.

Available adapters:
    - ``transformers``: Hugging Face Transformers ``LogitsProcessor``
    - ``llamacpp``: llama-cpp-python custom sampler callback
    - ``sglang``: SGLang custom logit processor
"""

from __future__ import annotations

__all__ = [
    "QRSamplerCallback",
    "QRSamplerCustomLogitProcessor",
    "QRSamplerLogitsProcessorHF",
]


def __getattr__(name: str) -> type:
    """Lazy-import adapter classes to avoid framework dependencies at import time.

    Args:
        name: Attribute name to look up.

    Returns:
        The requested adapter class.

    Raises:
        AttributeError: If the name is not a known adapter class.
    """
    if name == "QRSamplerLogitsProcessorHF":
        from qr_sampler.adapters.transformers import QRSamplerLogitsProcessorHF

        return QRSamplerLogitsProcessorHF
    if name == "QRSamplerCallback":
        from qr_sampler.adapters.llamacpp import QRSamplerCallback

        return QRSamplerCallback
    if name == "QRSamplerCustomLogitProcessor":
        from qr_sampler.adapters.sglang import QRSamplerCustomLogitProcessor

        return QRSamplerCustomLogitProcessor
    raise AttributeError(f"module 'qr_sampler.adapters' has no attribute {name!r}")
