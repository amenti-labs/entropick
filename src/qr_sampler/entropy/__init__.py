"""Entropy source subsystem for entropick.

Re-exports the ABC, registry, and all built-in source implementations
for convenient access::

    from qr_sampler.entropy import EntropySource, EntropySourceRegistry
    from qr_sampler.entropy import SystemEntropySource, MockUniformSource
"""

from qr_sampler.entropy.base import EntropySource
from qr_sampler.entropy.fallback import FallbackEntropySource
from qr_sampler.entropy.mock import MockUniformSource
from qr_sampler.entropy.registry import EntropySourceRegistry, register_entropy_source
from qr_sampler.entropy.system import SystemEntropySource

# QuantumGrpcSource is registered by its own module but imported lazily to
# avoid a mandatory dependency on grpcio.

__all__ = [
    "EntropySource",
    "EntropySourceRegistry",
    "FallbackEntropySource",
    "MockUniformSource",
    "SystemEntropySource",
    "register_entropy_source",
]
