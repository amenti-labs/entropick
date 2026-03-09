"""Signal amplification subsystem for entropick.

Converts raw entropy bytes into a uniform float u in (eps, 1-eps) that drives
CDF-based token selection. The amplification preserves even tiny biases
in the entropy source.
"""

from qr_sampler.amplification.base import AmplificationResult, SignalAmplifier
from qr_sampler.amplification.calibration import calibrate_population_stats, measure_entropy_rate
from qr_sampler.amplification.ecdf import ECDFAmplifier
from qr_sampler.amplification.registry import AmplifierRegistry
from qr_sampler.amplification.zscore import ZScoreMeanAmplifier

__all__ = [
    "AmplificationResult",
    "AmplifierRegistry",
    "ECDFAmplifier",
    "SignalAmplifier",
    "ZScoreMeanAmplifier",
    "calibrate_population_stats",
    "measure_entropy_rate",
]
