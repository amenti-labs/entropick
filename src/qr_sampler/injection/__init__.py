"""Entropy injection methods for qr-sampler.

Three independent methods that reshape the probability distribution at
different stages of the sampling pipeline.
"""

from qr_sampler.injection.correlated_walk import CorrelatedWalk
from qr_sampler.injection.logit_noise import LogitNoise
from qr_sampler.injection.temp_variance import TempVariance

__all__ = ["CorrelatedWalk", "LogitNoise", "TempVariance"]
