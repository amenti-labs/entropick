"""Entropy injection methods for qr-sampler.

Three independent methods that reshape the probability distribution at
different stages of the sampling pipeline. All three operate within the
CDF-selection pipeline and always produce one-hot logit output.

M1 (LogitNoise): direct per-logit quantum noise before temperature scaling.
M2 (TempVariance): per-token temperature modulation via quantum entropy.
M3 (CorrelatedWalk): temporal correlation across tokens via a per-request walk.
"""

from qr_sampler.injection.correlated_walk import CorrelatedWalk
from qr_sampler.injection.logit_noise import LogitNoise
from qr_sampler.injection.temp_variance import TempVariance

__all__ = ["CorrelatedWalk", "LogitNoise", "TempVariance"]
