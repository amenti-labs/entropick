"""Entropy injection methods for qr-sampler.

Three independent methods that reshape the probability distribution at
different stages of the sampling pipeline. All three operate within the
CDF-selection pipeline and always produce one-hot logit output.

LogitPerturbation: direct per-logit quantum noise before temperature scaling.
TemperatureModulation: per-token temperature modulation via quantum entropy.
SelectionDrift: temporal correlation across tokens via a per-request drift.
"""

from qr_sampler.injection.logit_perturbation import LogitPerturbation
from qr_sampler.injection.selection_drift import SelectionDrift
from qr_sampler.injection.temp_modulation import TemperatureModulation

__all__ = ["LogitPerturbation", "SelectionDrift", "TemperatureModulation"]
