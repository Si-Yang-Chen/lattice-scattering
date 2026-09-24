"""v2 amplitude layer: real K-matrix models, the registry and the phase space.

New parameterisations go in :mod:`models_extra` (or any module) and are made reachable by
name with :func:`registry.register`; :func:`registry.registered` lists what exists.
"""

from __future__ import annotations

from .left_hand_cut import EqualMassExchangeDomain, LeftHandCutReport, LeftHandCutDomainError
from .complex_plane import Pole, amplitude, amplitude_inverse, solve_poles
from .models import ConstantK, PolePolynomialK
from .models_extra import (
    BuggK,
    ChiralEREPcotdelta,
    ChiralPoleK,
    ConformalMapK,
    PolygonPcotdelta,
)
from .pipi_chiral import (
    PipiChiralDomainError,
    PipiChiralPhaseObservablePredictor,
    PipiNLOChiral,
    pipi_chiral_constants_from_ell_at_fpi,
)
from .partial_wave_adapters import PartialWaveJMatrixAdapter
from .inverse_models import RationalInverseK, ReducedERE, UnitaryChPTLO
from .phase_space import cm_complex, cm_real_axis, effective_inverse_k, physical_rho
from .resonance_models import (
    ChungBW,
    ChungK,
    HattedKAdapter,
    HattedKJLSAdapter,
    P33BW,
    P33K,
)
from .registry import UnknownModel, build, register, registered, specification

__all__ = [
    "EqualMassExchangeDomain",
    "LeftHandCutReport",
    "LeftHandCutDomainError",
    "ConstantK",
    "PolePolynomialK",
    "BuggK",
    "ChiralEREPcotdelta",
    "ChiralPoleK",
    "ConformalMapK",
    "PolygonPcotdelta",
    "PipiChiralDomainError",
    "PipiNLOChiral",
    "PipiChiralPhaseObservablePredictor",
    "pipi_chiral_constants_from_ell_at_fpi",
    "PartialWaveJMatrixAdapter",
    "RationalInverseK",
    "ReducedERE",
    "UnitaryChPTLO",
    "ChungBW",
    "ChungK",
    "P33BW",
    "P33K",
    "HattedKJLSAdapter",
    "HattedKAdapter",
    "UnknownModel",
    "build",
    "register",
    "registered",
    "specification",
    "Pole",
    "amplitude",
    "amplitude_inverse",
    "solve_poles",
    "cm_complex",
    "cm_real_axis",
    "effective_inverse_k",
    "physical_rho",
]
