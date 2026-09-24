"""Real-axis two-body kinematics, in temporal lattice units.

Definitions follow arXiv:1401.3312v4 Eq.24. No legacy code imports.
Negative k_squared is retained below threshold; complex sheets are not defined here.
"""

from dataclasses import dataclass
from math import isfinite, pi, sqrt
from numbers import Integral


def _positive(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite real")
    value = float(value)
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite real")
    return value


@dataclass(frozen=True)
class LatticeFrame:
    """Spatial extent in sites, xi=a_s/a_t, integer total momentum d."""

    spatial_sites: int
    anisotropy: float
    d: tuple[int, int, int] = (0, 0, 0)

    def __post_init__(self):
        if isinstance(self.spatial_sites, bool) or not isinstance(self.spatial_sites, Integral) or self.spatial_sites <= 0:
            raise ValueError("spatial_sites must be a positive integer")
        object.__setattr__(self, "anisotropy", _positive(self.anisotropy, "anisotropy"))
        d = tuple(self.d)
        if len(d) != 3 or any(isinstance(x, bool) or not isinstance(x, Integral) for x in d):
            raise ValueError("d must contain three integers")
        object.__setattr__(self, "d", d)

    @property
    def length_at(self):
        """L/a_t = (L/a_s) xi."""
        return self.spatial_sites * self.anisotropy

    @property
    def momentum_at(self):
        return tuple(2 * pi * x / self.length_at for x in self.d)


@dataclass(frozen=True)
class TwoBodyPoint:
    s_at2: float
    gamma: float
    k_squared_at2: float
    q_squared: float
    alpha: float


def two_body_point(*, energy_lab_at, mass1_at, mass2_at, frame):
    """Inputs are explicitly a_t E and a_t m; no unit inference/conversion."""
    energy = _positive(energy_lab_at, "energy_lab_at")
    m1 = _positive(mass1_at, "mass1_at")
    m2 = _positive(mass2_at, "mass2_at")
    s = energy**2 - sum(p*p for p in frame.momentum_at)
    if not isfinite(s) or s <= 0:
        raise ValueError("total four-momentum must be timelike")
    k2 = (s - (m1+m2)**2) * (s - (m1-m2)**2) / (4*s)
    return TwoBodyPoint(s, energy/sqrt(s), k2,
                       k2*(frame.length_at/(2*pi))**2,
                       (1+(m1*m1-m2*m2)/s)/2)


def shifted_vector(n, *, frame, point):
    """gamma_hat^-1(n-alpha*d), with compression parallel to d only."""
    n = tuple(float(x) for x in n)
    if len(n) != 3 or not all(isfinite(x) for x in n):
        raise ValueError("n must contain three finite real components")
    x = tuple(v-point.alpha*d for v, d in zip(n, frame.d))
    d2 = sum(d*d for d in frame.d)
    if d2 == 0:
        return x
    parallel = sum(v*d for v, d in zip(x, frame.d))/d2
    return tuple(v + (1/point.gamma-1)*parallel*d for v, d in zip(x, frame.d))
