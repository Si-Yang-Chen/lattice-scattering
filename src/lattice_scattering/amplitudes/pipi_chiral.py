"""Two-flavor one-loop I=2 pi-pi amplitude from arXiv:1107.5023.

The amplitude uses the paper's dimensionless t convention and physical
elastic-cut logarithm branch. The observable adapter predicts the source's
dimensionless k cot(delta) / m_pi rows in their supplied order; covariance
handling and fitting are deliberately left to the separate observable fitter.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atanh, isfinite, log, pi, sqrt

import numpy as np

from .registry import register

__all__ = [
    "PipiChiralDomainError",
    "PipiNLOChiral",
    "PipiChiralPhaseObservablePredictor",
    "pipi_chiral_constants_from_ell_at_fpi",
]


class PipiChiralDomainError(ValueError):
    """An input lies outside the real branch or validity domain of the model."""


def _real_scalar(name: str, value, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or np.iscomplexobj(value) or np.ndim(value) != 0:
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite real scalar") from error
    if not isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def pipi_chiral_constants_from_ell_at_fpi(*, ell1, ell2, ell3, ell4) -> tuple[float, float, float]:
    """Convert ell_i^r(mu=f_pi) to the paper's (C1, C2, C4) once.

    The ell_i are dimensionless, renormalized two-flavor constants at the
    paper's scale mu=f_pi. Callers that already have C1/C2/C4 should pass them
    directly to PipiNLOChiral instead.
    """
    l1, l2, l3, l4 = (
        _real_scalar(name, value)
        for name, value in (("ell1", ell1), ("ell2", ell2), ("ell3", ell3), ("ell4", ell4))
    )
    c1 = -(4.0 * l1 + 4.0 * l2 + l3 - l4) / (2.0 * pi) - 1.0 / (128.0 * pi**3)
    c2 = 32.0 * pi * (12.0 * l1 + 4.0 * l2 + 7.0 * l3 - 3.0 * l4) + 31.0 / (6.0 * pi)
    c4 = (
        (212.0 * l1 + 40.0 * l2 + 123.0 * l3 - 69.0 * l4) / (5184.0 * pi**2)
        + 701.0 / (622080.0 * pi**4)
    )
    return c1, c2, c4


@register(
    "pipi-nlo-chiral",
    description="Two-flavor one-loop I=2 pi-pi NLO amplitude, with the 1107.5023 elastic branch",
    required=("pion_mass", "decay_constant", "C1", "C2", "C4"),
)
@dataclass(frozen=True)
class PipiNLOChiral:
    """The complete two-flavor one-loop I=2 pi-pi NLO amplitude.

    All masses, momenta and f_pi use one consistent natural-unit system;
    C1, C2 and C4 are the dimensionless paper constants. The source's real-axis
    phase-shift convention is defined for k >= 0. Its subthreshold logarithm
    branch is not fixed by the frozen capability spec, so evaluation below
    s_phys=4 m_pi^2 is rejected explicitly.
    """

    pion_mass: float
    decay_constant: float
    C1: float
    C2: float
    C4: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "pion_mass", _real_scalar("pion_mass", self.pion_mass, positive=True))
        object.__setattr__(
            self, "decay_constant", _real_scalar("decay_constant", self.decay_constant, positive=True)
        )
        for name in ("C1", "C2", "C4"):
            object.__setattr__(self, name, _real_scalar(name, getattr(self, name)))

    @property
    def m_pi(self) -> float:
        """Paper m_pi in the model's declared natural-unit system."""
        return self.pion_mass

    @property
    def f_pi(self) -> float:
        """Paper f_pi in the model's declared natural-unit system."""
        return self.decay_constant

    @classmethod
    def from_ell_at_fpi(
        cls, *, pion_mass, decay_constant, ell1, ell2, ell3, ell4
    ) -> "PipiNLOChiral":
        """Construct from ell_i^r(mu=f_pi) using Eq. (11) exactly once."""
        c1, c2, c4 = pipi_chiral_constants_from_ell_at_fpi(
            ell1=ell1, ell2=ell2, ell3=ell3, ell4=ell4
        )
        return cls(pion_mass, decay_constant, c1, c2, c4)

    @property
    def crossed_channel_branch_endpoint_s_phys(self) -> float:
        """Nearest crossed-channel logarithm endpoint, s_phys=0."""
        return 0.0

    @property
    def t_lo_zero_s_phys(self) -> float:
        """Exact inverse-expansion pole where t_LO=0: s_phys=2 m_pi^2."""
        return 2.0 * self.m_pi**2

    @property
    def two_pion_threshold_s_phys(self) -> float:
        """Elastic threshold and two-pion branch point, s_phys=4 m_pi^2."""
        return 4.0 * self.m_pi**2

    @property
    def four_pion_threshold_s_phys(self) -> float:
        """Four-pion inelastic boundary, s_phys=16 m_pi^2."""
        return 16.0 * self.m_pi**2

    def s_breakpoints(self) -> np.ndarray:
        """Exact real scan breakpoints in physical invariant s_phys units.

        The list includes the crossed-channel endpoint, the t_LO inverse
        pole, the elastic threshold and the four-pion finite-volume boundary.
        Consumers should split real scans at every returned value.
        """
        return np.asarray(
            [
                self.crossed_channel_branch_endpoint_s_phys,
                self.t_lo_zero_s_phys,
                self.two_pion_threshold_s_phys,
                self.four_pion_threshold_s_phys,
            ],
            dtype=float,
        )

    def t_LO(self, k) -> float:
        """Evaluate Eq. (10)'s leading-order amplitude for real k >= 0."""
        momentum = _real_scalar("k", k)
        if momentum < 0.0:
            raise PipiChiralDomainError("the source real-axis branch requires k >= 0")
        return -(self.m_pi**2 + 2.0 * momentum**2) / (8.0 * pi * self.f_pi**2)

    def _threshold_log_data(self, k: float) -> tuple[float, float, float]:
        """Return u, L_t and L_t/u without threshold 0/0 forms."""
        if k == 0.0:
            return 0.0, 0.0, -2.0
        energy = float(np.hypot(k, self.m_pi))
        u = k / energy
        if u < 0.5:
            # This form is accurate at threshold, where both logs vanish.
            lt = -2.0 * atanh(u)
            lt_over_u = -2.0 * atanh(u) / u
        else:
            # Avoid cancellation in (1-u)/(1+u) as u approaches one.
            lt = 2.0 * (log(self.m_pi) - log(energy) - log(1.0 + k / energy))
            lt_over_u = lt / u
        return u, lt, lt_over_u

    def _t_nlo_from_real_k(self, k: float) -> complex:
        """Eq. (10), evaluated with combined analytic threshold limits."""
        m2 = self.m_pi**2
        f2 = self.f_pi**2
        k2 = k * k
        u, lt, lt_over_u = self._threshold_log_data(k)
        # For the specified elastic branch, L_s = L_t + i*pi. At k=0 the
        # products below use their combined limits, never v or 1/k**2 alone.
        u_ls = u * complex(lt, pi)
        v_lt = lt_over_u
        lt2_factor = lt_over_u**2 * (u**2 + (13.0 / 12.0) * (1.0 - u**2))

        value = -m2**2 / f2**2 * (self.C1 - 31.0 / (384.0 * pi**3))
        value += (m2 * k2 / f2**2) * (
            301.0 / (1152.0 * pi**3) - self.C2 / (128.0 * pi**2) - 0.5 * self.C1
        )
        value += (k2**2 / f2**2) * (
            14.0 / (45.0 * pi**3)
            - (19.0 * self.C1 / 8.0 - 9.0 * self.C2 / (512.0 * pi**2) + 216.0 * pi * self.C4)
        )
        value -= (
            (3.0 * m2**2 / 32.0 + 5.0 * m2 * k2 / 12.0 + 5.0 * k2**2 / 9.0)
            / (4.0 * pi**3 * f2**2)
            * log(m2 / f2)
        )
        value += (
            (m2**2 / 4.0 + m2 * k2 + k2**2)
            / (16.0 * pi**3 * f2**2)
            * u_ls
        )
        value += (
            (3.0 * m2**2 / 16.0 + 7.0 * m2 * k2 / 9.0 + 11.0 * k2**2 / 18.0)
            / (8.0 * pi**3 * f2**2)
            * v_lt
        )
        value -= m2**2 / (128.0 * pi**3 * f2**2) * lt2_factor
        if not isfinite(value.real) or not isfinite(value.imag):
            raise PipiChiralDomainError("Eq. (10) produced a nonfinite amplitude")
        return complex(value)

    def t_NLO(self, k) -> complex:
        """Evaluate the complete non-LO Eq. (10) for real k >= 0.

        The elastic physical-cut convention is used: L_s=ln|...|+i*pi;
        L_t is the real logarithm printed in the frozen specification.
        """
        momentum = _real_scalar("k", k)
        if momentum < 0.0:
            raise PipiChiralDomainError("the source real-axis branch requires k >= 0")
        return self._t_nlo_from_real_k(momentum)

    def p_cot_delta(self, s_phys) -> float:
        """Return dimensionful p cot(delta) from Eq. (20), with p=k.

        The exact two-pion threshold is evaluated by its combined analytic
        limit. Positive real energies below threshold are refused because
        the frozen specification does not choose their logarithm continuation.
        """
        s_value = _real_scalar("s_phys", s_phys)
        if s_value == self.crossed_channel_branch_endpoint_s_phys:
            raise PipiChiralDomainError("s_phys=0 is the crossed-channel square-root/log branch endpoint")
        if s_value <= 0.0:
            raise PipiChiralDomainError("real-axis evaluation requires s_phys > 0")
        if s_value == self.t_lo_zero_s_phys:
            raise ZeroDivisionError("t_LO=0 at s_phys=2 m_pi^2; Eq. (20) has an inverse-expansion pole")
        if s_value < self.two_pion_threshold_s_phys:
            raise PipiChiralDomainError(
                "subthreshold logarithm continuation is not specified; the approved real-axis branch requires k >= 0"
            )

        k2 = s_value / 4.0 - self.m_pi**2
        # The equality branch prevents roundoff in s/4 from perturbing the
        # explicitly finite threshold limit.
        k = 0.0 if s_value == self.two_pion_threshold_s_phys else sqrt(k2)
        t_lo = self.t_LO(k)
        if t_lo == 0.0:
            raise ZeroDivisionError("t_LO=0; Eq. (20) has an inverse-expansion pole")
        t_nlo = self._t_nlo_from_real_k(k)
        root_factor = sqrt(s_value) / (2.0 * self.m_pi)
        rhs = root_factor * (1.0 / t_lo - t_nlo / (t_lo * t_lo)) + 1j * k / self.m_pi

        if k > 0.0:
            u, _, _ = self._threshold_log_data(k)
            expected_imaginary = u * t_lo**2
            imaginary_scale = max(1.0, abs(expected_imaginary), abs(t_nlo.imag))
            if abs(t_nlo.imag - expected_imaginary) > 2e-12 * imaginary_scale:
                raise ArithmeticError("Eq. (10) imaginary part failed the elastic unitarity check")
            rhs_scale = max(1.0, abs(rhs.real), abs(root_factor * t_nlo / (t_lo * t_lo)), abs(k / self.m_pi))
            if abs(rhs.imag) > 2e-12 * rhs_scale:
                raise ArithmeticError("Eq. (20) failed to cancel the elastic unitarity imaginary part")

        result = self.m_pi * rhs.real
        if not isfinite(result):
            raise PipiChiralDomainError("Eq. (20) produced nonfinite p cot(delta)")
        return result

    def dimensionless_k_cot_delta(self, s_phys) -> float:
        """Return the fitted source observable k cot(delta) / m_pi."""
        return self.p_cot_delta(s_phys) / self.m_pi

    def validate_elastic_finite_volume_domain(self, s_phys) -> float:
        """Validate a real two-body finite-volume energy and return s_phys.

        This adapter is elastic-only: 4 m_pi^2 <= s_phys < 16 m_pi^2.
        In particular, the four-pion threshold is excluded because a two-body
        finite-volume condition is no longer justified there.
        """
        s_value = _real_scalar("s_phys", s_phys)
        if s_value == self.crossed_channel_branch_endpoint_s_phys:
            raise PipiChiralDomainError("s_phys=0 is the crossed-channel square-root/log branch endpoint")
        if s_value < 0.0:
            raise PipiChiralDomainError("real-axis evaluation requires s_phys > 0")
        if s_value == self.t_lo_zero_s_phys:
            raise ZeroDivisionError("t_LO=0 at s_phys=2 m_pi^2; inverse-amplitude pole")
        if s_value < self.two_pion_threshold_s_phys:
            raise PipiChiralDomainError("the elastic finite-volume domain starts at s_phys=4 m_pi^2")
        if s_value >= self.four_pion_threshold_s_phys:
            raise PipiChiralDomainError(
                "two-body finite-volume evaluation is restricted below the four-pion threshold s_phys=16 m_pi^2"
            )
        return s_value

    def inverse(self, s_phys) -> np.ndarray:
        """Return the registered S-wave protocol inverse 2 p cot(delta)/sqrt(s_phys)."""
        s_value = self.validate_elastic_finite_volume_domain(s_phys)
        value = 2.0 * self.p_cot_delta(s_value) / sqrt(s_value)
        if not isfinite(value):
            raise PipiChiralDomainError("protocol inverse is nonfinite")
        return np.asarray([[value]], dtype=float)

    def k(self, s_phys) -> np.ndarray:
        """Return the scalar protocol K matrix, refusing a zero inverse pole."""
        inverse = float(self.inverse(s_phys)[0, 0])
        if inverse == 0.0:
            raise ZeroDivisionError("zero reduced inverse has no finite K value")
        return np.asarray([[1.0 / inverse]], dtype=float)


@dataclass(frozen=True)
class PipiChiralPhaseObservablePredictor:
    """Ordered direct-observable adapter for k cot(delta) / m_pi rows.

    A later correlated fitter can build one adapter per parameter vector and
    call predict on the dataset's ordered s_phys coordinates. No sorting, row
    matching, covariance construction or fitting is performed.
    """

    amplitude: PipiNLOChiral

    def __post_init__(self) -> None:
        if not isinstance(self.amplitude, PipiNLOChiral):
            raise TypeError("amplitude must be a PipiNLOChiral instance")

    def predict(self, s_phys_rows) -> np.ndarray:
        """Predict source-normalized observables in exactly the input row order."""
        if np.iscomplexobj(s_phys_rows):
            raise ValueError("s_phys_rows must be real")
        rows = np.asarray(s_phys_rows, dtype=float)
        if rows.ndim != 1 or not rows.size or not np.all(np.isfinite(rows)):
            raise ValueError("s_phys_rows must be a finite nonempty real vector")
        values = []
        for s_phys in rows:
            coordinate = self.amplitude.validate_elastic_finite_volume_domain(s_phys)
            values.append(self.amplitude.dimensionless_k_cot_delta(coordinate))
        return np.asarray(values, dtype=float)

    __call__ = predict
