"""Single-channel inverse-K and reduced effective-range models.

The models in this module keep the inverse form as the primary representation.  This matters
at a K pole: ``K^{-1}`` can be perfectly regular there even though constructing K itself is
not.  All polynomial coefficient sequences use ascending powers.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np

from .models import _scalar
from .phase_space import cm_real_axis
from .registry import register

__all__ = ["RationalInverseK", "ReducedERE", "UnitaryChPTLO"]


def _coefficients(values, name: str) -> np.ndarray:
    """A non-empty, finite, real coefficient sequence stored read-only."""
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a non-empty finite real sequence")
    try:
        items = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a non-empty finite real sequence") from exc
    if not items:
        raise ValueError(f"{name} must be a non-empty finite real sequence")
    if any(
        isinstance(item, (bool, np.bool_))
        or np.iscomplexobj(item)
        or np.ndim(item) != 0
        for item in items
    ):
        raise ValueError(f"{name} must be a non-empty finite real sequence")
    try:
        value = np.asarray(items, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a non-empty finite real sequence") from exc
    if value.ndim != 1 or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a non-empty finite real sequence")
    value.setflags(write=False)
    return value


def _s_scalar(value, name: str = "s") -> tuple[complex, bool]:
    """Validate a finite scalar ``s`` and report whether its input was real."""
    if (
        isinstance(value, (bool, np.bool_))
        or isinstance(value, (str, bytes))
        or np.ndim(value) != 0
    ):
        raise ValueError(f"{name} must be a finite real or complex scalar")
    try:
        result = complex(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real or complex scalar") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real or complex scalar")
    real_input = not np.iscomplexobj(value) and result.imag == 0.0
    return result, real_input


def _polyval(coefficients: np.ndarray, value, name: str) -> complex | float:
    """Evaluate an ascending-power polynomial, refusing overflow and NaN."""
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        result = np.polynomial.polynomial.polyval(value, coefficients)
    if not np.isfinite(result):
        raise ValueError(f"{name} polynomial evaluation is non-finite")
    return result


def _polynomial_zero(value, x, coefficients: np.ndarray) -> bool:
    """Use a scale-aware roundoff check for polynomial roots evaluated numerically."""
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        bound = np.polynomial.polynomial.polyval(abs(x), np.abs(coefficients))
    magnitude = abs(value)
    if bound == 0.0:
        return magnitude == 0.0
    if not np.isfinite(bound):
        return magnitude == 0.0
    return magnitude <= 16.0 * np.finfo(float).eps * float(bound)


def _real_polynomial_roots(coefficients: np.ndarray) -> tuple[float, ...]:
    """Return finite real roots of an ascending-power polynomial."""
    end = len(coefficients)
    while end > 1 and coefficients[end - 1] == 0.0:
        end -= 1
    trimmed = coefficients[:end]
    if len(trimmed) == 1:
        return ()
    scale = float(np.max(np.abs(trimmed)))
    if scale == 0.0:
        return ()
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        roots = np.polynomial.polynomial.polyroots(trimmed / scale)
    candidates: list[float] = []
    for root in roots:
        real = float(np.real(root))
        imag = float(np.imag(root))
        if not np.isfinite(real) or not np.isfinite(imag):
            continue
        if abs(imag) > 1e-10 * max(1.0, abs(real)):
            continue
        value = _polyval(trimmed / scale, real, "denominator")
        if _polynomial_zero(value, real, trimmed / scale):
            candidates.append(real)
    candidates.sort()
    distinct: list[float] = []
    for value in candidates:
        if not distinct or abs(value - distinct[-1]) > 1e-10 * max(1.0, abs(value)):
            distinct.append(value)
    return tuple(distinct)


def _p_squared(s, mass1: float, mass2: float):
    """Relativistic two-body momentum squared with real subthreshold continuation."""
    return (s - (mass1 + mass2) ** 2) * (s - (mass1 - mass2) ** 2) / (4.0 * s)


@register(
    "rational-inverse-k",
    description="Single-channel rational inverse K: K^-1(s)=N(x)/D(x)",
    required=("numerator_coefficients", "denominator_coefficients"),
)
class RationalInverseK:
    r"""``K^{-1}(s)=N(x)/D(x)``, with ``x=(s-s_ref)/scale_squared``.

    Coefficients are in ascending powers.  The inverse is evaluated directly, so a zero of
    ``N`` is a regular zero of ``K^{-1}`` (a pole of K).  A zero of ``D`` is an inverse
    singularity and is refused.  ``s_breakpoints`` reports only the latter.
    """

    def __init__(
        self,
        *,
        numerator_coefficients,
        denominator_coefficients=(1.0,),
        s_ref: float = 0.0,
        scale_squared: float = 1.0,
    ):
        numerator = _coefficients(numerator_coefficients, "numerator_coefficients")
        denominator = _coefficients(denominator_coefficients, "denominator_coefficients")
        if not np.any(denominator != 0.0):
            raise ValueError("denominator_coefficients must not be identically zero")
        self.numerator_coefficients = numerator
        self.denominator_coefficients = denominator
        self.s_ref = _scalar(s_ref, "s_ref")
        self.scale_squared = _scalar(scale_squared, "scale_squared", positive=True)

    def _parts(self, s):
        value, real_input = _s_scalar(s)
        s_eval = value.real if real_input else value
        with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
            x = (s_eval - self.s_ref) / self.scale_squared
        if not np.isfinite(x):
            raise ValueError("dimensionless s variable is non-finite")
        numerator = _polyval(self.numerator_coefficients, x, "numerator")
        denominator = _polyval(self.denominator_coefficients, x, "denominator")
        if _polynomial_zero(denominator, x, self.denominator_coefficients):
            raise ValueError(f"inverse K denominator D(x) vanishes at s={value!r}")
        return numerator, denominator, x, real_input

    def inverse(self, s) -> np.ndarray:
        """Evaluate ``K^-1`` directly, including regular ``N(x)=0`` points."""
        numerator, denominator, _, real_input = self._parts(s)
        result = numerator / denominator
        if not np.isfinite(result):
            raise ValueError("inverse K evaluation is non-finite")
        if real_input:
            return np.asarray([[float(np.real(result))]])
        return np.asarray([[complex(result)]], dtype=complex)

    def k(self, s) -> np.ndarray:
        """Return K where finite; refuse its poles without affecting ``inverse``."""
        numerator, denominator, x, real_input = self._parts(s)
        if _polynomial_zero(numerator, x, self.numerator_coefficients):
            raise ValueError("K has a pole where the inverse numerator N(x) vanishes")
        result = denominator / numerator
        if not np.isfinite(result):
            raise ValueError("K evaluation is non-finite")
        if real_input:
            return np.asarray([[float(np.real(result))]])
        return np.asarray([[complex(result)]], dtype=complex)

    def s_breakpoints(self) -> np.ndarray:
        """Real roots of D mapped back to s; numerator roots are not breakpoints."""
        values = [self.s_ref + self.scale_squared * root
                  for root in _real_polynomial_roots(self.denominator_coefficients)]
        return np.asarray(sorted(value for value in values if np.isfinite(value)), dtype=float)


@register(
    "reduced-ere",
    description="Reduced effective-range model R_ell=k^(2ell+1) cot(delta_ell)",
    required=("mass1", "mass2", "ell", "coefficients"),
)
class ReducedERE:
    r"""Polynomial reduced ERE, ``R_ell(s)=sum_n c_n [k^2(s)]^n``.

    ``reduced(s)`` is the native JLS ``weighting='scale'`` callback and remains finite at
    threshold.  ``inverse(s)`` supplies the hatted inverse ``(2/E) R_ell/(k^2)^ell`` where
    that view is defined.
    """

    def __init__(self, *, mass1: float, mass2: float, ell: int, coefficients):
        self.mass1 = _scalar(mass1, "mass1", positive=True)
        self.mass2 = _scalar(mass2, "mass2", positive=True)
        if isinstance(ell, (bool, np.bool_)) or not isinstance(ell, Integral) or int(ell) < 0:
            raise ValueError("ell must be an integral nonnegative value")
        self.ell = int(ell)
        self.coefficients = _coefficients(coefficients, "coefficients")

    def k_squared(self, s):
        value, real_input = _s_scalar(s)
        s_eval = value.real if real_input else value
        if s_eval == 0:
            raise ValueError("s=0 is outside the two-body momentum formula")
        result = _p_squared(s_eval, self.mass1, self.mass2)
        if not np.isfinite(result):
            raise ValueError("k^2 evaluation is non-finite")
        return float(np.real(result)) if real_input else complex(result)

    def reduced(self, s):
        """Return the finite reduced inverse ``R_ell`` including at threshold."""
        k2 = self.k_squared(s)
        value = _polyval(self.coefficients, k2, "reduced ERE")
        return float(np.real(value)) if not np.iscomplexobj(k2) else complex(value)

    def jls_scale(self, s):
        """Alias documenting the native scale-weighted JLS callback convention."""
        return self.reduced(s)

    def inverse(self, s) -> np.ndarray:
        """Return ``Khat^-1=(2/E)R_ell/(k^2)^ell`` where finite."""
        value, real_input = _s_scalar(s)
        s_eval = value.real if real_input else value
        if s_eval == 0 or (real_input and s_eval < 0):
            raise ValueError("a positive s is required for the hatted inverse")
        k2 = self.k_squared(s)
        reduced = self.reduced(s)
        if self.ell == 0:
            quotient = reduced
        elif k2 == 0:
            if np.any(self.coefficients[: self.ell] != 0.0):
                raise ValueError(
                    "hatted inverse is singular at k^2=0; use the finite reduced ERE view"
                )
            quotient = (
                _polyval(self.coefficients[self.ell :], 0.0, "reduced ERE")
                if len(self.coefficients) > self.ell
                else 0.0
            )
        else:
            quotient = reduced / (k2 ** self.ell)
        energy = np.sqrt(s_eval)
        result = 2.0 * quotient / energy
        if not np.isfinite(result):
            raise ValueError("hatted inverse evaluation is non-finite")
        if real_input:
            return np.asarray([[float(np.real(result))]])
        return np.asarray([[complex(result)]], dtype=complex)

    def k(self, s) -> np.ndarray:
        value = self.inverse(s)[0, 0]
        if value == 0:
            raise ValueError("K has a pole where the hatted inverse vanishes")
        return np.asarray([[1.0 / value]])

    def s_breakpoints(self) -> np.ndarray:
        """Thresholds where the hatted inverse has a genuine centrifugal singularity."""
        if self.ell == 0 or not np.any(self.coefficients[: self.ell] != 0.0):
            return np.zeros(0, dtype=float)
        return np.unique(
            np.asarray(
                [(self.mass1 + self.mass2) ** 2, (self.mass1 - self.mass2) ** 2],
                dtype=float,
            )
        )


@register(
    "unitarized-chpt-lo",
    description="D-pion LO unitarized-ChPT inverse K with threshold-subtracted CM amplitude",
    required=("pion_mass", "heavy_mass", "decay_constant", "alpha", "mu"),
)
class UnitaryChPTLO:
    r"""D-pion LO unitarized-ChPT model from arXiv HTML Eqs. (11)-(12) of
    ``2102.04973v2``.

    The inverse K is constructed as a :class:`RationalInverseK` with ``s_ref=0`` and
    ``scale_squared=1``.  ``t_inverse`` adds the threshold-subtracted Chew-Mandelstam loop,
    with ``I((m_D+m_pi)^2)=0``.
    """

    def __init__(
        self,
        *,
        pion_mass: float,
        heavy_mass: float,
        decay_constant: float,
        alpha: float,
        mu: float,
    ):
        self.pion_mass = _scalar(pion_mass, "pion_mass", positive=True)
        self.heavy_mass = _scalar(heavy_mass, "heavy_mass", positive=True)
        self.decay_constant = _scalar(decay_constant, "decay_constant", positive=True)
        self.alpha = _scalar(alpha, "alpha")
        self.mu = _scalar(mu, "mu", positive=True)

        mpi = self.pion_mass
        md = self.heavy_mass
        p0 = -(md**2 - mpi**2) ** 2
        p1 = -2.0 * (md**2 + mpi**2)
        p2 = 3.0
        subtraction_constant = self.alpha / np.pi + (2.0 / np.pi) * (
            md / (mpi + md) * np.log(md / mpi) + np.log(mpi / self.mu)
        )
        numerator = (
            subtraction_constant * p0,
            64.0 * np.pi * self.decay_constant**2 + subtraction_constant * p1,
            subtraction_constant * p2,
        )
        self.rational_model = RationalInverseK(
            numerator_coefficients=numerator,
            denominator_coefficients=(p0, p1, p2),
            s_ref=0.0,
            scale_squared=1.0,
        )

    def inverse(self, s) -> np.ndarray:
        """The direct rational ``K^-1(s)``; source point ``s=0`` is excluded."""
        value, _ = _s_scalar(s)
        if value == 0:
            raise ValueError("the LO ChPT expression is undefined at s=0")
        return self.rational_model.inverse(s)

    def k(self, s) -> np.ndarray:
        value, _ = _s_scalar(s)
        if value == 0:
            raise ValueError("the LO ChPT expression is undefined at s=0")
        return self.rational_model.k(s)

    def s_breakpoints(self) -> np.ndarray:
        """Real denominator roots of the rational inverse K."""
        return self.rational_model.s_breakpoints()

    def t_inverse(self, s) -> np.ndarray:
        """``K^-1+I(s)`` on the real axis with the source threshold subtraction ``I=0``."""
        value, real_input = _s_scalar(s)
        if not real_input:
            raise ValueError("t_inverse is defined on the real axis")
        inverse = self.inverse(value.real)
        loop = cm_real_axis(
            value.real,
            self.pion_mass,
            self.heavy_mass,
            subtraction=0.0,
        )
        return inverse + np.asarray([[loop]], dtype=complex)
