"""Elastic resonance models and the hatted-``K`` to JLS bridge.

The two resonance forms in this module return the paper convention

``Khat = tan(delta) / rho``

through the existing amplitude protocol: :meth:`k` returns ``Khat`` and
:meth:`inverse` returns ``Khat**-1 = rho*cot(delta)``.  The finite-volume JLS
carrier uses a different reduced inverse.  :class:`HattedKJLSAdapter` applies
the approved scale-weighted conversion

``R_scale = (sqrt(s)/2) * k(s)**(2*ell) * Khat**-1``.

The models keep the algebraic threshold cancellation available to the adapter
through ``k2_power_inverse``.  This matters for P waves: evaluating
``k2 * inverse`` as two floating point factors at threshold would manufacture
an avoidable ``0 * infinity`` failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np

from .models import _scalar
from .registry import register

__all__ = [
    "ChungBW",
    "ChungK",
    "P33BW",
    "P33K",
    "HattedKJLSAdapter",
    "HattedKAdapter",
]


def _coerce_s(value: Any, name: str = "s") -> tuple[complex, bool]:
    """Return a finite scalar ``s`` and whether the caller supplied it as real."""

    if _is_boolean_scalar(value) or np.ndim(value) != 0:
        raise ValueError(f"{name} must be a finite scalar")
    try:
        z = complex(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not np.isfinite(z):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if z == 0:
        raise ValueError(f"{name} must be nonzero for two-body momentum")
    return z, not np.iscomplexobj(value)


def _momentum_squared(s: Any, mass1: float, mass2: float) -> tuple[complex, bool]:
    """Two-body CM momentum squared and the real-input flag.

    The expression is kept rational in ``s`` so complex continuation does not
    depend on an arbitrary square-root branch.  A square root is only needed
    by the bridge, where the principal ``sqrt(s)`` is the physical ``E_cm``
    continuation.
    """

    z, real_input = _coerce_s(s)
    threshold = (mass1 + mass2) ** 2
    pseudothreshold = (mass1 - mass2) ** 2
    return (z - threshold) * (z - pseudothreshold) / (4.0 * z), real_input


def _one_by_one(value: Any, *, real_input: bool, name: str) -> np.ndarray:
    """Make the protocol's one-by-one real/complex matrix, rejecting NaNs."""

    z = complex(value)
    if not np.isfinite(z):
        raise ValueError(f"{name} is non-finite")
    if real_input:
        scale = max(1.0, abs(z.real))
        if abs(z.imag) > 1e-13 * scale:
            raise ValueError(f"{name} is complex for a real supported point")
        return np.asarray([[float(z.real)]], dtype=float)
    return np.asarray([[z]], dtype=complex)


def _positive_parameter(value: Any, name: str) -> float:
    return _real_scalar(value, name, positive=True)


def _is_boolean_scalar(value: Any) -> bool:
    """Recognise Python and NumPy Boolean scalars, including zero-D arrays."""

    if isinstance(value, (bool, np.bool_)):
        return True
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    return array.ndim == 0 and array.dtype.kind == "b"


def _real_scalar(value: Any, name: str, *, positive: bool = False) -> float:
    """Validate a finite real scalar without accepting Boolean-as-integer input."""

    if _is_boolean_scalar(value):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        return _scalar(value, name, positive=positive)
    except (TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if _is_boolean_scalar(value) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


@register(
    "chung-bw",
    description="Chung hatted-K pole with Blatt-Weisskopf S/P numerator",
    required=("pole_mass", "coupling", "mass1", "mass2", "ell"),
)
class ChungBW:
    r"""Chung's elastic S/P-wave hatted ``K`` model.

    For ``ell`` in ``{0, 1}``,

    ``Khat = coupling**2 * B_ell(k,k_alpha)**2 / (pole_mass**2 - s)``.

    The P-wave barrier factor is

    ``F_1(k)**2 = 2*(k*range_parameter)**2 /
    (1 + (k*range_parameter)**2)``;

    keeping the square in this algebraic form makes the complex continuation
    independent of a square-root sign.  ``inverse`` is evaluated directly from
    the displayed analytic inverse, so it is finite and exactly zero at the
    bare pole even though :meth:`k` refuses that raw-K pole.
    """

    def __init__(
        self,
        *,
        pole_mass: float,
        coupling: float,
        mass1: float,
        mass2: float,
        ell: int,
        range_parameter: float | None = None,
    ) -> None:
        self.pole_mass = _positive_parameter(pole_mass, "pole_mass")
        self.coupling = _real_scalar(coupling, "coupling")
        if self.coupling == 0.0:
            raise ValueError("coupling must be nonzero")
        self.mass1 = _positive_parameter(mass1, "mass1")
        self.mass2 = _positive_parameter(mass2, "mass2")
        self.ell = _integer(ell, "ell")
        if self.ell not in (0, 1):
            raise ValueError("ell must be 0 or 1")

        if self.ell == 1:
            if range_parameter is None:
                raise ValueError("range_parameter is required for ell=1")
            self.range_parameter = _positive_parameter(range_parameter, "range_parameter")
        else:
            # A range has no meaning for the S-wave.  If supplied, validate it
            # rather than silently accepting a malformed paper parameter.
            if range_parameter is not None:
                self.range_parameter = _positive_parameter(range_parameter, "range_parameter")
            else:
                self.range_parameter = None

        self._pole_s = self.pole_mass**2
        self._threshold_s = (self.mass1 + self.mass2) ** 2
        self._pseudothreshold_s = (self.mass1 - self.mass2) ** 2
        self._k2_alpha = self.k_squared(self._pole_s)
        if self.ell == 1:
            r2 = self.range_parameter**2
            denominator = 1.0 + self._k2_alpha * r2
            if denominator == 0:
                raise ValueError(
                    "range_parameter makes the Blatt-Weisskopf denominator vanish at the pole"
                )
            f2_alpha = 2.0 * self._k2_alpha * r2 / denominator
            if f2_alpha == 0:
                raise ValueError(
                    "F_1(k_alpha)=0; choose a pole away from the two-body thresholds"
                )
            if not np.isfinite(f2_alpha):
                raise ValueError("F_1(k_alpha) must be finite")
            self._f2_alpha = complex(f2_alpha)
        else:
            self._f2_alpha = 1.0 + 0.0j

    def k_squared(self, s):
        """Return the analytic two-body momentum squared ``k**2(s)``."""

        value, real_input = _momentum_squared(s, self.mass1, self.mass2)
        if real_input:
            return float(np.real(value))
        return value

    def _barrier_squared(self, s):
        if self.ell == 0:
            return 1.0 + 0.0j
        k2, _ = _momentum_squared(s, self.mass1, self.mass2)
        r2 = self.range_parameter**2
        denominator = 1.0 + k2 * r2
        if denominator == 0:
            raise ValueError("Blatt-Weisskopf denominator vanishes at this complex s")
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            value = (2.0 * k2 * r2 / denominator) / self._f2_alpha
        if not np.isfinite(value):
            raise ValueError("Chung Blatt-Weisskopf factor is non-finite")
        return complex(value)

    def k(self, s) -> np.ndarray:
        """Return hatted ``K`` and refuse its raw bare pole."""

        z, real_input = _coerce_s(s)
        if z == self._pole_s:
            raise ValueError("exact Chung bare K pole: use inverse(s) for its analytic limit")
        barrier2 = self._barrier_squared(s)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            value = self.coupling**2 * barrier2 / (self._pole_s - z)
        return _one_by_one(value, real_input=real_input, name="Chung K")

    def inverse(self, s) -> np.ndarray:
        """Return the hatted inverse ``rho*cot(delta)``.

        At the P-wave thresholds the inverse is genuinely singular.  Refuse
        the exact points with a descriptive error instead of dividing by zero.
        """

        z, real_input = _coerce_s(s)
        if self.ell == 0:
            value = (self._pole_s - z) / self.coupling**2
        else:
            k2, _ = _momentum_squared(s, self.mass1, self.mass2)
            if k2 == 0:
                raise ValueError(
                    "Chung P-wave inverse diverges at the two-body threshold or pseudothreshold"
                )
            r2 = self.range_parameter**2
            # Use the analytic inverse, not 1/B**2 evaluated as a quotient.
            # Besides avoiding 0*infinity at threshold (reported above), this
            # remains exactly finite at the zero of F_1's denominator.
            value = (
                (self._pole_s - z)
                / self.coupling**2
                * self._f2_alpha
                * (1.0 + k2 * r2)
                / (2.0 * k2 * r2)
            )
        return _one_by_one(value, real_input=real_input, name="Chung inverse")

    def k2_power_inverse(self, s):
        """Analytic ``k**(2*ell) * Khat**-1`` for the JLS bridge.

        The P-wave expression is algebraically simplified before evaluation,
        so the physical threshold has a finite value rather than ``0*inf``.
        """

        z, real_input = _coerce_s(s)
        if self.ell == 0:
            value = (self._pole_s - z) / self.coupling**2
        else:
            k2, _ = _momentum_squared(s, self.mass1, self.mass2)
            r2 = self.range_parameter**2
            value = (
                (self._pole_s - z)
                / self.coupling**2
                * self._f2_alpha
                * (1.0 + k2 * r2)
                / (2.0 * r2)
            )
        return complex(value) if not real_input else float(np.real(value))

    def s_breakpoints(self) -> np.ndarray:
        """Real inverse singularities relevant to the physical root scan."""

        if self.ell == 1:
            return _momentum_breakpoints(self._threshold_s, self._pseudothreshold_s)
        return np.zeros(0, dtype=float)


@register(
    "p33-bw",
    description="Elastic P33 Breit-Wigner hatted-K model",
    required=("pole_mass", "coupling", "mass1", "mass2"),
)
class P33BW:
    r"""The ``2101.00689v2`` elastic P33 reduced Breit-Wigner model.

    ``Khat = coupling**2 * k**2 /
    [12*pi*(pole_mass**2-s)]`` and therefore

    ``Khat**-1 = 12*pi*(pole_mass**2-s)/(coupling**2*k**2)``.

    No above-threshold restriction is imposed on ``pole_mass``.  The displayed
    expression is analytic for a subthreshold pole; only the physical momentum
    threshold is singular in the inverse form.
    """

    def __init__(self, *, pole_mass: float, coupling: float, mass1: float, mass2: float) -> None:
        self.pole_mass = _positive_parameter(pole_mass, "pole_mass")
        self.coupling = _real_scalar(coupling, "coupling")
        if self.coupling == 0.0:
            raise ValueError("coupling must be nonzero")
        self.mass1 = _positive_parameter(mass1, "mass1")
        self.mass2 = _positive_parameter(mass2, "mass2")
        self.ell = 1
        self._pole_s = self.pole_mass**2
        self._threshold_s = (self.mass1 + self.mass2) ** 2
        self._pseudothreshold_s = (self.mass1 - self.mass2) ** 2
        if self._pole_s in (self._threshold_s, self._pseudothreshold_s):
            raise ValueError("P33 pole_mass cannot coincide with a two-body branch point")

    def k_squared(self, s):
        value, real_input = _momentum_squared(s, self.mass1, self.mass2)
        if real_input:
            return float(np.real(value))
        return value

    def k(self, s) -> np.ndarray:
        z, real_input = _coerce_s(s)
        if z == self._pole_s:
            raise ValueError("exact P33 bare K pole: use inverse(s) for its analytic limit")
        k2, _ = _momentum_squared(s, self.mass1, self.mass2)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            value = self.coupling**2 * k2 / (12.0 * np.pi * (self._pole_s - z))
        return _one_by_one(value, real_input=real_input, name="P33 K")

    def inverse(self, s) -> np.ndarray:
        z, real_input = _coerce_s(s)
        k2, _ = _momentum_squared(s, self.mass1, self.mass2)
        if k2 == 0:
            raise ValueError("P33 inverse diverges at the two-body threshold or pseudothreshold")
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            value = 12.0 * np.pi * (self._pole_s - z) / (self.coupling**2 * k2)
        return _one_by_one(value, real_input=real_input, name="P33 inverse")

    def k2_power_inverse(self, s):
        """Analytic ``k**2*Khat**-1`` used at and away from threshold."""

        z, real_input = _coerce_s(s)
        value = 12.0 * np.pi * (self._pole_s - z) / self.coupling**2
        return complex(value) if not real_input else float(np.real(value))

    def s_breakpoints(self) -> np.ndarray:
        return _momentum_breakpoints(self._threshold_s, self._pseudothreshold_s)


def _momentum_breakpoints(threshold: float, pseudothreshold: float) -> np.ndarray:
    """Positive real zeros of ``k**2`` where a P-wave inverse is singular."""

    return np.unique(
        np.asarray([value for value in (threshold, pseudothreshold) if value > 0.0], dtype=float)
    )


@dataclass(frozen=True)
class HattedKJLSAdapter:
    """Adapt one hatted-K partial wave to a scale-weighted JLS block.

    ``twice_S`` and ``twice_J`` declare the active LS incidence.  The adapter
    returns only that block, allowing callers to combine independently bound
    S31/P33 adapters under ``selected_j_sectors`` without padding an omitted
    wave with a zero or a large surrogate.
    """

    model: Any
    ell: int
    twice_S: int
    twice_J: int
    mass1: float | None = None
    mass2: float | None = None

    def __post_init__(self) -> None:
        ell = _integer(self.ell, "ell")
        twice_s = _integer(self.twice_S, "twice_S")
        twice_j = _integer(self.twice_J, "twice_J")
        if twice_j < abs(2 * ell - twice_s) or twice_j > 2 * ell + twice_s:
            raise ValueError(
                f"twice_J={twice_j} is not triangle allowed for (ell, twice_S)=({ell}, {twice_s})"
            )
        if (twice_j - (2 * ell + twice_s)) % 2:
            raise ValueError(
                f"twice_J={twice_j} has the wrong triangle parity for (ell, twice_S)=({ell}, {twice_s})"
            )
        if not hasattr(self.model, "k") or not hasattr(self.model, "inverse"):
            raise ValueError("hatted-K model must provide k(s) and inverse(s)")
        model_ell = getattr(self.model, "ell", None)
        if model_ell is not None and _integer(model_ell, "model.ell") != ell:
            raise ValueError(f"adapter ell={ell} does not match model ell={model_ell}")
        model_m1 = getattr(self.model, "mass1", None)
        model_m2 = getattr(self.model, "mass2", None)
        if (model_m1 is None) != (model_m2 is None):
            raise ValueError("hatted-K model must expose both mass1 and mass2, or neither")
        m1 = self.mass1 if self.mass1 is not None else model_m1
        m2 = self.mass2 if self.mass2 is not None else model_m2
        if m1 is None or m2 is None:
            raise ValueError("adapter requires positive mass1 and mass2")
        m1 = _positive_parameter(m1, "mass1")
        m2 = _positive_parameter(m2, "mass2")
        if model_m1 is not None and (
            m1 != _positive_parameter(model_m1, "model.mass1")
            or m2 != _positive_parameter(model_m2, "model.mass2")
        ):
            raise ValueError("adapter masses must match the hatted-K model's mass1 and mass2")
        object.__setattr__(self, "ell", ell)
        object.__setattr__(self, "twice_S", twice_s)
        object.__setattr__(self, "twice_J", twice_j)
        object.__setattr__(self, "mass1", m1)
        object.__setattr__(self, "mass2", m2)

    def _k2(self, s):
        value, real_input = _momentum_squared(s, self.mass1, self.mass2)
        return (float(np.real(value)) if real_input else value), real_input

    def reduced_inverse(self, s) -> np.ndarray:
        """Return ``R_scale = (sqrt(s)/2) k**(2ell) Khat**-1`` as a 1x1 block."""

        z, real_input = _coerce_s(s)
        if real_input and z.real <= 0.0:
            raise ValueError("JLS adapter requires positive real s for E_cm")
        # Models in this module expose an analytic cancellation hook.  A
        # generic model can still use the ordinary product away from kinematic
        # thresholds.
        k2, _ = self._k2(s)
        has_cancellation = hasattr(self.model, "k2_power_inverse")
        if has_cancellation:
            # Away from a true P-wave threshold, also enforce the model
            # protocol's exact one-channel, finite real inverse contract.  At
            # k^2=0 the model inverse is singular by design and the analytic
            # cancellation hook is the supported reduced form.
            if self.ell == 0 or k2 != 0:
                self._inverse_value(s, real_input)
            product = self.model.k2_power_inverse(s)
        else:
            inverse = self._inverse_value(s, real_input)
            product = (k2**self.ell) * inverse
        if np.ndim(product) != 0:
            raise ValueError("hatted-K cancellation hook must return a scalar")
        product = complex(product)
        value = np.sqrt(z) * product / 2.0
        return _one_by_one(value, real_input=real_input, name="JLS reduced inverse")

    def _inverse_value(self, s, real_input: bool) -> complex:
        inverse = np.asarray(self.model.inverse(s))
        if inverse.shape != (1, 1):
            raise ValueError("hatted-K model inverse must have exact shape (1, 1)")
        checked = _one_by_one(
            inverse[0, 0], real_input=real_input, name="hatted-K model inverse"
        )
        return complex(checked[0, 0])

    def block(self, s) -> np.ndarray:
        return self.reduced_inverse(s)

    def blocks(self, s) -> dict[int, np.ndarray]:
        return {self.twice_J: self.reduced_inverse(s)}

    def threshold_block(self, s, scale: float) -> np.ndarray:
        """Equivalent threshold-weighted block for positive real ``k**2``.

        This is optional convenience for callers that deliberately choose the
        volume-dependent threshold weighting.  Production paper integrations
        should use :meth:`blocks` with ``weighting='scale'``.
        """

        scale = _positive_parameter(scale, "scale")
        z, real_input = _coerce_s(s)
        k2, _ = self._k2(s)
        if real_input and k2 < 0:
            raise ValueError("threshold weighting requires nonnegative real k^2")
        if self.ell > 0 and k2 == 0:
            raise ValueError("threshold weighting is singular at k^2=0 for ell>0")
        # Eq. (4) of normalization-bridge-v1 for a one-wave block.
        factor = (2.0 * np.sqrt(k2)) ** (2 * self.ell) * scale ** (2 * self.ell + 1)
        value = factor * complex(self.reduced_inverse(s)[0, 0])
        return _one_by_one(value, real_input=real_input, name="threshold-weighted inverse")

    def __call__(self, s) -> np.ndarray:
        return self.reduced_inverse(s)

    def s_breakpoints(self) -> np.ndarray:
        """Delegate real model breakpoints for callers preparing root scans."""

        if not hasattr(self.model, "s_breakpoints"):
            return np.zeros(0, dtype=float)
        return np.asarray(self.model.s_breakpoints(), dtype=float)


# Friendly compatibility spellings for code that calls the models K forms.
ChungK = ChungBW
P33K = P33BW
HattedKAdapter = HattedKJLSAdapter
