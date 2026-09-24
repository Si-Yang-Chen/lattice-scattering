"""Chew-Mandelstam self-energy and the N-channel S-wave phase space (P4).

The S-wave Chew-Mandelstam function on the physical upper rim is taken from
the legacy convention (1411.2004 App. B, as implemented in
``amplitudes.chew_mandelstam``), so the two can be compared pointwise:

    xi = (s - (m1+m2)^2) / s
    rho(s) = sqrt((s - (m1+m2)^2)(s - (m1-m2)^2)) / s
    I(s) = rho/pi * log((xi + rho)/(xi - rho))
           - xi/pi * (m2-m1)/(m1+m2) * log(m2/m1) + subtraction

Above threshold the log is rationalised (``2 log(rho + xi) - log(xi*4 m1 m2/s)
- i*pi``) to avoid the cancellation between ``rho`` and ``xi`` at high energy.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "cm_complex",
    "cm_real_axis",
    "effective_inverse_k",
    "physical_rho",
]


def _real_scalar(value, name: str) -> float:
    if isinstance(value, bool) or np.iscomplexobj(value) or np.ndim(value) != 0 or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real scalar")
    return float(value)


def _masses(mass1, mass2) -> tuple[float, float]:
    m1 = _real_scalar(mass1, "mass1")
    m2 = _real_scalar(mass2, "mass2")
    if m1 <= 0.0 or m2 <= 0.0:
        raise ValueError("positive masses required")
    return m1, m2


def cm_real_axis(s, mass1: float, mass2: float, *, subtraction: float = 0.0) -> complex:
    """``I(s + i0)`` for real ``s`` above the pseudothreshold.

    ``Im I = -rho`` above threshold and ``I(sth) = subtraction``.  The
    subtraction is a real constant, not an energy dependent removal.
    """
    s = _real_scalar(s, "s")
    m1, m2 = _masses(mass1, mass2)
    subtraction = _real_scalar(subtraction, "subtraction")
    if s <= (m1 - m2) ** 2:
        raise ValueError("s above the pseudothreshold required")
    threshold = (m1 + m2) ** 2
    if s == threshold:
        return complex(subtraction)
    xi = (s - threshold) / s
    rho = np.sqrt(complex((s - threshold) * (s - (m1 - m2) ** 2))) / s
    if s > threshold:
        # rho - xi is a small difference at high energy; rationalise it.
        denominator = xi * 4.0 * m1 * m2 / s
        logarithm = 2.0 * np.log(rho.real + xi) - np.log(denominator) - 1j * np.pi
    else:
        logarithm = np.log((xi + rho) / (xi - rho))
    unequal = xi / np.pi * (m2 - m1) / (m1 + m2) * np.log(m2 / m1)
    return complex(subtraction + rho / np.pi * logarithm - unequal)


def physical_rho(s, mass1: float, mass2: float):
    """Right-cut momentum branch: positive above the upper rim.

    Domain ``Re(s) > pseudothreshold``; on the real cut the upper rim is
    selected.  Complex ``s`` follows the Schwarz reflection below the cut.
    """
    m1, m2 = _masses(mass1, mass2)
    z = complex(s)
    if not np.isfinite(z) or z.real <= (m1 - m2) ** 2:
        raise ValueError("requires finite s with Re(s) > pseudothreshold")
    if z.imag < 0:
        return -np.conj(physical_rho(z.conjugate(), m1, m2))
    return np.sqrt(z - (m1 + m2) ** 2) * np.sqrt(z - (m1 - m2) ** 2) / z


def cm_complex(s, mass1: float, mass2: float, *, subtraction: float = 0.0, sheet: int = 1) -> complex:
    """``I(s)`` on the physical sheet (1) or the adjacent sheet (2).

    Sheet 2 is ``I_II = I_I + 2 i rho_I``.  Real arguments take the upper rim;
    the lower half plane follows Schwarz reflection.
    """
    if sheet not in (1, 2):
        raise ValueError("sheet must be 1 or 2")
    m1, m2 = _masses(mass1, mass2)
    subtraction = _real_scalar(subtraction, "subtraction")
    z = complex(s)
    rho = physical_rho(z, m1, m2)
    if z.imag == 0:
        value = cm_real_axis(z.real, m1, m2, subtraction=subtraction)
    elif z.imag < 0:
        value = np.conj(cm_complex(z.conjugate(), m1, m2, subtraction=subtraction))
    else:
        xi = 1.0 - (m1 + m2) ** 2 / z
        value = subtraction + rho / np.pi * np.log((xi + rho) / (xi - rho))
        value -= xi / np.pi * (m2 - m1) / (m1 + m2) * np.log(m2 / m1)
    return complex(value + (2j * rho if sheet == 2 else 0.0))


def effective_inverse_k(model, masses, phase_space: str, s, *, subtractions=()) -> np.ndarray:
    """Channel-basis effective inverse ``K`` on the real axis.

    ``simple`` is the physical ``K(s)^{-1}``; ``chew-mandelstam`` adds the
    real part of the CM self-energy, ``K(s)^{-1} + I(s) + i rho(s)``.  The
    closed channels contribute ``i rho = -|rho|``, which must be kept: omitting
    it double counts their infinite-volume piece already present in the box.
    """
    masses = np.asarray(masses, dtype=float)
    if masses.ndim != 2 or masses.shape[1] != 2 or not len(masses):
        raise ValueError("masses must be an (N, 2) array")
    if phase_space not in ("simple", "chew-mandelstam"):
        raise ValueError(f"unknown phase space {phase_space!r}")
    inverse = np.asarray(model.inverse(s), dtype=complex)
    if inverse.shape != (len(masses),) * 2:
        raise ValueError("model inverse has the wrong channel dimension")
    if phase_space == "simple":
        if not np.allclose(inverse.imag, 0.0, atol=1e-12 * max(1.0, np.max(np.abs(inverse)))):
            raise ValueError("simple phase space requires a real K inverse on the axis")
        return inverse.real
    subs = tuple(subtractions) if len(subtractions) else (0.0,) * len(masses)
    if len(subs) != len(masses):
        raise ValueError("one subtraction per channel required")
    s = _real_scalar(s, "s")
    diagonal = np.array(
        [
            cm_real_axis(s, m1, m2, subtraction=sub) + 1j * physical_rho(s, m1, m2)
            for (m1, m2), sub in zip(masses, subs)
        ]
    )
    value = inverse + np.diag(diagonal)
    if not np.allclose(value.imag, 0.0, atol=1e-9 * max(1.0, np.max(np.abs(value.real)))):
        raise ValueError("the effective inverse K is not real on the real axis")
    return value.real
