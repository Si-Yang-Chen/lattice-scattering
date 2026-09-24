"""Complex-plane poles of the coupled amplitude (P6).

The amplitude is ``t(s) = (K(s)^{-1} + I(s))^{-1}``, so its poles are the zeros
of the *inverse* amplitude matrix:

* ``simple`` phase space: ``t^{-1} = K^{-1} - i diag(rho_i)``
  (the inverse of the legacy ``t_from_k``, ``t = (1 - i K rho)^{-1} K``);
* ``chew-mandelstam``: ``t^{-1} = K^{-1} + diag(I_i)`` with the full complex CM
  function on the requested sheet.

Pole searches are **local**: the caller supplies a rectangle that avoids cuts,
thresholds and model singularities, and the returned residues follow the legacy
convention ``t(s) ~ R / (s_p - s)``.  No completeness claim is made.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .models import PolePolynomialK
from .phase_space import cm_complex, physical_rho

__all__ = ["Pole", "amplitude", "amplitude_inverse", "solve_poles"]


def _masses(channel_masses) -> np.ndarray:
    masses = np.asarray(channel_masses, dtype=float)
    if (
        masses.ndim != 2
        or masses.shape[1] != 2
        or not len(masses)
        or not np.all(np.isfinite(masses))
        or np.any(masses <= 0.0)
    ):
        raise ValueError("an (N, 2) array of positive finite mass pairs is required")
    return masses


def _sheets(sheets, count: int) -> tuple[int, ...]:
    if sheets is None:
        return (1,) * count
    values = tuple(int(s) for s in sheets)
    if len(values) != count or any(s not in (1, 2) for s in values):
        raise ValueError("one sheet (1 or 2) per channel required")
    return values


def amplitude_inverse(model, masses, phase_space: str, s, *, sheets=None, subtractions=()):
    """``t(s)^{-1}`` at complex ``s`` for the requested sheets."""
    masses = _masses(masses)
    count = len(masses)
    if phase_space not in ("simple", "chew-mandelstam"):
        raise ValueError(f"unknown phase space {phase_space!r}")
    k_inverse = np.asarray(model.inverse(s), dtype=complex)
    if k_inverse.shape != (count, count):
        raise ValueError("model inverse has the wrong channel dimension")
    if not np.all(np.isfinite(k_inverse)):
        raise ValueError("non-finite K inverse")
    sheets = _sheets(sheets, count)
    if phase_space == "simple":
        loops = np.array([physical_rho(s, m1, m2) for m1, m2 in masses], dtype=complex)
        return k_inverse - 1j * np.diag(loops)
    subs = tuple(subtractions) if len(subtractions) else (0.0,) * count
    if len(subs) != count:
        raise ValueError("one subtraction per channel required")
    loops = np.array(
        [cm_complex(s, m1, m2, subtraction=sub, sheet=sheet) for (m1, m2), sub, sheet in zip(masses, subs, sheets)],
        dtype=complex,
    )
    return k_inverse + np.diag(loops)


def amplitude(model, masses, phase_space: str, s, *, sheets=None, subtractions=()):
    """``t(s)`` (the inverse of :func:`amplitude_inverse`)."""
    inverse = amplitude_inverse(
        model, masses, phase_space, s, sheets=sheets, subtractions=subtractions
    )
    return np.linalg.solve(inverse, np.eye(len(inverse)))


@dataclass(frozen=True)
class Pole:
    """One locally refined simple pole."""

    s: complex
    sheet: tuple[int, ...]
    residue: np.ndarray
    smallest_singular_value: float
    evaluations: int
    scope: str = "local simple pole; t(s) ~ residue / (pole - s)"

    def as_dict(self) -> dict:
        return {
            "s": complex(self.s),
            "mass": float(np.sqrt(self.s).real) if self.s.real > 0 else None,
            "sheet": list(self.sheet),
            "residue": self.residue.tolist(),
            "smallest_singular_value": float(self.smallest_singular_value),
            "evaluations": int(self.evaluations),
            "scope": self.scope,
        }


def _refine(matrix, initial, window, *, step, tolerance):
    """Refine one zero of ``matrix`` inside ``window`` (legacy-compatible)."""
    lower, upper = (np.asarray(bound, dtype=float) for bound in window)
    if lower.shape != (2,) or upper.shape != (2,) or np.any(lower >= upper):
        raise ValueError("window requires finite [Re, Im] lower and upper bounds")
    start = [complex(initial).real, complex(initial).imag]
    scale = max(float(np.linalg.norm(matrix(complex(initial)), ord=2)), 1.0)

    def objective(x):
        value = np.linalg.det(matrix(complex(*x)) / scale)
        return [value.real, value.imag]

    result = least_squares(
        objective, start, bounds=(lower, upper), xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=300
    )
    z = complex(*result.x)
    value = matrix(z)
    _, singular, vh = np.linalg.svd(value)
    if not result.success or singular[-1] > tolerance * scale:
        return None
    if len(singular) > 1 and singular[-2] < 100.0 * tolerance * scale:
        raise RuntimeError("multiple zero modes: the simple-pole residue is undefined")
    margin = np.minimum(result.x - lower, upper - result.x)
    if np.any(margin <= step):
        return None
    vector = vh[-1].conj()
    finer = (matrix(z + step / 2) - matrix(z - step / 2)) / step
    slope = vector.T @ finer @ vector
    if abs(slope) < 1e-8 * scale:
        return None
    residue = -np.outer(vector, vector) / slope
    return z, residue, float(singular[-1]), int(result.nfev)


def solve_poles(
    model,
    masses,
    phase_space: str,
    window,
    *,
    sheets=None,
    subtractions=(),
    samples: int = 21,
    initial=(),
    step: float = 1e-5,
    tolerance: float = 1e-9,
    cluster_tolerance: float = 1e-6,
    lhc_domain=None,
) -> tuple[Pole, ...]:
    """Locally refine every simple pole found on a coarse complex grid.

    The grid over ``window`` (``[(Re_lo, Im_lo), (Re_hi, Im_hi)]``) supplies
    starting points; each is refined with a bounded least-squares solve and the
    results are clustered with ``cluster_tolerance``.  Extra starting points can
    be supplied through ``initial``.  A zero that cannot be refined to a simple
    pole is skipped, not reported as a pole.
    """
    masses = _masses(masses)
    count = len(masses)
    sheets = _sheets(sheets, count)
    lower, upper = (np.asarray(bound, dtype=float) for bound in window)
    if lower.shape != (2,) or upper.shape != (2,) or np.any(lower >= upper):
        raise ValueError("window requires finite [Re, Im] lower and upper bounds")
    if lhc_domain is not None:
        from .left_hand_cut import EqualMassExchangeDomain
        if not isinstance(lhc_domain, EqualMassExchangeDomain):
            raise ValueError("lhc_domain must be an EqualMassExchangeDomain")
        if lhc_domain.units != "temporal_lattice" or count != 1 or any(m != lhc_domain.mass for m in masses[0]):
            raise ValueError("lhc_domain requires one matching equal-mass channel in temporal_lattice units")
        # Check the whole search rectangle before the optimizer, not only its
        # seeds. A horizontal real cut intersects a rectangle iff its boundary
        # does (apart from the impossible fully enclosed semi-infinite ray).
        boundary = [complex(lower[0], lower[1]), complex(upper[0], lower[1]),
                    complex(upper[0], upper[1]), complex(lower[0], upper[1]),
                    complex(lower[0], lower[1])]
        lhc_domain.check((lower[0], upper[0]), path_s=boundary,
                         sheet="I" if sheets[0] == 1 else "II").require_path()
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 3:
        raise ValueError("samples must be an integer >= 3")

    def matrix(z):
        return amplitude_inverse(
            model, masses, phase_space, z, sheets=sheets, subtractions=subtractions
        )

    real_axis = np.linspace(lower[0], upper[0], samples)
    imag_axis = np.linspace(lower[1], upper[1], samples)
    starts = [complex(*start) for start in initial]
    if not starts:
        for re in real_axis:
            for im in imag_axis:
                starts.append(complex(re, im))

    found: list[Pole] = []
    for start in starts:
        try:
            refined = _refine(matrix, start, (lower, upper), step=step, tolerance=tolerance)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            continue
        if refined is None:
            continue
        z, residue, smallest, evaluations = refined
        duplicate = False
        for index, existing in enumerate(found):
            if abs(existing.s - z) <= cluster_tolerance:
                duplicate = True
                # Keep the better-converged representative.
                if smallest < existing.smallest_singular_value:
                    found[index] = Pole(
                        s=z,
                        sheet=sheets,
                        residue=residue,
                        smallest_singular_value=smallest,
                        evaluations=evaluations,
                    )
                break
        if not duplicate:
            found.append(
                Pole(
                    s=z,
                    sheet=sheets,
                    residue=residue,
                    smallest_singular_value=smallest,
                    evaluations=evaluations,
                )
            )
    found.sort(key=lambda pole: (pole.s.real, pole.s.imag))
    return tuple(found)
