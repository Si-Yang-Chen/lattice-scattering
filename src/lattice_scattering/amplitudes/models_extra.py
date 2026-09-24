"""Additional K-matrix parameterisations met while transcribing papers.

Each one is registered, so it is reachable by name and ``registered()`` lists it.  The
printed parameters match the papers' own symbols; the docstring of each class records the
equation it implements and the paper it came from, because a parameterisation without its
convention is unusable.

A recurring subtlety: these papers print ``K`` (so that ``K = tan delta``) *or* the hatted
``K-hat = tan(delta)/rho``, and the two differ by ``rho = 2p/sqrt(s)``.  The kernel stores
``K^-1 = rho cot(delta)``, which is ``K-hat^-1``.  ``2006.14035v2`` states the relation
outright in its Eq. 7, so the conversions below are anchored rather than guessed.
"""

from __future__ import annotations

import numpy as np

from .models import _scalar, _symmetric
from .registry import register

__all__ = ["ChiralEREPcotdelta", "ChiralPoleK", "BuggK", "ConformalMapK", "PolygonPcotdelta"]


def _p_squared(s, mass1: float, mass2: float):
    """Two-body cm momentum squared, ``(s - (m1+m2)^2)(s - (m1-m2)^2) / (4 s)``."""
    total = (mass1 + mass2) ** 2
    gap = (mass1 - mass2) ** 2
    return (s - total) * (s - gap) / (4.0 * s)


@register("chiral-ere-pcotdelta",
          description="Chirally modified ERE: k cot(delta) = (Mpi/Epi)(1/a0 + r0 k^2/2)",
          required=("scattering_length", "effective_range", "pion_mass", "heavy_mass",
                    "mass1", "mass2"))
class ChiralEREPcotdelta:
    """``2604.18286v1`` Eq. (9), the MERE parameterisation.

    ``k cot(delta) = (Mpi / Epi) (1/a0 + r0 k^2 / 2)`` with
    ``Epi = (s + Mpi^2 - M_heavy^2) / (2 sqrt(s))``.  The factor ``Mpi/Epi`` is what makes it
    chirally modified; dropping it gives the ordinary ERE.

    ``heavy_mass`` is the second hadron mass of the channel (the ``D`` in the paper's
    ``Epi``), not a resonance mass.
    """

    def __init__(self, *, scattering_length: float, effective_range: float,
                 pion_mass: float, heavy_mass: float, mass1: float, mass2: float):
        self.a0 = _scalar(scattering_length, "scattering_length")
        self.r0 = _scalar(effective_range, "effective_range")
        self.m_pi = _scalar(pion_mass, "pion_mass", positive=True)
        self.m_heavy = _scalar(heavy_mass, "heavy_mass", positive=True)
        self.mass1 = _scalar(mass1, "mass1", positive=True)
        self.mass2 = _scalar(mass2, "mass2", positive=True)

    def k_cot_delta(self, s):
        s = complex(s)
        e_pi = (s + self.m_pi**2 - self.m_heavy**2) / (2.0 * np.sqrt(s))
        p2 = _p_squared(s, self.mass1, self.mass2)
        return (self.m_pi / e_pi) * (1.0 / self.a0 + 0.5 * self.r0 * p2)

    def inverse(self, s):
        """``K^-1 = rho cot(delta) = (2/sqrt(s)) k cot(delta)`` -- Eq. 7's ``K-hat``."""
        s = complex(s)
        return np.array([[2.0 / np.sqrt(s) * self.k_cot_delta(s)]])

    def k(self, s):
        return np.linalg.inv(self.inverse(s))

    def s_breakpoints(self):
        return np.zeros(0, dtype=float)


@register("chiral-pole-k",
          description="Chirally modified K-matrix: K = (Epi/Mpi) g0^2/(m0^2 - s)",
          required=("pole_mass", "coupling", "pion_mass", "heavy_mass",
                    "mass1", "mass2"))
class ChiralPoleK:
    """``2604.18286v1`` Eq. (9), the MK parameterisation.

    ``K(s) = (Epi / Mpi) g0^2 / (m0^2 - s)`` with the same ``Epi`` as
    :class:`ChiralEREPcotdelta`.  Note this one is the *unhatted* ``K``, so our stored
    ``K^-1 = rho cot(delta)`` is ``K^-1`` divided by ``rho``.
    """

    def __init__(self, *, pole_mass: float, coupling: float, pion_mass: float,
                 heavy_mass: float, mass1: float, mass2: float):
        self.m0 = _scalar(pole_mass, "pole_mass", positive=True)
        self.g0 = _scalar(coupling, "coupling")
        self.m_pi = _scalar(pion_mass, "pion_mass", positive=True)
        self.m_heavy = _scalar(heavy_mass, "heavy_mass", positive=True)
        self.mass1 = _scalar(mass1, "mass1", positive=True)
        self.mass2 = _scalar(mass2, "mass2", positive=True)

    def k(self, s):
        real_input = not np.iscomplexobj(s)
        s = complex(s)
        if s == self.m0**2:
            raise ValueError("exact bare K pole: use a formulation with an explicit limit")
        e_pi = (s + self.m_pi**2 - self.m_heavy**2) / (2.0 * np.sqrt(s))
        value = (e_pi / self.m_pi) * self.g0**2 / (self.m0**2 - s)
        return np.asarray([[value.real if real_input else value]])

    def inverse(self, s):
        """``rho cot(delta)``; ``K = tan(delta)`` so ``K^-1 = rho / K``."""
        s = complex(s)
        p2 = _p_squared(s, self.mass1, self.mass2)
        rho = 2.0 * np.sqrt(p2) / np.sqrt(s) if np.iscomplexobj(p2) else 2.0 * np.sqrt(p2) / np.sqrt(s)
        return np.array([[rho / self.k(s)[0, 0]]])

    def s_breakpoints(self):
        return np.asarray([self.m0**2], dtype=float)


@register("bugg-k",
          description="Bugg's Adler-zero pole: K-hat = G0(s)^2/(m0^2 - s), G0^2 = G0_0^2 (s-sA)/(sA-m0^2)",
          required=("pole_mass", "coupling", "adler_zero"))
class BuggK:
    """``2006.14035v2`` Eqs. (21)-(23), Bugg's parameterisation.

    ``K-hat(s) = G0(s)^2 / (m0^2 - s)`` with
    ``G0(s) = G0_0 sqrt((s - sA)/(sA - m0^2))``, so the amplitude carries the Adler zero at
    ``s = sA`` by construction.  The overall sign convention is the paper's: ``G0_0`` may
    carry the sign, and only its square appears.
    """

    def __init__(self, *, pole_mass: float, coupling: float, adler_zero: float):
        self.m0 = _scalar(pole_mass, "pole_mass", positive=True)
        self.g0 = _scalar(coupling, "coupling")
        self.s_adler = _scalar(adler_zero, "adler_zero")
        if self.s_adler == self.m0**2:
            raise ValueError("adler_zero and pole_mass^2 coincide: G0(s) is not defined")

    def envelope_squared(self, s):
        s = complex(s)
        return self.g0**2 * (s - self.s_adler) / (self.s_adler - self.m0**2)

    def inverse(self, s):
        """``K-hat^-1 = (m0^2 - s) (sA - m0^2) / (G0_0^2 (s - sA))``.

        The paper multiplies the pole by the envelope ``s - sA`` so the *T* matrix vanishes at
        the Adler zero.  Written as an inverse that puts ``s - sA`` in the denominator, so
        ``sA`` is a pole of the inverse -- a singular point the scan must split at, not a
        smooth zero.
        """
        real_input = not np.iscomplexobj(s)
        s = complex(s)
        if s == self.m0**2:
            raise ValueError("exact bare K pole: use a formulation with an explicit limit")
        if s == self.s_adler:
            raise ValueError("exact Adler zero: the inverse diverges here")
        value = (self.m0**2 - s) * (self.s_adler - self.m0**2) / (self.g0**2 * (s - self.s_adler))
        return np.asarray([[value.real if real_input else value]])

    def k(self, s):
        return np.linalg.inv(self.inverse(s))

    def s_breakpoints(self):
        """The bare pole and the Adler zero: both are poles of the inverse form."""
        return np.unique(np.asarray([self.m0**2, self.s_adler], dtype=float))


@register("conformal-map-k",
          description="Conformal map with Adler zero: K-hat^-1 = F(s) sum_k c_k omega(s)^k",
          required=("coefficients", "alpha", "s0", "s_adler", "delta_kpi"))
class ConformalMapK:
    """``2006.14035v2`` Eqs. (24)-(26), the conformal-map parameterisation.

    ``K-hat^-1(s) = F(s) sum_k c_k omega(s)^k`` with ``F(s) = 1/(s - sA)`` and

        ``omega(y) = (sqrt(y) - alpha sqrt(y0 - y)) / (sqrt(y) + alpha sqrt(y0 - y))``,
        ``y(s) = ((s - Delta_Kpi) / (s + Delta_Kpi))^2``, ``y0 = y(s0)``.

    ``F(s)`` supplies the Adler zero.  ``s_adler`` is the Adler zero ``sA`` and
    ``delta_kpi = m_K^2 - m_pi^2`` as printed.
    """

    def __init__(self, *, coefficients, alpha: float, s0: float, s_adler: float,
                 delta_kpi: float):
        self.coefficients = np.array(coefficients, dtype=float).reshape(-1)
        if not self.coefficients.size or not np.all(np.isfinite(self.coefficients)):
            raise ValueError("coefficients must be a non-empty finite real sequence")
        if isinstance(alpha, (bool, np.bool_)) or (
            np.ndim(alpha) == 0 and np.asarray(alpha).dtype.kind == "b"
        ):
            raise ValueError("alpha must be a finite real scalar")
        self.alpha = _scalar(alpha, "alpha")
        self.s0 = _scalar(s0, "s0", positive=True)
        self.s_adler = _scalar(s_adler, "s_adler")
        self.delta = _scalar(delta_kpi, "delta_kpi", positive=True)
        if self.alpha <= 0.0:
            raise ValueError("conformal map alpha must be positive")
        self.y0 = self._y(self.s0)

    def _y(self, s):
        s = complex(s)
        return ((s - self.delta) / (s + self.delta)) ** 2

    def omega(self, s):
        """The conformal variable; ``omega(s0) = 1`` because ``y(s0) = y0``.

        The map needs ``y(s) <= y0``.  Since ``y`` rises monotonically to 1, ``s0`` is the
        upper end of the real domain and anything above it would make ``sqrt(y0 - y)``
        imaginary and the result silently NaN.  Refusing is better than returning NaN, which
        a root scan would treat as a free-pole-like hole.
        """
        y = self._y(s)
        gap = self.y0 - y
        if not np.iscomplexobj(s):
            # On the real axis a negative gap is a domain error, and the imaginary part the
            # map would otherwise produce is a silent NaN source for the root scan.
            if float(np.real(gap)) < -1e-14:
                raise ValueError(
                    f"conformal map is defined only for y <= y0, i.e. s <= s0={self.s0!r}; "
                    f"got s={s!r} with y={float(np.real(y))!r} > y0={self.y0!r}"
                )
        gap = gap if np.iscomplexobj(gap) else max(np.real(gap), 0.0)
        root_y = np.sqrt(y)
        root_rest = np.sqrt(gap)
        return (root_y - self.alpha * root_rest) / (root_y + self.alpha * root_rest)

    def inverse(self, s):
        """``K-hat^-1``; ``F(s) = 1/(s - sA)`` supplies the Adler zero.

        ``omega`` is asked with the caller's own ``s``, not with a coerced complex copy: the
        domain check keys off whether the input was real, and coercing first would hide a
        real-axis domain error behind a complex one.
        """
        real_input = not np.iscomplexobj(s)
        omega = self.omega(s)
        s = complex(s)
        if np.isfinite(np.abs(s)) and abs(s - self.s_adler) < 1e-15:
            raise ValueError("exact Adler zero: use a formulation with an explicit limit")
        series = sum(c * omega**k for k, c in enumerate(self.coefficients))
        value = series / (s - self.s_adler)
        return np.asarray([[value.real if real_input else value]])

    def k(self, s):
        return np.linalg.inv(self.inverse(s))

    def s_breakpoints(self):
        """``sA``: ``F(s)`` diverges there, so the determinant can flip sign across it."""
        return np.asarray([self.s_adler], dtype=float)


@register("polygon-pcotdelta",
          description="Paper-printed p cot(delta) as a polynomial in p^2 (any order)",
          required=("mass1", "mass2", "coefficients"))
class PolygonPcotdelta:
    """``p cot(delta) = sum_k c_k (p^2)^k``, the plainest printed amplitude form.

    Kept separate from :class:`ChiralEREPcotdelta` so an ordinary effective-range expansion is
    available without inventing a chiral factor.  ``coefficients`` are in ascending powers of
    ``p^2``.  The kernel stores ``K^-1 = rho cot(delta)``, and Eq. 7 of ``2006.14035v2`` fixes
    the relation as ``K-hat = tan(delta)/rho``, so ``K^-1 = (2/sqrt(s)) p cot(delta)``.
    """

    def __init__(self, *, coefficients, mass1: float, mass2: float):
        self.coefficients = np.array(coefficients, dtype=float).reshape(-1)
        if not self.coefficients.size or not np.all(np.isfinite(self.coefficients)):
            raise ValueError("coefficients must be a non-empty finite real sequence")
        self.mass1 = _scalar(mass1, "mass1", positive=True)
        self.mass2 = _scalar(mass2, "mass2", positive=True)

    def p_cot_delta(self, s):
        p2 = _p_squared(complex(s), self.mass1, self.mass2)
        return sum(c * p2**k for k, c in enumerate(self.coefficients))

    def inverse(self, s):
        real_input = not np.iscomplexobj(s)
        s = complex(s)
        value = 2.0 / np.sqrt(s) * self.p_cot_delta(s)
        return np.asarray([[value.real if real_input else value]])

    def k(self, s):
        return np.linalg.inv(self.inverse(s))

    def s_breakpoints(self):
        return np.zeros(0, dtype=float)
