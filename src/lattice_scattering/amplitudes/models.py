"""Real S-wave K-matrix parameterisations for the v2 amplitude layer (P4).

Real symmetric ``K(s)`` in the channel basis, matching the semantics of the
legacy ``amplitudes.coupled`` models so the two can be compared pointwise:

    K(s) = g g^T / (m^2 - s) + sum_n C_n ((s - s_ref)/scale^2)^n

Only the real open-channel parameterisation lives here; the finite-volume
conversion and phase space are in
:mod:`lattice_scattering.finite_volume.coupled_s`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .registry import register

__all__ = ["ConstantK", "PolePolynomialK"]


def _symmetric(matrix, name: str) -> np.ndarray:
    """A finite real symmetric square matrix, read-only."""
    if np.iscomplexobj(matrix):
        raise ValueError(f"{name} must be real; complex parts are never discarded")
    value = np.array(matrix, dtype=float, copy=True)
    if (
        value.ndim != 2
        or value.shape[0] == 0
        or value.shape[0] != value.shape[1]
        or not np.all(np.isfinite(value))
        or not np.allclose(value, value.T, atol=1e-14, rtol=1e-14)
    ):
        raise ValueError(f"{name} must be a finite real symmetric square matrix")
    value.setflags(write=False)
    return value


def _scalar(value, name: str, *, positive: bool = False) -> float:
    """A finite real scalar, optionally required to be positive."""
    if (
        isinstance(value, (bool, np.bool_))
        or np.iscomplexobj(value)
        or np.ndim(value) != 0
        or not np.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite real scalar")
    out = float(value)
    if positive and out <= 0.0:
        raise ValueError(f"{name} must be positive")
    return out


@dataclass(frozen=True)
class ConstantK:
    """A constant real symmetric ``K``."""

    matrix: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "matrix", _symmetric(self.matrix, "ConstantK.matrix"))

    def k(self, s) -> np.ndarray:
        _scalar(s, "s")
        return self.matrix

    def inverse(self, s) -> np.ndarray:
        return _inverse(self.k(s), "ConstantK")

    def s_breakpoints(self) -> np.ndarray:
        points = []
        if np.linalg.matrix_rank(self.matrix) < len(self.matrix):
            points.append(0.0)
        return np.unique(np.asarray(points, dtype=float))


@dataclass(frozen=True)
class PolePolynomialK:
    """One bare pole plus a polynomial background, ``K = g g^T/(m^2-s) + P(s)``.

    The exact bare pole ``s == mass**2`` is rejected rather than silently
    evaluated; its limit is a separate formulation.
    """

    mass: float
    couplings: np.ndarray
    coefficients: tuple = ()
    s_ref: float = 0.0
    scale_squared: float = 1.0

    def __post_init__(self) -> None:
        mass = _scalar(self.mass, "PolePolynomialK.mass", positive=True)
        s_ref = _scalar(self.s_ref, "PolePolynomialK.s_ref")
        scale = _scalar(self.scale_squared, "PolePolynomialK.scale_squared", positive=True)
        if np.iscomplexobj(self.couplings):
            raise ValueError("PolePolynomialK.couplings must be real")
        g = np.array(self.couplings, dtype=float, copy=True)
        if g.ndim != 1 or not len(g) or not np.all(np.isfinite(g)):
            raise ValueError("PolePolynomialK.couplings must be a finite real vector")
        g.setflags(write=False)
        coefficients = tuple(
            _symmetric(c, f"PolePolynomialK.coefficients[{i}]")
            for i, c in enumerate(self.coefficients)
        )
        for c in coefficients:
            if c.shape != (len(g), len(g)):
                raise ValueError("PolePolynomialK coefficient dimensions differ from couplings")
        object.__setattr__(self, "mass", mass)
        object.__setattr__(self, "s_ref", s_ref)
        object.__setattr__(self, "scale_squared", scale)
        object.__setattr__(self, "couplings", g)
        object.__setattr__(self, "coefficients", coefficients)

    def k(self, s) -> np.ndarray:
        """``K(s)``; the exact bare pole is refused, complex ``s`` is allowed.

        A real ``s`` yields a real array: the legacy kernels (``t_from_k``,
        ``symmetric_matrix``) reject complex dtypes rather than silently
        discarding an imaginary part, so the dtype must reflect the input.
        """
        real_input = not np.iscomplexobj(s)
        s = complex(s)
        if not np.isfinite(s):
            raise ValueError(f"finite s required, got {s!r}")
        if s == self.mass**2:
            raise ValueError("exact bare K pole: use a formulation with an explicit limit")
        x = (s - self.s_ref) / self.scale_squared
        value = np.outer(self.couplings, self.couplings) / (self.mass**2 - s)
        for n, coefficient in enumerate(self.coefficients):
            value = value + coefficient * x**n
        if real_input:
            return np.asarray(value.real)
        return np.asarray(value)

    def inverse(self, s) -> np.ndarray:
        """``K(s)^{-1}``; a singular ``K`` raises instead of being regularised.

        With one coefficient, the background is a constant matrix ``C`` and
        ``K = C + g g.T / (m²-s)``.  For invertible ``C``, use the
        Sherman--Morrison formula rather than forming the large rank-one term
        close to the bare pole.  The exact bare pole remains outside ``k`` and
        ``inverse``'s domain, and the real K zero at
        ``s = m² + g.T C⁻¹ g`` remains a true singularity of the inverse.
        Singular constant backgrounds and non-constant backgrounds retain the
        direct strict inverse path.
        """
        real_input = not np.iscomplexobj(s)
        s = complex(s)
        if not np.isfinite(s):
            raise ValueError(f"finite s required, got {s!r}")
        if s == self.mass**2:
            raise ValueError("exact bare K pole: use a formulation with an explicit limit")

        if len(self.coefficients) == 1:
            background = self.coefficients[0]
            dimension = len(self.couplings)
            if np.linalg.matrix_rank(background) == dimension:
                identity = np.eye(dimension)
                background_inverse = np.linalg.solve(background, identity)
                pole_direction = np.linalg.solve(background, self.couplings)
                coupling_norm = float(self.couplings @ pole_direction)
                zero_at = self.mass**2 + coupling_norm
                if s == zero_at:
                    raise ValueError(
                        "PolePolynomialK: singular K has no inverse; use a formulation with an explicit limit"
                    )
                denominator = self.mass**2 - s + coupling_norm
                if denominator == 0.0:
                    raise ValueError(
                        "PolePolynomialK: singular K has no inverse; use a formulation with an explicit limit"
                    )
                inverse = background_inverse - np.outer(pole_direction, pole_direction) / denominator
                if real_input:
                    return np.asarray(inverse.real)
                return np.asarray(inverse)

        return _inverse(self.k(s), "PolePolynomialK")

    def s_breakpoints(self) -> np.ndarray:
        """Known real singular points of the inverse form (bare pole and K zero).

        For an invertible constant background ``C``,
        ``det K = det C * (m^2 - s + g^T C^{-1} g) / (m^2 - s)``, so the only
        possible K zero is known in closed form.  Higher backgrounds report the
        bare pole only and rely on interval splitting elsewhere.
        """
        points = [self.mass**2]
        if len(self.coefficients) == 1:
            coefficient = self.coefficients[0]
            if np.linalg.matrix_rank(coefficient) == len(coefficient):
                points.append(
                    self.mass**2
                    + float(self.couplings @ np.linalg.solve(coefficient, self.couplings))
                )
        return np.unique(np.asarray(points, dtype=float))


def _inverse(matrix: np.ndarray, name: str) -> np.ndarray:
    """Inverse of a real symmetric matrix, refusing singular input."""
    values = np.asarray(matrix)
    if values.shape[0] != values.shape[1]:
        raise ValueError(f"{name} requires a square matrix")
    if np.iscomplexobj(values):
        singular = np.linalg.matrix_rank(values) < values.shape[0]
    else:
        singular = np.linalg.matrix_rank(values.astype(float)) < values.shape[0]
    if singular:
        raise ValueError(f"{name}: singular K has no inverse; use a formulation that avoids it")
    return np.linalg.solve(values, np.eye(values.shape[0]))


# The two long-standing models are put in the registry too, so ``registered()`` is the single
# place to look and a caller never has to know which module a model happens to live in.
register("constant-k", description="A constant real symmetric K",
         required=("matrix",), replace=True)(ConstantK)
register("pole-polynomial-k",
         description="Bare pole plus polynomial background: K = g g^T/(m^2-s) + sum C_n x^n",
         required=("mass", "couplings"), replace=True)(PolePolynomialK)
