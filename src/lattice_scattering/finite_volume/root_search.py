"""Finite-resolution Hermitian eigenvalue root search, not a completeness proof."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .zeta import FreePoleError


class SpectralSearchError(ValueError):
    def __init__(self, message, *, energy=None, interval=None, kind="refinement"):
        super().__init__(message)
        self.energy, self.interval, self.kind = energy, interval, kind


def hermitian_eigenvalues(matrix):
    """Validate before eigh, which otherwise silently uses only one triangle."""
    value = np.asarray(matrix, dtype=complex)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or not value.shape[0]:
        raise ValueError("nonempty square quantization matrix required")
    if not np.all(np.isfinite(value)):
        raise ValueError("non-finite quantization matrix")
    scale = max(1.0, float(np.max(np.abs(value))))
    if np.max(np.abs(value - value.conj().T)) > 1e-10 + 1e-12 * scale:
        raise ValueError("non-Hermitian quantization matrix")
    # Scale each term before adding to avoid overflowing a finite diagonal.
    values = np.linalg.eigvalsh(0.5 * value + 0.5 * value.conj().T)
    if not np.all(np.isfinite(values)):
        raise ValueError("non-finite quantization eigenvalues")
    return values


def matrix_root_residual(matrix):
    """min|lambda| / max(1, max|lambda|), for the supplied matrix convention.

    This is a backward residual, not an energy error or a completeness test.
    The absolute floor is meaningful only in the caller's matrix normalization.
    """
    values = hermitian_eigenvalues(matrix)
    return float(np.min(np.abs(values)) / max(1.0, float(np.max(np.abs(values)))))


def _extremum(function, left, right, xtol):
    """Golden-section minimization with an absolute energy stopping rule.

    scipy's bounded minimizer also stops at sqrt(eps)*|E|, too coarse for
    near-degenerate levels at the root scanner's usual 1e-13 tolerance.
    """
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = float(left), float(right)
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc, fd = function(c), function(d)
    for _ in range(256):
        if b - a <= max(xtol, 8 * np.spacing(max(1.0, abs(a), abs(b)))):
            return c if fc <= fd else d
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = function(c)
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = function(d)
    raise SpectralSearchError("extremum refinement did not converge", interval=(left, right))


def spectral_roots(matrix_at, left, right, *, samples, subdivisions, xtol, residual_tol):
    """Follow ordered eigenvalues and refine sampled extrema and sign brackets.

    Ordered eigenvalues of a continuous Hermitian matrix are continuous even
    at degeneracies; no ambiguous eigenvector assignment is needed. Midpoint
    subdivision resolves extra structure. Each sampled local extremum triggers
    adaptive scalar refinement; a same-sign endpoint pair may hide two roots.
    Unseen oscillations or extrema narrower than the grid remain unresolved.
    """
    cache = {}
    dimension = None

    def eig(energy):
        nonlocal dimension
        energy = float(energy)
        if energy not in cache:
            try:
                values = hermitian_eigenvalues(matrix_at(energy))
            except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
                # Preserve the public scanner's structured errors and pole policy.
                if hasattr(exc, "kind") or isinstance(exc, FreePoleError):
                    raise
                raise SpectralSearchError(str(exc), energy=energy, kind="evaluation") from exc
            if dimension is None:
                dimension = len(values)
            if len(values) != dimension:
                raise SpectralSearchError("matrix dimension changed during scan", energy=energy)
            cache[energy] = values
        return cache[energy]

    def residual(energy):
        values = eig(energy)
        return np.min(np.abs(values)) / max(1.0, float(np.max(np.abs(values))))

    def accept(energy):
        value = residual(energy)
        if value > residual_tol:
            raise SpectralSearchError(
                f"candidate root residual {value:.3e} exceeds {residual_tol:.3e}; "
                "tighten xtol or declare a missing model breakpoint",
                energy=energy, kind="residual",
            )
        roots.append(float(energy))

    grid = np.linspace(left, right, (samples - 1) * 2**subdivisions + 1)
    values = np.array([eig(x) for x in grid])
    roots = []
    for branch in range(values.shape[1]):
        function = lambda energy: float(eig(energy)[branch])
        nodes = list(zip(grid, values[:, branch]))
        # Refine a strict minimum or maximum, not a flat small constant.
        for i in range(1, len(grid) - 1):
            a, b, c = values[i - 1:i + 2, branch]
            if b < min(a, c) or b > max(a, c):
                direction = 1 if b < min(a, c) else -1
                point = _extremum(lambda x: direction * function(x), grid[i-1], grid[i+1], xtol)
                value = function(point)
                nodes.append((point, value))
                # A tangent candidate must turn toward zero from both sides.
                if (a > 0 and c > 0 and 0 <= value < min(a, c)) or (
                    a < 0 and c < 0 and max(a, c) < value <= 0
                ):
                    scale = max(1.0, float(np.max(np.abs(eig(point)))))
                    if abs(value) <= residual_tol * scale:
                        accept(point)
        nodes.sort()
        for energy, value in nodes:
            if value == 0:
                accept(energy)
        for (a, fa), (b, fb) in zip(nodes[:-1], nodes[1:]):
            if fa == 0 or fb == 0 or np.signbit(fa) == np.signbit(fb):
                continue
            try:
                root = brentq(function, a, b, xtol=xtol, rtol=4*np.finfo(float).eps)
                if residual(root) > residual_tol:
                    # A steep physical eigenvalue can need a tighter energy
                    # tolerance than the requested grid/root merge tolerance.
                    # Retry without an absolute floor before rejecting it.
                    root = brentq(function, a, b, xtol=np.nextafter(0., 1.),
                                  rtol=4*np.finfo(float).eps, maxiter=200)
            except (ValueError, RuntimeError) as exc:
                if hasattr(exc, "kind") or isinstance(exc, FreePoleError):
                    raise
                raise SpectralSearchError(str(exc), interval=(a, b)) from exc
            accept(root)
    return roots
