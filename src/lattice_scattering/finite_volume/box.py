"""General finite-volume box matrices for arbitrary total momentum ``d`` (v2 P3).

Two entry points:

``orbital_box``
    The orbital (spinless) ``ell``-block matrix in the real-loop spherical
    basis, with the exact normalisation and phase convention of the legacy
    ``finite_volume.partial_waves.orbital_box``: each ``ell`` block is stored
    with ``m`` ascending, the blocks are interleaved in the input order of
    ``ells``, and the whole matrix is multiplied by
    ``sqrt(4*pi) / (gamma * pi**1.5)`` at the end.

``mixed_box``
    The ``(ell, 2S)`` direct sum: for every ``twice_S`` present the orbital block
    of its degrees is ``kron``-ed with the identity of the spin ``(2S+1)`` space
    and placed at the sector blocks.  Sectors of different ``twice_S`` never
    mix.

Generalisation over the legacy kernels: ``d`` is arbitrary (the wide-domain v2
:func:`~lattice_scattering.finite_volume.zeta.harmonic_zeta_wide` is used) and
the orbital degrees may reach ``MAX_ELL`` (8).  Only the algebraically shared
kernel ``symmetry.harmonic_products.conjugate_product_coefficients`` is imported
at runtime, as allowed by the v2 interface contract.

``harmonic_zeta_wide`` is memoised here on its full numeric key: a single
quantization scan evaluates the same few ``(q2, ell, d, gamma, alpha, ...)``
points hundreds of times, and the kernel is a pure function of them.  The legacy
kernel ``higher_zeta.harmonic_zeta`` is itself ``lru_cache``-decorated, so this
matches the reference implementation's behaviour.
"""

from __future__ import annotations

from functools import lru_cache
from numbers import Integral

import numpy as np

from lattice_scattering.symmetry.harmonic_products import conjugate_product_coefficients

from .zeta import MAX_ELL, harmonic_zeta_wide  # noqa: F401  (MAX_ELL re-exported)

__all__ = ["mixed_box", "orbital_box"]

#: Memoisation budget of :func:`_zeta_values` (entries, not bytes).
_ZETA_CACHE_ENTRIES = 8192


def _degree(value) -> int:
    """One orbital degree as an integer in ``0..MAX_ELL``."""
    from .zeta import MAX_ELL as limit

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"orbital degrees must be integers in 0..{limit}, got {value!r}")
    out = int(value)
    if not 0 <= out <= limit:
        raise ValueError(f"orbital degrees must be integers in 0..{limit}, got {out}")
    return out


def _waves(ells) -> tuple[int, ...]:
    """Distinct orbital degrees in the caller's order."""
    waves = tuple(_degree(ell) for ell in ells)
    if not waves:
        raise ValueError("at least one orbital degree is required")
    if len(set(waves)) != len(waves):
        raise ValueError(f"distinct orbital degrees required, got {waves}")
    return waves


def _dimension(value, *, name: str = "max_dimension") -> int:
    """A positive integer basis budget."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _spin(value) -> int:
    """One nonnegative integer ``twice_S``."""
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(f"twice_S must be a nonnegative integer, got {value!r}")
    return int(value)


def _sectors(sectors, max_dimension: int) -> tuple[tuple[tuple[int, int], ...], tuple[int, ...]]:
    """Validate ``(ell, twice_S)`` sectors and return them with their sizes.

    The dimension budget is checked *before* any geometry work, so an oversized
    request never pays for a Zeta evaluation.
    """
    pairs: list[tuple[int, int]] = []
    for sector in sectors:
        if not isinstance(sector, (tuple, list)) or len(sector) != 2:
            raise ValueError(f"sectors must be (ell, twice_S) pairs, got {sector!r}")
        pairs.append((_degree(sector[0]), _spin(sector[1])))
    if not pairs:
        raise ValueError("at least one (ell, twice_S) sector is required")
    if len(set(pairs)) != len(pairs):
        raise ValueError(f"distinct sectors required, got {tuple(pairs)}")
    sizes = tuple((2 * ell + 1) * (twice_S + 1) for ell, twice_S in pairs)
    if sum(sizes) > max_dimension:
        raise ValueError(
            f"mixed basis of dimension {sum(sizes)} exceeds the budget {max_dimension}"
        )
    return tuple(pairs), sizes


def _zeta_values(q2, ell, d, gamma, alpha, settings_key) -> np.ndarray:
    """Memoised ``harmonic_zeta_wide(...).values`` on the full numeric key."""
    return harmonic_zeta_wide(q2, ell, d=d, gamma=gamma, alpha=alpha, **dict(settings_key)).values


_zeta_values_cached = lru_cache(maxsize=_ZETA_CACHE_ENTRIES)(_zeta_values)


def _settings_key(settings: dict) -> tuple[tuple[str, object], ...]:
    """Hashable, sorted view of the pass-through Zeta settings."""
    items = []
    for key in sorted(settings):
        value = settings[key]
        try:
            hash(value)
        except TypeError:
            value = repr(value)
        items.append((key, value))
    return tuple(items)


def orbital_box(q2, ells, *, d, gamma, alpha, **settings) -> np.ndarray:
    """Orbital box matrix, ``sum(2*ell+1)`` dimensional, ``m`` ascending.

    ``ells`` is an ordered sequence of distinct degrees.  For every allowed
    ``L`` the harmonic sum ``Z_LM`` is rephased by ``q2**((ell+ell'-L)/2)`` and
    contracted with the exact product coefficients of
    ``conj(Y_ell,m) Y_ell',m'``.  The rephasing uses integer powers of the
    possibly negative ``q2``, so closed channels are covered.

    ``ell + ell' > MAX_ELL`` would need an unvalidated harmonic order and is
    rejected explicitly rather than silently truncated.
    """
    waves = _waves(ells)
    required = sorted(
        {
            degree
            for left in waves
            for right in waves
            for degree in range(abs(left - right), left + right + 1, 2)
        }
    )
    from .zeta import MAX_ELL as limit

    if required[-1] > limit:
        raise ValueError(
            f"orbital degrees require harmonic order L up to {required[-1]} for ell={waves}; "
            f"the v2 Zeta kernel is validated for L <= {limit} only"
        )
    settings_key = _settings_key(settings)
    zeta = {
        degree: _zeta_values_cached(q2, degree, d, gamma, alpha, settings_key)
        for degree in required
    }
    offsets = np.cumsum([0] + [2 * ell + 1 for ell in waves])
    result = np.zeros((int(offsets[-1]), int(offsets[-1])), dtype=complex)
    for i, left in enumerate(waves):
        for j, right in enumerate(waves):
            coefficients, labels = conjugate_product_coefficients(left, right)
            weights = np.array(
                [
                    q2 ** ((left + right - degree) // 2) * zeta[degree][degree + m]
                    if abs(left - right) <= degree <= left + right
                    and (left + right - degree) % 2 == 0
                    else 0.0
                    for degree, m in labels
                ]
            )
            result[offsets[i] : offsets[i + 1], offsets[j] : offsets[j + 1]] = np.einsum(
                "ijk,k->ij", coefficients, weights
            )
    result *= np.sqrt(4 * np.pi) / (gamma * np.pi**1.5)
    if not np.allclose(result, result.conj().T, atol=1e-10, rtol=1e-12):
        raise ArithmeticError(
            "non-Hermitian orbital box for "
            f"ell={waves}, d={tuple(d)}, q2={q2}, gamma={gamma}, alpha={alpha}"
        )
    return result


def mixed_box(q2, sectors, *, d, gamma, alpha, max_dimension: int = 256, **settings) -> np.ndarray:
    """Block-diagonal ``(ell, 2S)`` box with orbital mixing at fixed ``S``.

    ``sectors`` is an ordered sequence of distinct ``(ell, twice_S)`` pairs and
    is shared by every channel (per-channel sector content is a P4 concern).
    The ``twice_S`` blocks carry the identity of the spin space, so sectors of
    different total spin never mix.
    """
    budget = _dimension(max_dimension)
    pairs, sizes = _sectors(sectors, budget)
    offsets = np.cumsum([0] + list(sizes))
    result = np.zeros((int(offsets[-1]), int(offsets[-1])), dtype=complex)
    for spin in sorted({twice_S for _, twice_S in pairs}):
        indices = [index for index, (_, value) in enumerate(pairs) if value == spin]
        waves = [pairs[index][0] for index in indices]
        orbital = orbital_box(q2, waves, d=d, gamma=gamma, alpha=alpha, **settings)
        local = np.cumsum([0] + [2 * ell + 1 for ell in waves])
        for a, i in enumerate(indices):
            for b, k in enumerate(indices):
                result[offsets[i] : offsets[i + 1], offsets[k] : offsets[k + 1]] = np.kron(
                    orbital[local[a] : local[a + 1], local[b] : local[b + 1]], np.eye(spin + 1)
                )
    if not np.allclose(result, result.conj().T, atol=1e-10, rtol=1e-12):
        raise ArithmeticError(
            "non-Hermitian mixed box for "
            f"sectors={pairs}, d={tuple(d)}, q2={q2}, gamma={gamma}, alpha={alpha}"
        )
    return result
