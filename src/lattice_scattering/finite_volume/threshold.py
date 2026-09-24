"""Center-of-mass momentum factors for threshold-weighted JLS blocks.

For ``weighting="threshold"``, the assembly applies ``(2k_i)**(-ell_i)``
on each side of the supplied reduced inverse. ``weighting="scale"`` uses
a different normalization. The same numerical blocks must not be reused
across these choices without an explicit transformation; see docs/conventions.md.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np

__all__ = [
    "threshold_factor",
    "threshold_scaled_inverse",
    "two_k",
]

#: ``2 k_i`` below this magnitude is treated as a threshold singularity rather
#: than a number, because ``k^{-ell}`` overflows and the finite-volume condition
#: is not defined there.
_MIN_TWO_K = 1e-12


def two_k(k_squared_at2) -> np.ndarray:
    """``2 k_i`` from each channel's ``k^2`` in lattice units.

    Below threshold ``k^2 < 0``; the real-axis condition needs the positive
    branch, so a negative ``k^2`` is reported as a negative ``2k`` (``2i|k|``
    would be the sheet-continued value, which this real-axis helper refuses to
    invent).  Callers are expected to have checked their window.
    """
    values = np.asarray(k_squared_at2, dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError(f"a finite 1-d array of k^2 per channel is required, got {k_squared_at2!r}")
    return np.sign(values) * 2.0 * np.sqrt(np.abs(values))


def threshold_factor(k_squared_at2, ell) -> np.ndarray:
    """``(2 k_i)^{-ell_i}`` per channel.

    ``ell`` may be a scalar (uniform partial wave) or one entry per channel.
    A channel at its own threshold (``k = 0``) makes the factor divergent for
    ``ell > 0``; that is a genuine singularity of the parameterisation, so it is
    reported rather than regularised silently.
    """
    k2 = np.asarray(k_squared_at2, dtype=float)
    if k2.ndim != 1 or not len(k2):
        raise ValueError(f"one k^2 per channel required, got {k_squared_at2!r}")
    ells = np.asarray(ell)
    if ells.ndim == 0:
        ells = np.full(k2.shape, ells)
    if ells.shape != k2.shape:
        raise ValueError(f"ell must be scalar or per-channel ({k2.shape}), got {ells.shape}")
    # Validate the ORIGINAL values: comparing ``ints == round(ints)`` after casting
    # to an integer dtype would silently accept 1.5 as 1 and produce a wrong
    # threshold factor.  Float ell values only need to be integral.
    as_float = np.asarray(ells, dtype=float)
    if not np.all(np.isfinite(as_float)):
        raise ValueError(f"finite ell required, got {ell!r}")
    if not np.all(as_float == np.round(as_float)) or np.any(as_float < 0):
        raise ValueError(f"non-negative integer ell required, got {ell!r}")
    ells = np.round(as_float).astype(int)

    two_k_values = two_k(k2)
    factors = np.ones_like(two_k_values)
    for index, (value, power) in enumerate(zip(two_k_values, ells)):
        power = int(round(float(power)))
        if power == 0:
            continue
        if abs(value) < _MIN_TWO_K:
            raise ValueError(
                f"channel {index} is at (or numerically on) its threshold with ell={power}: "
                "(2k)^-ell diverges there and the finite-volume condition is not defined"
            )
        factors[index] = value ** (-power)
    return factors


def threshold_scaled_inverse(
    inverse_blocks,
    k_squared_at2,
    ell,
    *,
    channels: int = 1,
    labels=None,
) -> np.ndarray:
    """Apply ``(2k_i)^{-ell_i} . Kinv . (2k_j)^{-ell_j}`` to a block set.

    ``inverse_blocks`` maps ``twice_J -> (labels, matrix)`` exactly as produced by
    :func:`lattice_scattering.finite_volume.jls_matrix._j_matrices`, i.e. each
    matrix is over channel-major ``(channel, sector)`` entries.  The scaling is
    applied per entry using that entry's own channel and ``ell``, which is what
    makes a mixed ``S+D`` channel basis correct: an ``ell=2`` column is scaled by
    ``(2k)^{-2}`` while an ``ell=0`` column is untouched.

    Returns a new mapping of the same shape with scaled matrices; the input is not
    modified.
    """
    factors = threshold_factor(k_squared_at2, ell)
    n_channels = int(channels)
    if factors.shape != (n_channels,):
        raise ValueError(
            f"expected one (2k)^-ell per channel ({n_channels}), got {factors.shape}"
        )

    scaled: dict[int, tuple[object, np.ndarray]] = {}
    for twice_J, (block_labels, matrix) in inverse_blocks.items():
        value = np.asarray(matrix, dtype=float)
        if value.ndim != 2 or value.shape[0] != value.shape[1]:
            raise ValueError(f"twice_J={twice_J}: square block required, got {value.shape}")
        if len(block_labels) != value.shape[0]:
            raise ValueError(
                f"twice_J={twice_J}: {len(block_labels)} labels for a {value.shape} block"
            )
        weights = np.empty(value.shape[0], dtype=float)
        for position, (channel, _sector_index) in enumerate(block_labels):
            if isinstance(channel, bool) or not isinstance(channel, Integral):
                raise ValueError(
                    f"twice_J={twice_J}: labels must be (channel, sector) pairs, got {channel!r}"
                )
            if not 0 <= int(channel) < n_channels:
                raise ValueError(f"twice_J={twice_J}: channel {channel} outside 0..{n_channels - 1}")
            weights[position] = factors[int(channel)]
        scaled[twice_J] = (block_labels, weights[:, None] * value * weights[None, :])
    return scaled
