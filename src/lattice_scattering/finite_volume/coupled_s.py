"""Channel-basis quantization conditions and JLS-delegated roots.

The direct ``simple_quantization_matrix`` and ``cm_quantization_matrix``
functions implement S-wave channel-basis conditions. ``coupled_s_roots``
uses the JLS root scanner, including supported higher orbital waves when
a group and irrep are supplied. Each channel uses its own two-body kinematics.
"""

from __future__ import annotations

from numbers import Integral
import warnings

import numpy as np

from lattice_scattering.kinematics import LatticeFrame, two_body_point

from ..amplitudes.phase_space import cm_real_axis, effective_inverse_k, physical_rho
from .jls_matrix import WEIGHTING_THRESHOLD, quantization_roots
from .zeta import harmonic_zeta_wide

__all__ = [
    "cm_quantization_matrix",
    "coupled_s_roots",
    "phase_space_inverse",
    "simple_quantization_matrix",
    "scalar_box",
]

#: Pad fraction of a pole-free interval left out at each end before scanning.
_PAD_FRACTION = 1e-7
#: Tolerance for merging numerically distinct free poles and roots.
_DEDUPE_TOLERANCE = 1e-9


def _masses(channel_masses) -> np.ndarray:
    masses = np.asarray(channel_masses, dtype=float)
    if (
        masses.ndim != 2
        or masses.shape[1] != 2
        or not len(masses)
        or not np.all(np.isfinite(masses))
        or np.any(masses <= 0.0)
    ):
        raise ValueError(f"an (N, 2) array of positive finite mass pairs is required, got {channel_masses!r}")
    return masses


def _window(energy_window) -> tuple[float, float]:
    values = np.asarray(energy_window, dtype=float)
    if values.shape != (2,) or not np.all(np.isfinite(values)) or not 0.0 < values[0] < values[1]:
        raise ValueError(f"a positive ordered (low, high) window is required, got {energy_window!r}")
    return float(values[0]), float(values[1])


def scalar_box(q2, frame: LatticeFrame, gamma: float, alpha: float) -> float:
    """``b(E) = Z00(q^2; d, gamma, alpha) / (gamma pi^{3/2})`` (wide-domain Zeta)."""
    if frame.d == (0, 0, 0):
        # Rest frame: the boost is trivial, so gamma is exactly 1.
        value = harmonic_zeta_wide(float(q2), 0, d=(0, 0, 0), gamma=1.0, alpha=0.0)
    else:
        value = harmonic_zeta_wide(float(q2), 0, d=frame.d, gamma=float(gamma), alpha=float(alpha))
    return float(np.real(value.values[0])) / (float(gamma) * np.pi**1.5)


def _points(energy: float, masses: np.ndarray, frame: LatticeFrame):
    return [
        two_body_point(energy_lab_at=energy, mass1_at=m1, mass2_at=m2, frame=frame)
        for m1, m2 in masses
    ]


def _channel_data(energy: float, masses: np.ndarray, frame: LatticeFrame):
    """``(s, e_cm, b, k2)`` for one lab energy, per channel kinematics."""
    points = _points(energy, masses, frame)
    s_at2 = points[0].s_at2
    if any(abs(point.s_at2 - s_at2) > 1e-12 * max(1.0, abs(s_at2)) for point in points):
        raise ArithmeticError("channel invariant masses disagree: s is not channel independent")
    e_cm = float(np.sqrt(s_at2))
    boxes = np.array(
        [scalar_box(point.q_squared, frame, point.gamma, point.alpha) for point in points]
    )
    k_squared = np.array([point.k_squared_at2 for point in points], dtype=float)
    return s_at2, e_cm, boxes, k_squared


def _ell_vector(ell, channels: int, *, what: str = "ell") -> np.ndarray:
    """One non-negative integer ``ell`` per channel.

    This used to refuse ``ell > 0`` because the channel-basis kernel carried an S-wave box
    while the paper's Eq. 6 factor ``(2k)^-ell`` sits on the K side.  That kernel is gone --
    ``coupled_s_roots`` now delegates to the JLS kernel, whose box carries the matching
    ``ell`` blocks -- so the refusal was left behind contradicting the very delegation it
    pointed at, and ``coupled_s_roots(..., ell=1)`` still raised.  Only the validation
    remains.
    """
    values = np.asarray(ell if ell is not None else 0, dtype=float)
    if values.ndim == 0:
        values = np.full(int(channels), float(values))
    if values.shape != (int(channels),):
        raise ValueError(f"{what} must be a scalar or one entry per channel ({channels})")
    if not np.all(np.isfinite(values)) or not np.all(values == np.round(values)) or np.any(values < 0):
        raise ValueError(f"non-negative integer {what} per channel required, got {ell!r}")
    return np.round(values).astype(int)


def _s_wave_only(ells, what: str = "ell"):
    """Refuse ``ell > 0`` where the box really is the single S-wave ``Z00``.

    The two condition builders below pair the paper's ``(2k)^-ell`` K-side factor with an
    S-wave box, so ``ell > 0`` there would apply a plausible-looking but wrong normalisation.
    The *root finder* is not in that position any more -- it delegates to the JLS kernel,
    whose box carries the matching ``ell`` blocks -- so the guard lives here, on the
    condition, and not on the shared ``ell`` validation.
    """
    if np.any(np.asarray(ells) > 0):
        raise ValueError(
            f"this S-wave condition has a Z00 box, so {what}>0 is wrong here; use the JLS "
            "path (v2.finite_volume.row_matrix / quantization_roots with "
            "weighting='threshold'), whose box carries the matching ell blocks"
        )


def _threshold_scale(k_squared: np.ndarray, ells: np.ndarray) -> np.ndarray:
    """``Q_ii = (2k_i)^{ell_i}``, the inverse of the paper's ``(2k)^{-ell}`` factor.

    The paper writes ``[t^-1] = (2k)^-ell K^-1 (2k)^-ell + I``.  Since
    ``(2k)^-ell K^-1 (2k)^-ell = (Q K Q)^-1`` with ``Q = diag((2k)^ell)``, the
    threshold factor is equivalent to replacing ``K`` by ``Q K Q``, which is the
    form both quantization matrices below already consume.

    Note: with an S-wave box only ``ell = 0`` is reachable through the public
    entry points; this helper keeps the general form so the derivation stays
    checkable against the JLS path.
    """
    two_k = np.sign(k_squared) * 2.0 * np.sqrt(np.abs(k_squared))
    scale = np.ones_like(two_k)
    for index, (value, power) in enumerate(zip(two_k, ells)):
        if power == 0:
            continue
        if abs(value) < 1e-12:
            raise ValueError(
                f"channel {index} is at its threshold with ell={power}: (2k)^-ell diverges"
            )
        scale[index] = value**power
    return scale


def simple_quantization_matrix(
    energy: float, masses, frame: LatticeFrame, model, ell=None
) -> np.ndarray:
    """``(E_cm L / 4 pi) K_th(s)^{-1} - diag(b)`` for the simple phase space.

    ``ell`` applies the paper's Eq. 6 threshold factor via
    ``K_th^{-1} = Q^-1 K^-1 Q^-1`` (``ell=0`` reproduces the S-wave condition).
    """
    masses = _masses(masses)
    s_at2, e_cm, boxes, k_squared = _channel_data(energy, masses, frame)
    inverse = effective_inverse_k(model, masses, "simple", s_at2)
    ells = _ell_vector(ell, len(masses))
    _s_wave_only(ells)
    if np.any(ells):
        q = _threshold_scale(k_squared, ells)
        inverse = inverse / q[:, None] / q[None, :]
    return (e_cm * frame.length_at / (4.0 * np.pi)) * inverse - np.diag(boxes)


def cm_quantization_matrix(
    energy: float, masses, frame: LatticeFrame, model, subtractions=(), ell=None
) -> np.ndarray:
    """K-matrix form of the Chew-Mandelstam condition (finite at K zeros).

    ``ell`` applies the paper's Eq. 6 factor by replacing ``K`` with
    ``Q K Q`` (``ell=0`` reproduces the S-wave condition bit-for-bit).
    """
    masses = _masses(masses)
    subs = tuple(subtractions) if len(subtractions) else (0.0,) * len(masses)
    if len(subs) != len(masses):
        raise ValueError("one subtraction per channel required")
    s_at2, e_cm, boxes, k_squared = _channel_data(energy, masses, frame)
    k_matrix = np.asarray(model.k(s_at2))
    if k_matrix.shape != (len(masses),) * 2:
        raise ValueError("model K has the wrong channel dimension")
    if np.iscomplexobj(k_matrix):
        # The real-axis condition needs a real K; refuse rather than silently
        # discarding an imaginary part.
        if not np.allclose(k_matrix.imag, 0.0, atol=1e-12 * max(1.0, np.max(np.abs(k_matrix)))):
            raise ValueError("the CM K-matrix form requires a real K on the real axis")
        k_matrix = k_matrix.real
    ells = _ell_vector(ell, len(masses))
    _s_wave_only(ells)
    if np.any(ells):
        q = _threshold_scale(k_squared, ells)
        k_matrix = q[:, None] * k_matrix * q[None, :]
    diagonal = np.array(
        [
            cm_real_axis(s_at2, m1, m2, subtraction=sub)
            + 1j * physical_rho(s_at2, m1, m2)
            - box * 4.0 * np.pi / (e_cm * frame.length_at)
            for (m1, m2), sub, box in zip(masses, subs, boxes)
        ]
    )
    if not np.allclose(diagonal.imag, 0.0, atol=1e-9 * max(1.0, np.max(np.abs(diagonal.real)))):
        raise ArithmeticError("the CM effective term is not real on the real axis")
    return np.eye(len(masses)) + k_matrix @ np.diag(diagonal.real)


def phase_space_inverse(energy: float, masses, frame: LatticeFrame, model, phase_space: str, subtractions=()):
    """Effective channel-basis inverse ``K`` at one lab energy (simple or CM)."""
    masses = _masses(masses)
    s_at2, _, _, _ = _channel_data(energy, masses, frame)
    return effective_inverse_k(model, masses, phase_space, s_at2, subtractions=subtractions)


def coupled_s_roots(
    frame: LatticeFrame,
    masses,
    model,
    phase_space: str,
    energy_window,
    *,
    subtractions=(),
    samples: int = 200,
    xtol: float = 1e-13,
    max_free_vectors: int = 400000,
    ell=None,
    group=None,
    irrep=None,
    row: int = 0,
    root_method: str = "eigenvalues",
    subdivisions: int = 1,
    residual_tol: float = 1e-10,
    lhc_domain=None,
) -> tuple[float, ...]:
    """Quantization levels in the channel basis, evaluated by the JLS kernel.

    This module used to carry a second, independent kernel: it formed the channel-basis
    condition and scanned it itself.  That kernel was S-wave only, and everything it could
    do the JLS kernel does too -- ``tests/test_v2_kernel_equivalence.py`` shows the two
    agreeing to machine precision in the rest frame, in moving frames, at unequal masses,
    across two channels (including a root below the second threshold) and for
    Chew-Mandelstam with and without a subtraction.  Keeping two kernels meant two places
    to fix every defect, and the interval-collapse bug that silently dropped a level lived
    in the deleted one while the JLS path never had it.

    The reduced inverse is matched by ``R = (sqrt(s) L / 4pi) K^-1`` under the
    **threshold** weighting; the same quantity under the default ``scale`` weighting would
    be ``(sqrt(s) / 2) K^-1``, and mixing the two is a silent 3.3e-2 error in the level.
    Hence the weighting is named rather than left to the default.

    Every channel carries ``twice_S = 0``, because the channel basis never carried a total
    spin.  A non-zero-spin channel must go through
    :func:`~lattice_scattering.finite_volume.jls_matrix.quantization_roots` directly,
    which takes explicit ``(ell, twice_S)`` sectors.  ``group`` and ``irrep`` are required
    once more than one sector survives, since the levels are then irrep dependent.
    """
    if phase_space not in ("simple", "chew-mandelstam"):
        raise ValueError(f"unknown phase space {phase_space!r}")
    masses = _masses(masses)
    channels = len(masses)
    ells = _ell_vector(ell, channels)
    sectors = tuple(sorted({(int(value), 0) for value in ells}))
    subs = tuple(subtractions) if len(subtractions) else (0.0,) * channels
    if len(subs) != channels:
        raise ValueError("one subtraction per channel required")
    if group is None or irrep is None:
        if len(sectors) > 1:
            raise ValueError(
                f"sectors {sectors} need an irrep: pass group= and irrep= so the levels are "
                f"projected onto one row, otherwise the determinant mixes irreps"
            )
        from ..symmetry.groups import double_cover, little_group

        group = double_cover(little_group(frame.d))
        irrep = "A1g" if frame.d == (0, 0, 0) else "A1"

    # ``effective_inverse_k`` with chew-mandelstam already returns K^-1 + I(s) + i rho(s),
    # so R carries the self-energy and no CM term is added downstream.
    def reduced_inverse(s_at2):
        inverse = effective_inverse_k(model, masses, phase_space, s_at2, subtractions=subs)
        return (np.sqrt(s_at2) * frame.length_at / (4.0 * np.pi)) * inverse

    return quantization_roots(
        energy_window,
        frame,
        masses,
        sectors,
        {2 * sector_ell: reduced_inverse for sector_ell, _ in sectors},
        group=group,
        irrep=irrep,
        row=row,
        samples=samples,
        root_method=root_method,
        subdivisions=subdivisions,
        residual_tol=residual_tol,
        lhc_domain=lhc_domain,
        xtol=xtol,
        max_free_vectors=max_free_vectors,
        weighting=WEIGHTING_THRESHOLD,
        breakpoints_at2=(model.s_breakpoints() if hasattr(model, "s_breakpoints") else None),
    )
