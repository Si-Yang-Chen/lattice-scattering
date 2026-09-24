"""JLS quantization assembly and irrep-row projection.

Supports caller-supplied reduced-inverse total-J blocks, channel-specific
kinematics, and arbitrary supported little groups. See docs/conventions.md
for sector ordering, weighting conventions, and root-scan limits."""

from __future__ import annotations

from numbers import Integral
from typing import Mapping, Protocol, runtime_checkable

import numpy as np

from lattice_scattering.kinematics import LatticeFrame, two_body_point
from lattice_scattering.kinematics.free import free_two_body_states
from lattice_scattering.symmetry.angular_momentum import coupled_j_basis

from ..amplitudes.phase_space import cm_real_axis, physical_rho
from ..core.jls import Sector
from ..symmetry.groups import subduce
from .box import _dimension, _sectors, mixed_box
from .jls_channels import (
    assemble_active_inverse,
    channel_row_basis,
    normalise_channel_layout,
    parity_allowed,
)
from .threshold import threshold_factor, two_k
from .root_search import SpectralSearchError, spectral_roots
from .zeta import (
    ALPHA_MAX,
    ALPHA_MIN,
    GAMMA_MAX,
    GAMMA_MIN,
    MAX_D2,
    Q2_MAX,
    Q2_MIN,
    SPLIT_MAX,
    SPLIT_MIN,
    FreePoleError,
    POLE_POLICIES,
)

__all__ = [
    "JMatrixModel",
    "RootScanError",
    "assemble_inverse",
    "quantization_roots",
    "row_matrix",
]

#: Phase-space conventions for the box side of the determinant.
PHASE_SPACE_SIMPLE = "simple"
PHASE_SPACE_CHEW_MANDELSTAM = "chew-mandelstam"

#: Weighting conventions for the reduced inverse.
#:
#: ``"scale"``      ``scale**(ell_a + ell_b + 1)`` with ``scale = L/(2 pi)``:
#:                  a constant, the legacy ``mixed_spin`` convention, and the
#:                  validated default (2++ mock, JLS/jls_matrix regression).
#: ``"threshold"``  ``(2 k_i)**-ell_i``: the paper's Eq. 6 threshold factor
#:                  (arXiv:2309.14071v1 Eq. 6, ``docs/conventions.md``).
#:                  Energy dependent, so it cannot be folded into a constant.
#:
#: They must not be swapped silently: at ``ell = 2`` the threshold convention
#: contributes ``k**-4``, which moves the roots rather than merely rescaling them.
WEIGHTING_SCALE = "scale"
WEIGHTING_THRESHOLD = "threshold"

#: Fraction of an adjacent interval kept away from exact free poles, whose
#: Zeta evaluation can lose Hermiticity through floating-point cancellation.
_FREE_POLE_PAD_FRACTION = 1e-7
#: ULPs used only to merge free-state energies that are numerically identical.
_FREE_POLE_MERGE_ULPS = 8.0
#: Scale-aware Hermiticity tolerance for the projected row, matching the legacy
#: ``mixed_spin``/``coupled_mixed_spin`` kernels (``atol=1e-10``, ``rtol=1e-12``).
#: The projected row is a cancellation of the ~1e36 box divergence close to a
#: free pole, so its small entries carry an absolute roundoff of order
#: ``eps * |full|``; a tighter absolute threshold rejects a correctly Hermitian
#: row and, because that rejection is roundoff dependent, made the root count
#: irreproducible across processes.
_HERMITIAN_ATOL = 1e-10
_HERMITIAN_RTOL = 1e-12


class RootScanError(ValueError):
    """A root scan domain check, evaluation, determinant, or refinement failed.

    The scanner used to turn evaluation failures into ``NaN`` values and skip
    the affected cells.  That made an unsupported kinematic point, a malformed
    amplitude block, or an unstable determinant indistinguishable from a cell
    containing no root.  This exception keeps the public failure in the
    ``ValueError`` family expected by the CLI while retaining the point and
    original exception for callers that need structured diagnostics.  A scan
    that returns normally is still limited by its finite sampling and local
    refinement; it does not certify the absence of unresolved narrow roots.

    Attributes
    ----------
    energy:
        Laboratory-frame energy at which evaluation failed, when known.
    cause:
        The original exception raised by a domain check, matrix or determinant
        evaluation, or root refiner, when one exists. It is also available
        through normal exception chaining (``__cause__``).
    kind:
        ``"domain"`` for the analytic scan-window preflight, ``"evaluation"``
        for a failed matrix call, ``"determinant"`` for a non-finite
        determinant result, ``"refinement"`` for a failed bracket, or
        ``"residual"`` when a spectral root fails its matrix residual gate.
    interval:
        Scan interval associated with a failure for which a single energy is
        not known, such as root-refinement setup or free-pole enumeration.
    """

    def __init__(
        self, message: str, *, energy=None, cause=None, kind: str = "evaluation", interval=None
    ):
        super().__init__(message)
        self.energy = None if energy is None else float(energy)
        self.cause = cause
        self.kind = str(kind)
        self.interval = None if interval is None else tuple(float(value) for value in interval)


@runtime_checkable
class JMatrixModel(Protocol):
    """Dynamical input: the reduced ``J`` blocks for one ``s_at2``.

    ``blocks(s)`` returns exactly the ``{twice_J: real symmetric ndarray}``
    mapping required by the call: every triangle-rule value by default, or
    only the explicitly retained values when ``selected_j_sectors`` is set.
    Each block follows channel-major order restricted to its active incidences.
    A plain ``{twice_J: callable(s_at2) -> ndarray}`` mapping is accepted too.
    """

    def blocks(self, s: float) -> dict[int, np.ndarray]:
        ...


def _scalar(value, name: str) -> float:
    if isinstance(value, bool) or np.iscomplexobj(value) or np.ndim(value) != 0:
        raise ValueError(f"{name} must be a real scalar")
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def _window(energy_window) -> tuple[float, float]:
    if np.iscomplexobj(energy_window) or np.ndim(energy_window) != 1:
        raise ValueError("energy_window must be a real (low, high) pair")
    values = np.asarray(energy_window, dtype=float)
    if values.shape != (2,) or not np.all(np.isfinite(values)) or not 0 < values[0] < values[1]:
        raise ValueError(f"positive ordered finite energy_window required, got {energy_window!r}")
    return float(values[0]), float(values[1])


def _masses(channel_masses) -> np.ndarray:
    if np.iscomplexobj(channel_masses):
        raise ValueError("N positive finite mass pairs required")
    values = np.asarray(channel_masses, dtype=float)
    if (
        values.ndim != 2
        or values.shape[1] != 2
        or not len(values)
        or not np.all(np.isfinite(values))
        or np.any(values <= 0)
    ):
        raise ValueError(f"N positive finite mass pairs required, got {channel_masses!r}")
    return values


def _preflight_zeta_domain(low, high, frame, masses, *, split):
    """Check the continuous two-body Zeta domain across the complete scan window.

    For each channel, ``q2(s)`` has at most one stationary point for positive
    ``s``: ``s = |m1**2 - m2**2|``.  ``gamma`` is monotone in lab energy and
    ``alpha`` is monotone in ``s``.  Checking both window ends plus that
    stationary point therefore checks the full kinematic domain without
    relying on the root-scan grid.  This does not pre-evaluate arbitrary
    caller-supplied amplitude callbacks.
    """

    def fail(energy, reason):
        cause = ValueError(reason)
        raise RootScanError(
            "root scan domain preflight failed at laboratory energy "
            f"{float(energy):.17g}: {reason}",
            energy=energy,
            cause=cause,
            kind="domain",
            interval=(low, high),
        ) from cause

    d2 = sum(component * component for component in frame.d)
    if d2 > MAX_D2:
        fail(low, f"d={frame.d} outside |d|^2 <= {MAX_D2}")

    try:
        split_value = _scalar(split, "split")
    except (ArithmeticError, TypeError, ValueError) as exc:
        fail(low, str(exc))
    if not SPLIT_MIN <= split_value <= SPLIT_MAX:
        fail(low, f"split outside supported range [{SPLIT_MIN}, {SPLIT_MAX}]: {split_value!r}")

    momentum_squared = sum(component * component for component in frame.momentum_at)
    for channel, (mass1, mass2) in enumerate(masses):
        energies = {float(low), float(high)}
        # q2 reaches its minimum at |m1^2-m2^2|.  This can lie inside the
        # closed-channel part of the window even when both endpoints are in
        # range, so endpoint-only validation would be insufficient.
        try:
            stationary_s = abs(float(mass1) ** 2 - float(mass2) ** 2)
        except (ArithmeticError, TypeError, ValueError) as exc:
            fail(low, f"channel {channel} kinematic-domain calculation failed: {exc}")
        if stationary_s > 0.0:
            try:
                stationary_energy = float(np.sqrt(momentum_squared + stationary_s))
            except (ArithmeticError, TypeError, ValueError) as exc:
                fail(low, f"channel {channel} kinematic-domain calculation failed: {exc}")
            if low < stationary_energy < high:
                energies.add(stationary_energy)

        for energy in sorted(energies):
            try:
                point = two_body_point(
                    energy_lab_at=energy,
                    mass1_at=float(mass1),
                    mass2_at=float(mass2),
                    frame=frame,
                )
            except (ArithmeticError, TypeError, ValueError) as exc:
                fail(energy, f"channel {channel} two-body kinematics failed: {exc}")

            if np.sqrt(point.s_at2) <= abs(float(mass1) - float(mass2)):
                fail(
                    energy,
                    f"channel {channel} lies below the pseudothreshold "
                    f"(|m1-m2|={abs(float(mass1) - float(mass2)):.17g})",
                )

            for name, value, lower, upper in (
                ("q2", point.q_squared, Q2_MIN, Q2_MAX),
                ("gamma", point.gamma, GAMMA_MIN, GAMMA_MAX),
                ("alpha", point.alpha, ALPHA_MIN, ALPHA_MAX),
            ):
                if not lower <= value <= upper:
                    fail(
                        energy,
                        f"channel {channel} {name}={value:.17g} outside supported range "
                        f"[{lower}, {upper}]",
                    )
            if d2 == 0 and point.gamma != 1.0:
                fail(energy, f"rest frame requires gamma == 1, got {point.gamma:.17g}")


def _allowed_j(pairs: tuple[tuple[int, int], ...]) -> dict[int, list[int]]:
    """``twice_J -> indices`` of the sectors admitted by the triangle rule."""
    allowed: dict[int, list[int]] = {}
    for index, (ell, twice_S) in enumerate(pairs):
        for twice_J in range(abs(2 * ell - twice_S), 2 * ell + twice_S + 1, 2):
            allowed.setdefault(twice_J, []).append(index)
    return allowed


def _blocks(j_amplitudes, s_at2: float, expected: set[int]) -> dict[int, np.ndarray]:
    """Resolve the caller's dynamical input into ``{twice_J: ndarray}``."""
    if isinstance(j_amplitudes, Mapping):
        if any(isinstance(key, bool) or not isinstance(key, Integral) for key in j_amplitudes):
            raise ValueError("twice-J keys must be integers")
        if set(int(key) for key in j_amplitudes) != expected:
            raise ValueError(
                f"twice-J matrices required for exactly the active blocks: expected {sorted(expected)}, "
                f"got {sorted(int(key) for key in j_amplitudes)}"
            )
        if any(not callable(f) for f in j_amplitudes.values()):
            raise ValueError("every twice-J entry must be a callable of s_at2")
        return {int(key): f(s_at2) for key, f in j_amplitudes.items()}
    if hasattr(j_amplitudes, "blocks"):
        raw = j_amplitudes.blocks(s_at2)
        if (
            not isinstance(raw, Mapping)
            or any(isinstance(key, bool) or not isinstance(key, Integral) for key in raw)
            or set(int(key) for key in raw) != expected
        ):
            raise ValueError(
                "JMatrixModel.blocks must supply exactly the active twice-J keys "
                f"{sorted(expected)}"
            )
        return {int(key): value for key, value in raw.items()}
    raise ValueError("j_amplitudes must be a {twice_J: callable} mapping or a JMatrixModel")


def _j_matrices(
    j_matrices, allowed: dict[int, list[int]], labels: dict[int, list[tuple[int, int]]]
) -> dict[int, tuple[list[tuple[int, int]], np.ndarray]]:
    """Validate the resolved blocks against the channel-major sector layout."""
    if not isinstance(j_matrices, Mapping) or any(
        isinstance(key, bool) or not isinstance(key, Integral) for key in j_matrices
    ):
        raise ValueError("integer twice-J keys required")
    if set(int(key) for key in j_matrices) != set(allowed):
        raise ValueError(
            f"twice-J matrices required for exactly the active blocks: expected {sorted(allowed)}, "
            f"got {sorted(int(key) for key in j_matrices)}"
        )
    validated: dict[int, tuple[list[tuple[int, int]], np.ndarray]] = {}
    for twice_J in allowed:
        raw = j_matrices[twice_J]
        if not isinstance(raw, np.ndarray) and not isinstance(raw, (list, tuple)):
            raise ValueError(
                f"J matrix for twice_J={twice_J} must be a real array over the allowed "
                "active channel-major incidences"
            )
        if np.iscomplexobj(raw):
            raise ValueError("finite real symmetric J matrix required")
        value = np.asarray(raw, dtype=float)
        order = len(labels[twice_J])
        # The symmetry test needs a scale-aware floor.  With atol=0 alone it compares
        # elements against rtol * their own magnitude, so entries that are themselves at the
        # rounding level -- which happens wherever a matrix element cancels -- fail against
        # a difference of the same size.  That rejected physically symmetric inverses and
        # turned whole scan points into NaN, losing the level at 0.7120502: there R reaches
        # 3.5e+01 while the asymmetry is 1.0e-12, so rtol=1e-12 allowed 3.5e-11 but the test
        # demanded exactness against a self-referential tolerance.
        floor = 1e-10 * float(np.max(np.abs(value))) if value.size else 0.0
        if (
            value.shape != (order, order)
            or not np.all(np.isfinite(value))
            or not np.allclose(value, value.T, rtol=1e-12, atol=floor)
        ):
            raise ValueError(
                f"J matrix for twice_J={twice_J} must be real symmetric of order {order}"
            )
        validated[twice_J] = (labels[twice_J], value)
    return validated


def assemble_inverse(
    sectors=None,
    *,
    scale,
    j_matrices,
    channels: int | None = None,
    max_dimension: int = 256,
    weighting: str = WEIGHTING_SCALE,
    channel_k_squared_at2=None,
    channel_sectors=None,
    channel_intrinsic_parities=None,
    intrinsic_parity: int = 1,
    selected_j_sectors=None,
) -> np.ndarray:
    """Assemble the reduced inverse in the JLS basis.

    ``weighting="scale"`` (default) gives
    ``sum_J scale**(ell_a+ell_b+1) R^J_ab (U_aJ U_bJ^dagger)``, the validated
    legacy convention.  ``weighting="threshold"`` gives
    ``sum_J (2k_a)**-ell_a R^J_ab (2k_b)**-ell_b (U_aJ U_bJ^dagger)``, the paper's
    Eq. 6 factor; it then *requires* ``channel_k_squared_at2``.

    ``sectors`` is an ordered list of distinct ``(ell, twice_S)`` pairs shared
    by every channel.  ``channel_sectors`` optionally replaces it with one
    ordered sector list per channel; in that path ``channels`` is inferred from
    the list when omitted.  ``selected_j_sectors`` maps each retained
    ``twice_J`` to ``(channel, ell, twice_S)`` incidences.  Its entries are
    canonicalised to channel-major sector order before validating the supplied
    matrices.  A missing mapping selects every triangle-rule incidence.

    ``U_aJ`` is ``coupled_j_basis(2*ell_a, 2*S_a, 2*J)``.  Two nonzero
    amplitudes are allowed exactly when their physical parities
    ``eta_channel * (-1)**ell`` agree.
    """
    if weighting not in (WEIGHTING_SCALE, WEIGHTING_THRESHOLD):
        raise ValueError(
            f"weighting must be {WEIGHTING_SCALE!r} or {WEIGHTING_THRESHOLD!r}, got {weighting!r}"
        )
    if weighting == WEIGHTING_THRESHOLD and channel_k_squared_at2 is None:
        raise ValueError(
            "weighting='threshold' requires channel_k_squared_at2 (one k^2 per channel); "
            "the (2k)^-ell factor is energy dependent and cannot be a constant"
        )
    budget = _dimension(max_dimension)
    layout = normalise_channel_layout(
        sectors,
        channels=channels,
        channel_sectors=channel_sectors,
        channel_intrinsic_parities=channel_intrinsic_parities,
        intrinsic_parity=intrinsic_parity,
        selected_j_sectors=selected_j_sectors,
        max_dimension=budget,
    )
    scale = _scalar(scale, "scale")
    if scale <= 0:
        raise ValueError("positive finite scale required")
    labels = {twice_j: list(entries) for twice_j, entries in layout.active.items()}
    validated = _j_matrices(j_matrices, layout.active, labels)
    for twice_j, (entries, value) in validated.items():
        for a, left in enumerate(entries):
            for b, right in enumerate(entries):
                if value[a, b] != 0 and not parity_allowed(layout, left, right):
                    raise ValueError("J amplitudes may only mix equal physical parity sectors")

    per_channel = None
    if weighting == WEIGHTING_THRESHOLD:
        # ``two_k`` gives ``2 k`` per channel (NOT raised to any power): one
        # channel can carry several sectors with different ``ell`` (mixed S+D),
        # so the exponent is taken per *sector*, not per channel.
        per_channel = two_k(channel_k_squared_at2)
        if per_channel.shape != (layout.channels,):
            raise ValueError(
                f"one k^2 per channel required ({layout.channels}), got {per_channel.shape}"
            )
        # ``(2k)^-ell`` only needs a physical branch where ``ell > 0``: at
        # ``ell = 0`` the factor is exactly 1 whatever the sign of ``k^2``, so a
        # channel whose every sector has ``ell = 0`` is perfectly well defined
        # below its threshold and must not be refused.  Refusing it dropped
        # legitimate sub-threshold levels (a two-channel scan lost the root at
        # 0.692702418, below the second channel's 0.70 threshold).
        needs_branch = [False] * layout.channels
        for entries in layout.active.values():
            for channel, sector_index in entries:
                needs_branch[channel] = needs_branch[channel] or (
                    layout.channel_pairs[channel][sector_index][0] > 0
                )
        for index, value in enumerate(per_channel):
            if value < 0 and needs_branch[index]:
                raise ValueError(
                    f"channel {index} is below its threshold (k^2 < 0); the paper's real-axis "
                    "(2k)^-ell factor needs the physical branch"
                )

    return assemble_active_inverse(
        layout,
        validated,
        scale=scale,
        weighting=weighting,
        per_channel=per_channel,
        max_dimension=budget,
    )


def _row_basis(pairs, group, irrep, intrinsic_parity, row, budget) -> np.ndarray:
    """One row basis of ``subduce`` for the supplied sectors and group."""
    if type(intrinsic_parity) != int or intrinsic_parity not in (-1, 1):
        raise ValueError(f"intrinsic parity must be +1 or -1, got {intrinsic_parity!r}")
    sub = subduce(
        [Sector(0, ell, twice_S) for ell, twice_S in pairs],
        group,
        irrep,
        intrinsic_parity=intrinsic_parity,
        max_dimension=budget,
    )
    if isinstance(row, bool) or not isinstance(row, Integral) or not 0 <= int(row) < sub["dimension"]:
        raise ValueError(
            f"invalid row {row!r} for irrep {irrep!r} of dimension {sub['dimension']} "
            f"in group {getattr(group, 'name', group)!r}"
        )
    if not sub["multiplicity"]:
        raise ValueError(
            f"irrep {irrep!r} is absent from sectors={pairs} of group "
            f"{getattr(group, 'name', group)!r}"
        )
    return sub["row_bases"][int(row)]


def row_matrix(
    energy,
    frame,
    channel_masses,
    sectors,
    j_amplitudes,
    *,
    group,
    irrep,
    row: int = 0,
    intrinsic_parity: int = 1,
    max_dimension: int = 256,
    weighting: str = WEIGHTING_SCALE,
    channel_sectors=None,
    channel_intrinsic_parities=None,
    selected_j_sectors=None,
    **settings,
) -> np.ndarray:
    """One irrep-row projection of ``inverse - box`` for the supplied channels.

    Each channel uses its own :func:`two_body_point`; the box of channel ``c``
    is subtracted from that channel's diagonal block.  The ``V^dagger (...) V``
    projection uses the ``row`` basis of ``subduce(group, irrep)``, which spans
    the same isotypic row space as the legacy hand-written tables, so the
    *determinant* is convention independent.

    ``weighting`` selects the reduced-inverse convention: ``"scale"`` (default,
    legacy ``L/(2 pi)`` constant) or ``"threshold"`` (the paper's Eq. 6
    ``(2k)^-ell`` factor, needed for ``ell > 0`` amplitudes such as V6-V9).

    With ``selected_j_sectors``, only the listed channel/LS incidences enter
    the inverse and the projected basis is the intersection of that active
    total-J space with the requested irrep row. Supply exactly the selected
    ``twice_J`` blocks, with each block ordered by channel and declared sector.
    """
    if not isinstance(frame, LatticeFrame):
        raise ValueError(f"LatticeFrame required, got {type(frame).__name__}")
    budget = _dimension(max_dimension)
    masses = _masses(channel_masses)
    channels = len(masses)
    layout = normalise_channel_layout(
        sectors,
        channels=channels,
        channel_sectors=channel_sectors,
        channel_intrinsic_parities=channel_intrinsic_parities,
        intrinsic_parity=intrinsic_parity,
        selected_j_sectors=selected_j_sectors,
        max_dimension=budget,
    )
    expected = set(layout.active)
    # Keep the historical basis path bit-for-bit for the common/full-J call.
    # The generalized path uses the exact active projector below; with all
    # incidences selected its W basis differs only by a unitary rotation.
    legacy_common = (
        channel_sectors is None
        and channel_intrinsic_parities is None
        and selected_j_sectors is None
    )
    if legacy_common:
        pairs, _ = _sectors(sectors, budget)
        basis = _row_basis(pairs, group, irrep, intrinsic_parity, row, budget)
    else:
        basis = None
    points = [
        two_body_point(energy_lab_at=energy, mass1_at=m1, mass2_at=m2, frame=frame)
        for m1, m2 in masses
    ]
    if any(np.sqrt(point.s_at2) <= abs(m1 - m2) for point, (m1, m2) in zip(points, masses)):
        raise ValueError(f"energy {energy!r} lies below the pseudothreshold of a channel")
    s_at2 = points[0].s_at2
    if any(abs(point.s_at2 - s_at2) > 1e-12 * max(1.0, abs(s_at2)) for point in points):
        raise ArithmeticError("channel invariant masses disagree: s_at2 is not channel independent")
    blocks = _blocks(j_amplitudes, s_at2, expected)
    full = assemble_inverse(
        sectors if channel_sectors is None else None,
        scale=frame.length_at / (2 * np.pi),
        j_matrices=blocks,
        channels=channels,
        max_dimension=budget,
        weighting=weighting,
        # The paper's factor is (2k_i)^-ell_i with k_i the channel's own cm
        # momentum at this energy, so it must be evaluated here, not folded in.
        channel_k_squared_at2=[point.k_squared_at2 for point in points],
        channel_sectors=channel_sectors,
        channel_intrinsic_parities=channel_intrinsic_parities,
        intrinsic_parity=intrinsic_parity,
        selected_j_sectors=selected_j_sectors,
    )
    for channel, point in enumerate(points):
        block = slice(layout.channel_offsets[channel], layout.channel_offsets[channel + 1])
        # No Chew-Mandelstam term is added here on purpose.  ``effective_inverse_k``
        # with ``phase_space="chew-mandelstam"`` already returns ``K^-1 + I(s) + i rho(s)``,
        # so the caller's ``R`` carries the self-energy and the condition is simply
        # ``R_CM - box = 0``.  Adding it again double-counted Sigma and put every
        # CM level 1e-5 to 2e-3 off -- measured on the 0++ mock at L=16 d=(0,0,0),
        # where the nine roots paired up one to one but each was displaced.
        full[block, block] -= mixed_box(
            point.q_squared,
            layout.channel_pairs[channel],
            d=frame.d,
            gamma=point.gamma,
            alpha=point.alpha,
            max_dimension=budget,
            **settings,
        )
    if legacy_common:
        projection = np.kron(np.eye(channels), basis)
    else:
        projection, _, _ = channel_row_basis(layout, group, irrep, row, max_dimension=budget)
    result = projection.conj().T @ full @ projection
    if not np.allclose(result, result.conj().T, atol=1e-10, rtol=1e-12):
        raise ArithmeticError(
            f"non-Hermitian {irrep!r} row {row} for group {getattr(group, 'name', group)!r}, "
            f"energy={energy!r}, sectors={layout.channel_pairs}"
        )
    return result


def _free_poles(masses, frame, low, high, max_free_vectors) -> list[float]:
    """Union of every channel's free two-body energies inside ``(low, high)``."""
    poles: list[float] = []
    for m1, m2 in masses:
        data = free_two_body_states(
            mass1_at=m1,
            mass2_at=m2,
            frame=frame,
            energy_max_at=high,
            max_vectors=max_free_vectors,
        )
        poles.extend(
            float(state["energy_lab_at"])
            for state in data["states"]
            if low <= state["energy_lab_at"] <= high
        )
    poles.sort()
    merged: list[float] = []
    for value in poles:
        merge_tolerance = _FREE_POLE_MERGE_ULPS * np.spacing(
            max(1.0, abs(value), abs(merged[-1]) if merged else 0.0)
        )
        if not merged or value - merged[-1] > merge_tolerance:
            merged.append(value)
    return merged


def quantization_roots(
    energy_window,
    frame,
    channel_masses,
    sectors,
    j_amplitudes,
    *,
    group,
    irrep,
    row: int = 0,
    intrinsic_parity: int = 1,
    samples: int = 60,
    xtol: float = 1e-13,
    max_free_vectors: int = 400000,
    max_dimension: int = 256,
    pole_policy: str | None = None,
    weighting: str = WEIGHTING_SCALE,
    breakpoints_at2=None,
    channel_sectors=None,
    channel_intrinsic_parities=None,
    selected_j_sectors=None,
    root_method: str = "eigenvalues",
    subdivisions: int = 1,
    residual_tol: float = 1e-10,
    lhc_domain=None,
    **settings,
) -> tuple[float, ...]:
    """Quantization levels in a laboratory-energy window.

    Default ``root_method="eigenvalues"`` follows ordered Hermitian eigenvalues,
    refines sign brackets and sampled local extrema (including tangent roots),
    and checks min|lambda| / max(1, max|lambda|) <= ``residual_tol``. Each initial
    grid cell is bisected ``subdivisions`` times before local adaptive refinement.
    The default adds one midpoint. ``samples`` and subdivision govern resolution,
    not a guarantee that all roots have been found. Roots within ``xtol`` merge.

    ``root_method="determinant"`` retains the historical sign-change-only search
    for comparison; its output has no automatic matrix-residual acceptance gate.
    Free poles and supplied ``breakpoints_at2`` split both methods. Small free-
    pole exclusion bands remain; arbitrary model singularities must be declared.
    Evaluation/refinement failures abort, never return a partial spectrum.
    An explicit ``pole_policy="raise"`` still propagates ``FreePoleError``.

    Optional ``lhc_domain`` is an EqualMassExchangeDomain for one matching channel.
    It checks the entire CM-squared-energy window before any matrix evaluation;
    unknown or excluded domains raise. Omitting it makes no LHC validity claim.
    """
    if not isinstance(frame, LatticeFrame):
        raise ValueError(f"LatticeFrame required, got {type(frame).__name__}")
    from scipy.optimize import brentq

    low, high = _window(energy_window)
    masses = _masses(channel_masses)
    if isinstance(samples, bool) or not isinstance(samples, Integral) or int(samples) < 2:
        raise ValueError(f"samples must be an integer >= 2, got {samples!r}")
    samples = int(samples)
    if root_method not in ("eigenvalues", "determinant"):
        raise ValueError("root_method must be eigenvalues or determinant")
    if isinstance(subdivisions, bool) or not isinstance(subdivisions, Integral) or not 0 <= subdivisions <= 8:
        raise ValueError("subdivisions must be an integer in 0..8")
    residual_tol = _scalar(residual_tol, "residual_tol")
    if not 0 < residual_tol < 1:
        raise ValueError("residual_tol must lie in (0, 1)")
    xtol = _scalar(xtol, "xtol")
    if xtol <= 0:
        raise ValueError("positive finite xtol required")
    if (
        isinstance(max_free_vectors, bool)
        or not isinstance(max_free_vectors, Integral)
        or int(max_free_vectors) < 1
    ):
        raise ValueError(f"positive integer max_free_vectors required, got {max_free_vectors!r}")
    if pole_policy is not None:
        settings["pole_policy"] = pole_policy
    settings.setdefault("pole_policy", "flag")
    if settings["pole_policy"] not in POLE_POLICIES:
        raise ValueError(
            f"pole_policy must be one of {POLE_POLICIES}, got {settings['pole_policy']!r}"
        )
    # ``phase_space`` and ``subtractions`` describe the caller's reduced inverse, not the box.
    # They travel here only because callers pass them alongside the kernel arguments, and
    # leaving them in ``settings`` forwarded them all the way into ``harmonic_zeta_wide``,
    # which rejected them with "unexpected keyword argument 'phase_space'" -- so every model
    # failed at the first scan point.  Consume them rather than propagate.
    settings.pop("phase_space", None)
    settings.pop("subtractions", None)
    if channel_sectors is None:
        pairs, _ = _sectors(sectors, _dimension(max_dimension))
    else:
        if sectors is not None:
            raise ValueError(
                "pass either the common sectors argument or channel_sectors, not both"
            )
        pairs = None

    if lhc_domain is not None:
        from ..amplitudes.left_hand_cut import EqualMassExchangeDomain
        if not isinstance(lhc_domain, EqualMassExchangeDomain):
            raise ValueError("lhc_domain must be an EqualMassExchangeDomain")
        if lhc_domain.units != "temporal_lattice":
            raise ValueError("lhc_domain must use temporal_lattice units")
        if len(masses) != 1 or any(m != lhc_domain.mass for m in masses[0]):
            raise ValueError("lhc_domain requires one matching equal-mass channel")
        momentum_squared = sum(x*x for x in frame.momentum_at)
        lhc_domain.check((low*low - momentum_squared, high*high - momentum_squared)).require_qc()

    _preflight_zeta_domain(
        low,
        high,
        frame,
        masses,
        split=settings.get("split", 1.0),
    )

    def matrix_at(energy: float):
        try:
            return row_matrix(
                energy,
                frame,
                masses,
                pairs,
                j_amplitudes,
                group=group,
                irrep=irrep,
                row=row,
                intrinsic_parity=intrinsic_parity,
                max_dimension=max_dimension,
                weighting=weighting,
                channel_sectors=channel_sectors,
                channel_intrinsic_parities=channel_intrinsic_parities,
                selected_j_sectors=selected_j_sectors,
                **settings,
            )
        except FreePoleError:
            # The caller explicitly requested strict free-pole handling.  The
            # default flag policy does not enter this branch; it leaves the
            # automatic free-pole edges and near-pole sign structure intact.
            raise
        except (ArithmeticError, ValueError, TypeError, np.linalg.LinAlgError) as exc:
            raise RootScanError(
                f"root scan failed at laboratory energy {float(energy):.17g}: {exc}",
                energy=energy,
                cause=exc,
                kind="evaluation",
            ) from exc

    def determinant(energy: float) -> float:
        """Scaled quantization determinant at ``energy`` (zero iff ``det == 0``).

        ``det`` is evaluated through :func:`numpy.linalg.slogdet`, which returns
        ``(sign, log|det|)`` and never overflows: close to a free pole ``|det|``
        reaches ``1e36``, so a bare determinant loses all precision.  The value
        returned is ``sign(det) * log1p(|det|)``, which

        * vanishes exactly at ``det == 0`` (unlike ``sign * log|det|``, which
          vanishes at ``|det| == 1`` and would report spurious roots),
        * preserves the sign, so its sign changes are exactly those of ``det``,
        * grows only logarithmically, so the huge near-pole magnitudes cannot
          swamp the ``brentq`` tolerance.

        Matrix evaluation failures are fatal.  Converting a domain error to
        ``nan`` would make it indistinguishable from a cell that contains no
        root and could silently return an incomplete level set.  The default
        ``pole_policy='flag'`` keeps free-pole calls finite; an explicit
        ``'raise'`` policy exposes the structured free-pole exception.
        """
        value = matrix_at(energy)
        try:
            with np.errstate(invalid="ignore", over="ignore"):
                sign, log_abs = np.linalg.slogdet(value)
        except (ArithmeticError, ValueError, TypeError, np.linalg.LinAlgError) as exc:
            raise RootScanError(
                f"root scan determinant failed at laboratory energy {float(energy):.17g}: {exc}",
                energy=energy,
                cause=exc,
                kind="determinant",
            ) from exc
        if not np.isfinite(log_abs):
            # ``log|det| = -inf`` means an exactly singular row: that *is* a root.
            if log_abs == -np.inf:
                return 0.0
            cause = ValueError(f"slogdet returned non-finite log|det|={log_abs!r}")
            raise RootScanError(
                f"non-finite root determinant at laboratory energy {float(energy):.17g}",
                energy=energy,
                cause=cause,
                kind="determinant",
            ) from cause
        log_abs = float(log_abs.real)
        # Use the continuous softplus transform for every finite determinant.
        # ``logaddexp`` remains stable for large positive values; its result
        # rounds to zero for sufficiently negative values, so clamp only that
        # underflow to the smallest positive subnormal.  Exact singular rows
        # were handled above and remain the sole source of an exact zero.
        magnitude = max(
            float(np.logaddexp(0.0, log_abs)),
            float(np.nextafter(0.0, 1.0)),
        )
        result = float(np.sign(sign.real) * magnitude)
        if not np.isfinite(result):
            cause = ValueError(f"slogdet returned non-finite determinant sign={sign!r}")
            raise RootScanError(
                f"non-finite root determinant at laboratory energy {float(energy):.17g}",
                energy=energy,
                cause=cause,
                kind="determinant",
            ) from cause
        return result

    try:
        free_poles = _free_poles(masses, frame, low, high, max_free_vectors)
    except (ArithmeticError, TypeError, ValueError) as exc:
        raise RootScanError(
            f"root scan could not enumerate free poles in laboratory-energy interval "
            f"[{low:.17g}, {high:.17g}]: {exc}",
            cause=exc,
            kind="evaluation",
            interval=(low, high),
        ) from exc
    edges = [low] + free_poles + [high]
    if breakpoints_at2 is not None:
        # The model's own real singular points (the bare K pole and any K zero) must split
        # the scan.  Across such a point the determinant flips SIGN without passing through
        # zero, so a sign-change scan reports the singular point itself as a level.  Path A
        # carried this split and documented the failure it prevents: with
        # PolePolynomialK(mass=0.75, g=0.30, c0=0.10) on a 16^3 rest frame the determinant
        # went -8.65e6 to +4.28e6 across E = 1.209339 and a phantom level was returned whose
        # |det| was 5.7e5 instead of 0.  The JLS path had no such split at all, so it would
        # report a phantom level at every K pole inside the window -- the 0++ mock amplitude
        # has its bare pole at mass_at = 0.7, inside the scan window.
        momentum_squared = float(sum(x * x for x in frame.momentum_at))
        for value in np.atleast_1d(breakpoints_at2):
            shifted = float(value) + momentum_squared
            if shifted > 0.0:
                energy = float(np.sqrt(shifted))
                if low < energy < high:
                    edges.append(energy)
    edges = sorted(set(edges))
    free_edges = set(free_poles)
    roots: list[float] = []
    for left, right in zip(edges[:-1], edges[1:]):
        if left in free_edges:
            start = left + (right - left) * _FREE_POLE_PAD_FRACTION
        elif left == low:
            start = left
        else:
            start = float(np.nextafter(left, right))
        if right in free_edges:
            stop = right - (right - left) * _FREE_POLE_PAD_FRACTION
        elif right == high:
            stop = right
        else:
            stop = float(np.nextafter(right, left))
        if start > stop:
            cause = ValueError("no representable interior energy remains after splitting")
            raise RootScanError(
                "root scan cannot sample the split interval "
                f"[{left:.17g}, {right:.17g}]: {cause}",
                cause=cause,
                kind="refinement",
                interval=(left, right),
            ) from cause
        if root_method == "eigenvalues":
            try:
                roots.extend(spectral_roots(
                    matrix_at, start, stop, samples=samples, subdivisions=subdivisions,
                    xtol=xtol, residual_tol=residual_tol,
                ))
            except SpectralSearchError as exc:
                raise RootScanError(
                    f"spectral root scan failed: {exc}", energy=exc.energy,
                    interval=exc.interval, kind=exc.kind, cause=exc,
                ) from exc
            continue
        grid = np.linspace(start, stop, samples)
        values = [determinant(float(point)) for point in grid]
        for i in range(samples - 1):
            first, second = values[i], values[i + 1]
            if not np.isfinite(first):
                cause = ValueError("non-finite sampled determinant")
                raise RootScanError(
                    f"non-finite root determinant at laboratory energy {float(grid[i]):.17g}",
                    energy=grid[i],
                    cause=cause,
                    kind="determinant",
                ) from cause
            if not np.isfinite(second):
                cause = ValueError("non-finite sampled determinant")
                raise RootScanError(
                    f"non-finite root determinant at laboratory energy {float(grid[i + 1]):.17g}",
                    energy=grid[i + 1],
                    cause=cause,
                    kind="determinant",
                ) from cause
            if first == 0.0:
                roots.append(float(grid[i]))
            if second == 0.0:
                roots.append(float(grid[i + 1]))
                continue
            if np.signbit(first) == np.signbit(second):
                continue
            try:
                root = float(brentq(determinant, float(grid[i]), float(grid[i + 1]), xtol=xtol))
            except (RootScanError, FreePoleError):
                raise
            except Exception as exc:
                raise RootScanError(
                    "root scan could not refine a finite sign-changing bracket "
                    f"[{float(grid[i]):.17g}, {float(grid[i + 1]):.17g}]: {exc}",
                    cause=exc,
                    kind="refinement",
                    interval=(grid[i], grid[i + 1]),
                ) from exc
            if not np.isfinite(root):
                cause = ValueError("root refinement returned a non-finite energy")
                raise RootScanError(
                    "root scan refinement returned a non-finite energy for bracket "
                    f"[{float(grid[i]):.17g}, {float(grid[i + 1]):.17g}]",
                    cause=cause,
                    kind="refinement",
                    interval=(grid[i], grid[i + 1]),
                ) from cause
            roots.append(root)
    roots.sort()
    merged: list[float] = []
    for value in roots:
        merge_tolerance = max(
            xtol,
            8.0 * np.spacing(max(1.0, abs(value), abs(merged[-1]) if merged else 0.0)),
        )
        if not merged or value - merged[-1] > merge_tolerance:
            merged.append(value)
    return tuple(merged)
