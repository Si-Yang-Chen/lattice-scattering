"""Wide-domain harmonic Zeta sums for moving frames.

Uses heat-kernel and Poisson terms with convergence diagnostics and a
relative free-spectrum-pole guard. The validated input domain is documented
in docs/conventions.md. Numerical diagnostics are not rigorous error bounds."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import ceil, log, pi, sqrt
from numbers import Integral
from time import perf_counter

import numpy as np
from scipy.integrate import quad_vec

from lattice_scattering.finite_volume.moving import _grid
from lattice_scattering.symmetry.solid_harmonics import solid_harmonics

#: Highest supported ``|d|**2`` (``|d| <= 6`` momentum classes).
MAX_D2 = 36
#: Highest supported harmonic degree (``solid_harmonics.MAX_DEGREE``).
MAX_ELL = 12
#: Supported real ``q2`` window (closed channels are part of it).
Q2_MIN, Q2_MAX = -48.0, 4.0
#: Supported boost / shift window.
GAMMA_MIN, GAMMA_MAX = 1.0, 1.52
ALPHA_MIN, ALPHA_MAX = 0.0, 1.0
SPLIT_MIN, SPLIT_MAX = 0.5, 2.0
#: Smallest Poisson-image cube radius accepted (legacy behaviour).
MIN_CUTOFF = 10
#: Default relative tolerance of the per-call stability diagnostics.
DEFAULT_TOL = 1e-11
#: Heat-kernel exponent constant of the direct-cube padding rule (legacy value).
_PADDING_HEAT = 40.0
#: Quadrature tolerances handed to ``scipy.integrate.quad_vec``.
_QUAD_ABS, _QUAD_REL = 1e-11, 1e-11
#: Shells added to the neglected Poisson-image cube when bounding its tail.
_TAIL_SHELLS = 2
#: Extra direct-cube shells measured as the truncation diagnostic.
_DIRECT_SHELLS = 2
#: Maximum number of padding/cutoff refinements per call.
_MAX_REFINE = 3
#: Default relative distance to a free-spectrum pole that is treated as a pole.
DEFAULT_POLE_TOLERANCE = 1e-9
#: Accepted ``pole_policy`` values.
POLE_POLICIES = ("raise", "flag")


class FreePoleError(ValueError):
    """A direct-cube lattice point sits on the free spectrum.

    Raised by :func:`harmonic_zeta_wide` when the relative pole distance drops
    to ``pole_tolerance`` or below and ``pole_policy='raise'``.  Subclassing
    ``ValueError`` keeps the legacy ``except ValueError`` behaviour of the
    exact-equality guard, while the attributes carry the offending kinematics
    and the guard numbers so a caller can log or re-dispatch without parsing the
    message.

    Attributes: ``ell``, ``d``, ``q2``, ``gamma``, ``alpha``, ``split`` (the
    call that hit the pole), ``pole_distance`` and ``pole_tolerance`` (relative
    numbers), ``nearest_lattice`` (the integer lattice vector closest to
    on-shell), ``delta`` (its ``|r|**2 - q2``) and ``source`` (``"direct"`` or
    ``"direct_tail"``, the cube the point belongs to).
    """

    def __init__(
        self,
        message: str,
        *,
        ell: int,
        d,
        q2: float,
        gamma: float,
        alpha: float,
        split: float,
        pole_distance: float,
        pole_tolerance: float,
        nearest_lattice,
        delta: float,
        source: str,
    ) -> None:
        super().__init__(message)
        self.ell = ell
        self.d = tuple(d)
        self.q2 = q2
        self.gamma = gamma
        self.alpha = alpha
        self.split = split
        self.pole_distance = pole_distance
        self.pole_tolerance = pole_tolerance
        self.nearest_lattice = tuple(nearest_lattice)
        self.delta = delta
        self.source = source


@dataclass(frozen=True)
class ZetaResult:
    """Harmonic zeta values ``Z_lm`` (``m = -ell..ell``) plus diagnostics.

    ``values`` is a complex array of length ``2*ell+1`` ordered by increasing
    ``m``.  ``diagnostics`` carries the split/cutoff/padding actually used, the
    direct (cube) and image (integral) contributions separately, and the
    stability numbers described in :func:`harmonic_zeta_wide`.

    Instances are **shared** by the memoised :func:`harmonic_zeta_wide`, so both
    ``values`` and the arrays inside ``diagnostics`` are write-protected here.
    The container types themselves are ordinary (a ``dict`` cannot be frozen),
    but every array a caller could otherwise mutate in place is read-only.
    """

    values: np.ndarray
    ell: int
    diagnostics: dict

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if values.flags.writeable:
            values.setflags(write=False)
        for entry in self.diagnostics.values():
            if isinstance(entry, np.ndarray) and entry.flags.writeable:
                entry.setflags(write=False)


def _scalar(name: str, value, lo: float | None = None, hi: float | None = None) -> float:
    """Return ``value`` as a finite real float inside ``[lo, hi]``."""
    if np.iscomplexobj(value) or np.ndim(value) != 0:
        raise ValueError(f"{name} must be a real scalar")
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite")
    if (lo is not None and out < lo) or (hi is not None and out > hi):
        raise ValueError(f"{name}={out} outside supported range [{lo}, {hi}]")
    return out


def _momentum(d) -> tuple[int, int, int]:
    """Return ``d`` as an integer triple with ``|d|**2 <= MAX_D2``."""
    try:
        entries = tuple(d)
    except TypeError as error:
        raise ValueError("d must be an integer triple") from error
    if len(entries) != 3 or any(isinstance(x, bool) or not isinstance(x, Integral) for x in entries):
        raise ValueError("d must be an integer triple")
    out = tuple(int(x) for x in entries)
    if sum(x * x for x in out) > MAX_D2:
        raise ValueError(f"d={out} outside |d|^2 <= {MAX_D2}")
    return out


def _degree(ell) -> int:
    """Return ``ell`` as an int in ``0..MAX_ELL``."""
    if isinstance(ell, bool) or not isinstance(ell, Integral) or not 0 <= ell <= MAX_ELL:
        raise ValueError(f"harmonic degree 0..{MAX_ELL} required")
    return int(ell)


def _cutoff(cutoff) -> int:
    """Return the requested image-cube radius as an int ``>= MIN_CUTOFF``."""
    if isinstance(cutoff, bool) or not isinstance(cutoff, Integral) or cutoff < MIN_CUTOFF:
        raise ValueError(f"cutoff must be an integer >= {MIN_CUTOFF}")
    return int(cutoff)


def _budget(points: int, ell: int, max_values) -> None:
    """Raise if a grid of ``points`` vectors exceeds the solid-harmonic budget."""
    if isinstance(max_values, bool) or not isinstance(max_values, Integral) or max_values < 0:
        raise ValueError("max_values must be a non-negative integer value budget")
    needed = points * (2 * ell + 1)
    if needed > int(max_values):
        raise ValueError(f"harmonic grid needs {needed} values, budget is {max_values}")


def _direct_padding(q2: float, norm: float, gamma: float, alpha: float, split: float, cutoff: int) -> int:
    """Direct-cube radius: legacy heat-kernel rule, ``alpha*|d|`` included.

    Neglected lattice points have ``|r| >= (padding + 1 - alpha*|d|)/gamma``, so
    the rule keeps ``split*(|r|**2 - q2) >= 40`` wherever the cube is cut.
    """
    spectral = sqrt(_PADDING_HEAT / split + max(q2, 0.0))
    return max(int(cutoff), int(ceil(alpha * norm + gamma * spectral)))


def _image_cutoff(q2: float, split: float, cutoff: int) -> int:
    """Poisson-image radius: smallest cube damping the tail below tolerance.

    At ``v <= sqrt(split)`` the image damping is bounded by
    ``exp(q2*split - pi**2*|w|**2/split)`` and neglected images have
    ``|w| >= image_cutoff + 1``; the rule keeps that bound below ``DEFAULT_TOL``.
    """
    needed = sqrt(split * (max(q2, 0.0) * split + log(1.0 / DEFAULT_TOL))) / pi
    return max(int(cutoff), int(ceil(needed)) + 1)


def _pole_probe(r: np.ndarray, delta: np.ndarray, lattice: np.ndarray, scale: float, source: str) -> dict:
    """Relative distance of the closest of ``r`` to the free spectrum.

    ``delta = |r|**2 - q2`` is the on-shell residual, ``lattice`` the integer
    lattice vector and ``scale`` the normalisation ``max(1, max|r|**2, |q2|)``
    supplied by :func:`_cube_scale`.  The ratio is dimensionless, so a single
    ``pole_tolerance`` means the same thing at ``q2 = 0`` and at ``q2 = -48``.
    It is also the right invariant for the failure this guard exists for: a
    physically on-shell point is represented with an absolute error of order
    ``eps * |r|**2``, hence ``pole_distance = O(eps)`` however the on-shell
    condition happens to round, while the nearest non-degenerate point of the
    cube sits at a relative distance far above ``eps``.  A leading ``1/delta``
    term then never exceeds ``1/pole_tolerance`` times its weight, instead of
    the ``1e15`` blow-up of a missed pole.
    """
    magnitude = np.abs(delta)
    index = int(np.argmin(magnitude))
    return {
        "source": source,
        "index": index,
        "pole_distance": float(magnitude[index] / scale),
        "delta": float(delta[index]),
        "rotated": np.array(r[index], dtype=float),
        "nearest_lattice": np.array(lattice[index], dtype=float),
        "solid_harmonic": None,
        "terms": None,
    }


def _cube_scale(points: np.ndarray, q2: float) -> float:
    """Normalisation ``max(1, max|r|**2, |q2|)`` of one probed cube."""
    largest = float(np.max(np.sum(points * points, axis=1))) if len(points) else 1.0
    return max(1.0, largest, abs(q2))


def _no_pole(source: str) -> dict:
    """A probe of an empty cube: infinitely far from the free spectrum."""
    return {
        "source": source,
        "index": -1,
        "pole_distance": np.inf,
        "delta": 0.0,
        "rotated": None,
        "nearest_lattice": None,
        "solid_harmonic": None,
        "terms": None,
    }


def _direct_sum(q2, ell, vector, unit, gamma, alpha, split, padding, max_values) -> tuple[np.ndarray, dict]:
    """Direct lattice sum over the cube ``|n|_inf <= padding`` plus pole probe.

    Returns the contribution and the closest-to-shell probe of this cube; the
    pole policy belongs to the caller.
    """
    lattice = _grid(padding)
    r = lattice - alpha * vector
    r = r + (1.0 / gamma - 1.0) * np.outer(r @ unit, unit)
    delta = np.sum(r * r, axis=1) - q2
    probe = _pole_probe(r, delta, lattice, _cube_scale(r, q2), "direct")
    harmonics = solid_harmonics(r, ell, max_values=max_values)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        terms = np.exp(-split * delta) / delta
    probe["solid_harmonic"] = np.array(harmonics[probe["index"]], dtype=complex)
    probe["terms"] = terms[probe["index"]]
    return np.einsum("ij,i->j", harmonics, terms), probe


def _direct_tail(q2, ell, vector, unit, gamma, alpha, split, padding, max_values) -> tuple[float, dict]:
    """Largest direct contribution of the nearest neglected cube shells.

    The measured value is ``max_m |sum_{padding < |n|_inf <= padding+2} ...|``:
    the leading term of the direct-cube truncation error, which decays
    super-exponentially in ``padding``.  Its lattice points carry the same
    ``1/delta`` pole as the direct cube, so they are probed as well: a pole in a
    neglected shell would make the result meaningless long before the truncation
    diagnostic notices.
    """
    lattice = _grid(padding + _DIRECT_SHELLS)
    shell = lattice[np.max(np.abs(lattice), axis=1) > padding]
    if not len(shell):
        return 0.0, _no_pole("direct_tail")
    r = shell - alpha * vector
    r = r + (1.0 / gamma - 1.0) * np.outer(r @ unit, unit)
    delta = np.sum(r * r, axis=1) - q2
    probe = _pole_probe(r, delta, shell, _cube_scale(r, q2), "direct_tail")
    harmonics = solid_harmonics(r, ell, max_values=max_values)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        terms = np.exp(-split * delta) / delta
    return float(np.max(np.abs(np.einsum("ij,i->j", harmonics, terms)))), probe


def _image_sum(q2, ell, vector, unit, gamma, alpha, split, image_cutoff, max_values) -> tuple[np.ndarray, int, float, dict]:
    """Poisson image term, its error estimate and its pole probe.

    The probe is diagnostic only: the image summands are analytic in ``q2``, so
    an image-cube point at ``|w|**2 == q2`` is *not* a singularity (see
    :func:`_pole_guard`).  It is reported because the task asks for the image
    cube to be measured on its own scale.
    """
    n = _grid(image_cutoff)
    n = n[np.any(n != 0.0, axis=1)]
    w = n + (gamma - 1.0) * np.outer(n @ unit, unit)
    w2 = np.sum(w * w, axis=1)
    probe = _pole_probe(w, w2 - q2, n, _cube_scale(w, q2), "image")
    weights = solid_harmonics(w, ell, max_values=max_values) * np.exp(2j * pi * alpha * (n @ vector))[:, None]
    degree = 2 * ell + 2

    def integrand(v):
        if v == 0.0:
            return np.concatenate(([2.0 * q2], np.zeros(2 * ell + 1)))
        damping = np.exp(q2 * v * v - pi * pi * w2 / (v * v))
        images = 2.0 * np.einsum("ij,i->j", weights, damping) / v**degree
        return np.concatenate(([2.0 * np.expm1(q2 * v * v) / (v * v)], images))

    integral, error = quad_vec(integrand, 0.0, sqrt(split), epsabs=_QUAD_ABS, epsrel=_QUAD_REL)
    contribution = gamma * (1j**ell) * pi ** (ell + 1.5) * integral[1:]
    if ell == 0:
        contribution = contribution + gamma * pi / 2.0 * (-2.0 / sqrt(split) + integral[0])
    return contribution, len(n), float(np.max(np.atleast_1d(error))), probe


def _image_tail_bound(q2, ell, gamma, alpha, split, image_cutoff, unit, vector, max_values) -> float:
    """Conservative bound on the Poisson images outside the image cube.

    Integrand magnitudes are bounded at ``v = sqrt(split)`` (the largest
    damping argument) and integrated over the interval, for the nearest
    ``_TAIL_SHELLS`` neglected shells; the shells beyond them are suppressed by
    the same Gaussian factor and are numerically negligible (measured, not
    claimed).
    """
    grid = _grid(image_cutoff + _TAIL_SHELLS)
    shell = grid[np.max(np.abs(grid), axis=1) > image_cutoff]
    if not len(shell):
        return 0.0
    w = shell + (gamma - 1.0) * np.outer(shell @ unit, unit)
    w2 = np.sum(w * w, axis=1)
    damping = np.exp(q2 * split - pi * pi * w2 / split)
    weights = np.abs(solid_harmonics(w, ell, max_values=max_values))
    weight_sum = np.sum(np.abs(weights) * damping[:, None], axis=0)
    return float(2.0 * sqrt(split) / split ** (ell + 1) * np.max(weight_sum))


def _validate_pole_tolerance(pole_tolerance) -> float:
    """Return ``pole_tolerance`` as a finite positive relative tolerance."""
    out = _scalar("pole_tolerance", pole_tolerance)
    if out <= 0.0:
        raise ValueError(f"pole_tolerance must be positive, got {out}")
    return out


def _pole_guard(probes, *, ell, dvec, q2, gamma, alpha, split, pole_tolerance, pole_policy) -> tuple[dict, bool]:
    """Apply the free-pole policy and return the closest probe and hit flag.

    ``probes`` are the direct cube and its neglection shells; the minimum
    relative distance wins.  ``pole_policy='raise'`` (default) raises
    :class:`FreePoleError`, ``'flag'`` returns the value with
    ``status='near_free_pole'`` and the leading ``1/delta`` coefficient of the
    amplified direct lattice point (``diagnostics['pole_leading']``).

    The Poisson image cube is deliberately **not** a raising probe.  Its
    summands are analytic in ``q2``: at ``|w|**2 == q2`` the integrand
    ``2*Y(w)*exp(q2*v**2 - pi**2*|w|**2/v**2)/v**(2*ell+2)`` remains finite (the
    Gaussian becomes ``exp(|w|**2*(v**2 - pi**2/v**2))``, whose saddle is well
    inside ``(0, sqrt(split))`` for the supported window) and the ``ell = 0``
    smooth part is subtracted before the integral.  Treating an image hit as a
    pole would therefore reject physically normal calls -- e.g. at
    ``d=(0,0,1), gamma=1.5, alpha=0.5`` the image point ``n=(0,0,1)`` sits at
    ``q2 = 2.25`` where the legacy kernel returns the finite ``Z00 = -24.700``.
    """
    nearest = min(probes, key=lambda probe: probe["pole_distance"])
    hit = nearest["pole_distance"] <= pole_tolerance
    if hit and pole_policy == "raise":
        lattice = tuple(int(round(x)) for x in nearest["nearest_lattice"])
        raise FreePoleError(
            "free-spectrum pole: ell={0}, d={1}, q2={2!r}, gamma={3!r}, alpha={4!r}, split={5!r} has a {6} "
            "lattice point n={7} with r={8} on the free spectrum: delta={9:.6e}, "
            "pole_distance={10:.3e} <= pole_tolerance={11:.3e}".format(
                ell,
                tuple(dvec),
                q2,
                gamma,
                alpha,
                split,
                nearest["source"],
                lattice,
                tuple(float(x) for x in nearest["rotated"]),
                nearest["delta"],
                nearest["pole_distance"],
                pole_tolerance,
            ),
            ell=ell,
            d=dvec,
            q2=q2,
            gamma=gamma,
            alpha=alpha,
            split=split,
            pole_distance=nearest["pole_distance"],
            pole_tolerance=pole_tolerance,
            nearest_lattice=lattice,
            delta=nearest["delta"],
            source=nearest["source"],
        )
    return nearest, hit


def _harmonic_zeta_wide_uncached(
    q2,
    ell,
    *,
    d,
    gamma,
    alpha,
    split=1.0,
    cutoff=10,
    max_values=10_000_000,
    pole_tolerance=DEFAULT_POLE_TOLERANCE,
    pole_policy="raise",
) -> ZetaResult:
    """Uncached implementation; use :func:`harmonic_zeta_wide` instead."""
    """Return ``Z_lm`` for ``m = -ell..ell`` on the wide v2 domain.

    Domain: ``|d|**2 <= 36``, ``ell <= 12``, real ``q2 in [-48, 4]`` (closed
    channels included), ``gamma in [1, 1.52]``, ``alpha in [0, 1]``,
    ``split in [0.5, 2]``.  ``d`` must be integer valued; the rest frame
    (``d == (0, 0, 0)``) requires ``gamma == 1``.

    Free-spectrum poles are detected by *relative distance*, not by floating
    point equality: a physically on-shell lattice point of the direct cube
    typically misses the on-shell condition ``|r|**2 == q2`` by O(1 ulp), so the
    equality test misses it and the ``1/delta`` term returns ``~1e15`` while the
    truncation diagnostics still say ``status='ok'`` (they certify convergence,
    not distance to the pole).  Let ``scale = max(1, max|r|**2, |q2|)`` of the
    probed cube and ``pole_distance = min|delta| / scale`` over the direct cube
    and its truncation shells (the image cube is measured separately and cannot
    host the singularity).  If ``pole_distance`` is at most ``pole_tolerance``
    (default ``1e-9``, must be finite and positive) the call either raises
    :class:`FreePoleError` (``pole_policy='raise'``, the default) or returns the
    value with ``status='near_free_pole'`` and the leading ``1/delta``
    coefficient of the amplified lattice point under
    ``diagnostics['pole_leading']`` (``pole_policy='flag'``).

    The image cube is probed too (``image_pole_distance``,
    ``image_nearest_lattice``) but deliberately not acted on: its summands are
    analytic in ``q2``, so ``|w|**2 == q2`` is not a singularity, and raising on
    it would reject the finite value the legacy kernel returns at, e.g.,
    ``d=(0,0,1), gamma=1.5, alpha=0.5, q2=2.25``.

    Conventions (identical to the legacy kernel and to ``solid_harmonics``):

    * complex regular solid harmonics with the Condon--Shortley phase, so
      ``Z_l,-m = (-1)**m conj(Z_l,m)`` and ``Z_l0`` is real for real input;
    * ``Z_lm`` transforms under a momentum-star rotation ``R`` with
      ``spin_rotation(q, 2*ell).conj()``.

    The direct cube and the Poisson image cube are chosen adaptively and the
    call re-runs itself with larger controls until the measured direct-cube
    shell contribution and the bounded image tail are below
    ``DEFAULT_TOL * max(1, max|Z|)`` (at most ``_MAX_REFINE`` refinements).

    If the measured stability does not reach the advertised level after
    ``_MAX_REFINE`` refinements the call raises ``ArithmeticError`` (with the
    identity of the offending ``(ell, d, q2, gamma, alpha, split)``) instead of
    returning an uncertified value.

    ``diagnostics`` keys: ``split``, ``cutoff`` (requested), ``padding`` and
    ``image_cutoff`` (used), ``direct`` and ``image`` (the two contributions per
    ``m``), ``direct_terms``, ``image_terms``, ``direct_tail`` (measured),
    ``image_tail_bound`` (bounded), ``integral_error`` (quadrature estimate),
    ``convergence`` (``{(ell, m): bool}``), ``tolerance``, ``tolerance_per_m``,
    ``conjugate_defect``, ``stable``, ``status``, ``refinements``, ``seconds``,
    ``pole_distance``, ``nearest_lattice``, ``pole_source``, ``pole_delta``,
    ``pole_tolerance``, ``pole_policy``, ``image_pole_distance``,
    ``image_nearest_lattice`` and (when a pole is flagged) ``pole_leading``.
    """
    start = perf_counter()
    ell = _degree(ell)
    dvec = _momentum(d)
    q2 = _scalar("q2", q2, Q2_MIN, Q2_MAX)
    gamma = _scalar("gamma", gamma, GAMMA_MIN, GAMMA_MAX)
    alpha = _scalar("alpha", alpha, ALPHA_MIN, ALPHA_MAX)
    split = _scalar("split", split, SPLIT_MIN, SPLIT_MAX)
    cutoff = _cutoff(cutoff)
    pole_tolerance = _validate_pole_tolerance(pole_tolerance)
    if pole_policy not in POLE_POLICIES:
        raise ValueError(f"pole_policy must be one of {POLE_POLICIES}, got {pole_policy!r}")
    if dvec == (0, 0, 0) and gamma != 1.0:
        raise ValueError("rest frame (d=0) requires gamma == 1")
    vector = np.array(dvec, dtype=float)
    norm = float(np.linalg.norm(vector))
    unit = vector / norm if norm else np.zeros(3)

    padding = _direct_padding(q2, norm, gamma, alpha, split, cutoff)
    image_cutoff = _image_cutoff(q2, split, cutoff)
    refinements = 0
    while True:
        _budget(max((2 * (padding + _DIRECT_SHELLS) + 1) ** 3, (2 * (image_cutoff + _TAIL_SHELLS) + 1) ** 3), ell, max_values)
        direct, direct_probe = _direct_sum(q2, ell, vector, unit, gamma, alpha, split, padding, max_values)
        image, image_terms, integral_error, image_probe = _image_sum(q2, ell, vector, unit, gamma, alpha, split, image_cutoff, max_values)
        values = direct + image
        tail_direct, tail_probe = _direct_tail(q2, ell, vector, unit, gamma, alpha, split, padding, max_values)
        tail_image = _image_tail_bound(q2, ell, gamma, alpha, split, image_cutoff, unit, vector, max_values)
        pole, hit = _pole_guard(
            (direct_probe, tail_probe),
            ell=ell,
            dvec=dvec,
            q2=q2,
            gamma=gamma,
            alpha=alpha,
            split=split,
            pole_tolerance=pole_tolerance,
            pole_policy=pole_policy,
        )
        tolerance = DEFAULT_TOL * max(1.0, float(np.max(np.abs(values))))
        # The return gate checks every m, including symmetry-zero components.
        # Refine against that same gate, not only the largest component: the
        # latter could stop early and then fail for a smaller component.
        refinement_tolerance = float(np.min(DEFAULT_TOL * np.maximum(1.0, np.abs(values))))
        if max(tail_direct, tail_image) <= refinement_tolerance or refinements >= _MAX_REFINE:
            break
        if tail_direct >= tail_image:
            padding += _DIRECT_SHELLS
        else:
            image_cutoff += _TAIL_SHELLS
        refinements += 1

    order = np.arange(ell + 1)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        defect = direct[ell - order] - (-1.0) ** order * direct[ell + order].conjugate()
        defect_image = image[ell - order] - (-1.0) ** order * image[ell + order].conjugate()
        tolerance_per_m = DEFAULT_TOL * np.maximum(1.0, np.abs(values))
    tapering = max(tail_direct, tail_image)
    convergence = {(ell, int(m)): bool(tapering <= tolerance_per_m[ell + m]) for m in range(-ell, ell + 1)}
    stable = bool(tapering <= tolerance and all(convergence.values()))
    if not stable:
        raise ArithmeticError(
            "unstable harmonic truncation for ell={0}, d={1}, q2={2}, gamma={3}, alpha={4}, split={5}: "
            "direct_tail={6:.3e}, image_tail_bound={7:.3e}, tolerance={8:.3e}".format(
                ell, dvec, q2, gamma, alpha, split, tail_direct, tail_image, tolerance
            )
        )
    diagnostics = {
        "convergence": convergence,
        "tolerance_per_m": tolerance_per_m,
        "status": "near_free_pole" if hit else "ok",
        "seconds": perf_counter() - start,
        "split": split,
        "cutoff": cutoff,
        "padding": padding,
        "image_cutoff": image_cutoff,
        "direct": direct,
        "image": image,
        "direct_terms": (2 * padding + 1) ** 3,
        "image_terms": image_terms,
        "direct_tail": tail_direct,
        "image_tail_bound": tail_image,
        "integral_error": integral_error,
        "conjugate_defect": float(max(np.max(np.abs(defect)), np.max(np.abs(defect_image)))),
        "tolerance": tolerance,
        "stable": stable,
        "refinements": refinements,
        "pole_distance": pole["pole_distance"],
        "nearest_lattice": tuple(int(round(x)) for x in pole["nearest_lattice"]) if pole["nearest_lattice"] is not None else None,
        "pole_source": pole["source"],
        "pole_delta": pole["delta"],
        "pole_tolerance": pole_tolerance,
        "pole_policy": pole_policy,
        "image_pole_distance": image_probe["pole_distance"],
        "image_nearest_lattice": tuple(int(round(x)) for x in image_probe["nearest_lattice"]),
        "pole_leading": None,
    }
    if hit:
        harmonics = pole["solid_harmonic"]
        terms = pole["terms"]
        diagnostics["pole_leading"] = {
            "nearest_lattice": tuple(int(round(x)) for x in pole["nearest_lattice"]),
            "rotated_lattice": tuple(float(x) for x in pole["rotated"]),
            "delta": pole["delta"],
            "inverse_delta": None if pole["delta"] == 0.0 else 1.0 / pole["delta"],
            "solid_harmonics": None if harmonics is None else tuple(complex(x) for x in np.asarray(harmonics)),
            "weighted_terms": None if terms is None or not np.isfinite(terms) else complex(terms),
        }
    return ZetaResult(values=values, ell=ell, diagnostics=diagnostics)


# --------------------------------------------------------------------------
# memoised public entry point
# --------------------------------------------------------------------------
#: Cache size: a scan evaluates the same lattice point repeatedly (``brentq``
#: iterates near a root, and the finite-volume kernels ask for the same
#: ``(q2, ell)`` from several channels), so a modest LRU pays for itself many
#: times over.  ``harmonic_zeta_wide`` is a pure function of its arguments.
_ZETA_CACHE_ENTRIES = 200_000


@lru_cache(maxsize=_ZETA_CACHE_ENTRIES)
def _harmonic_zeta_wide_cached(
    q2, ell, d, gamma, alpha, split, cutoff, max_values, pole_tolerance, pole_policy
) -> ZetaResult:
    """Hashable-argument trampoline around the uncached implementation."""
    return _harmonic_zeta_wide_uncached(
        q2,
        ell,
        d=d,
        gamma=gamma,
        alpha=alpha,
        split=split,
        cutoff=cutoff,
        max_values=max_values,
        pole_tolerance=pole_tolerance,
        pole_policy=pole_policy,
    )


def harmonic_zeta_wide(
    q2,
    ell,
    *,
    d,
    gamma,
    alpha,
    split=1.0,
    cutoff=10,
    max_values=10_000_000,
    pole_tolerance=DEFAULT_POLE_TOLERANCE,
    pole_policy="raise",
) -> ZetaResult:
    """Memoised :func:`_harmonic_zeta_wide_uncached`.

    Identical arguments return the **same** :class:`ZetaResult` object; its
    arrays are write-protected (see :class:`ZetaResult`) because every caller
    treats them as read-only.  Memoisation is what keeps a multi-frame root scan
    tractable: a fit's Jacobian perturbs the parameters by ``~1e-8`` and mostly
    revisits the same lattice points.

    The arguments are **hashed before validation**, so a wrong-typed argument
    raises the implementation's own ``ValueError`` (not a ``TypeError`` from
    coercion here).  The key is built with :func:`_cache_key`, which reproduces
    the value classes the implementation accepts.
    """
    return _call_cached(q2, ell, d, gamma, alpha, split, cutoff, max_values, pole_tolerance, pole_policy)


def _call_cached(q2, ell, d, gamma, alpha, split, cutoff, max_values, pole_tolerance, pole_policy):
    """Dispatch to the memo or straight to the implementation on a bad key."""
    key = _cache_key(q2, ell, d, gamma, alpha, split, cutoff, max_values, pole_tolerance, pole_policy)
    if key is None:
        return _harmonic_zeta_wide_uncached(
            q2, ell, d=d, gamma=gamma, alpha=alpha, split=split, cutoff=cutoff,
            max_values=max_values, pole_tolerance=pole_tolerance, pole_policy=pole_policy,
        )
    return _harmonic_zeta_wide_cached(*key)


def _cache_key(q2, ell, d, gamma, alpha, split, cutoff, max_values, pole_tolerance, pole_policy):
    """Hashable key, or ``None`` when a non-numeric argument must bypass caching.

    A ``None`` key makes the caller invoke the uncached implementation directly,
    so a malformed argument still raises the implementation's own
    ``ValueError``/``TypeError`` instead of this wrapper coercing it or failing
    early.
    """
    def plain(value):
        """A hashable plain real scalar, or ``None``."""
        if isinstance(value, bool) or np.iscomplexobj(value) or np.ndim(value) != 0:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    if (
        isinstance(d, (tuple, list))
        and len(d) == 3
        and all(isinstance(x, Integral) and not isinstance(x, bool) for x in d)
    ):
        d_key = (int(d[0]), int(d[1]), int(d[2]))
    else:
        d_key = None

    def integer(value):
        return int(value) if (isinstance(value, Integral) and not isinstance(value, bool)) else None

    # ``cutoff`` and ``max_values`` are validated as integers by the
    # implementation, so they must reach it as ``Integral``, not ``float``.
    ell_key = integer(ell)
    cutoff_key = integer(cutoff)
    max_values_key = integer(max_values)
    q2_key = plain(q2)
    gamma_key = plain(gamma)
    alpha_key = plain(alpha)
    split_key = plain(split)
    pole_tolerance_key = plain(pole_tolerance)
    parts = (
        q2_key, ell_key, d_key, gamma_key, alpha_key,
        split_key, cutoff_key, max_values_key, pole_tolerance_key,
    )
    if any(value is None for value in parts):
        return None
    return (*parts, str(pole_policy))


def zeta_cache_info():
    """Cache statistics of the memoised Zeta kernel (for diagnostics/tests)."""
    return _harmonic_zeta_wide_cached.cache_info()


def zeta_cache_clear() -> None:
    """Drop the memoised Zeta entries (useful in long-running fits)."""
    _harmonic_zeta_wide_cached.cache_clear()
