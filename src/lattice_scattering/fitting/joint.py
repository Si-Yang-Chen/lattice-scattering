"""Joint correlated fitting and level matching across frames and irreps.

The standalone ``match_levels`` helper performs a one-to-one assignment.  A
``JointFitProblem`` uses grouped one-to-one matching when all observations have
explicit :class:`Observation` group labels.  If labels are absent, callers must
explicitly select an independent per-observation strategy; the API does not
infer quantization groups from frame or irrep metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from lattice_scattering.fitting.energy_fit import FitConvergenceError, fit_energy_model

__all__ = [
    "FitConvergenceError",
    "JointFitProblem",
    "MatchResult",
    "fit_joint",
    "match_levels",
]


@dataclass(frozen=True)
class Observation:
    """One measured level to be matched against a predicted root."""

    level_id: str
    frame: object
    frame_id: str
    irrep: str
    row: int
    energy: float
    group: object | None = None


@dataclass(frozen=True)
class MatchResult:
    """Assignment between observed levels and predicted roots.

    ``pairs`` holds ``(observation_index, root_index)``; ``distances`` follow
    the same sorted pair order. ``extra_roots`` and ``missing_levels`` are the
    unmatched indices of each side. Unmatched entries are reported.
    """

    pairs: tuple[tuple[int, int], ...]
    extra_roots: tuple[int, ...]
    missing_levels: tuple[int, ...]
    distances: tuple[float, ...]
    history: tuple[str, ...] = ()

    @property
    def n_matched(self) -> int:
        return len(self.pairs)

    def as_dict(self) -> dict:
        return {
            "matched": [[int(a), int(b)] for a, b in self.pairs],
            "extra_roots": [int(i) for i in self.extra_roots],
            "missing_levels": [int(i) for i in self.missing_levels],
            "distances": [float(d) for d in self.distances],
            "history": list(self.history),
        }


def match_levels(
    observed,
    predicted,
    *,
    overlaps=None,
    tolerance: float | None = None,
    previous=None,
) -> MatchResult:
    """One-to-one assignment of ``observed`` energies to ``predicted`` roots.

    ``overlaps`` is an optional ``(n_observed, n_predicted)`` matrix of
    eigenvector-overlap magnitudes.  When given, the assignment maximises total
    overlap (with energy distance as a small tie-breaker); without it, it
    minimises total energy distance.  In both cases the assignment is globally
    optimal subject to the tolerance and first maximises the number of matched
    pairs.  This avoids the nearest-first failure mode in dense spectra where a
    locally closest pair consumes the only candidate for another level.

    A predicted root and an observed level may only be paired when their
    distance is within ``tolerance`` (default: 10% of a nonzero observed span,
    otherwise 0.05); the
    leftovers are returned explicitly as ``extra_roots`` / ``missing_levels``.
    ``previous`` (a prior :class:`MatchResult`) is recorded for continuity
    diagnostics only.  Returned pairs and distances are ordered by observation
    index, independent of the order used by the assignment solver.
    """
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if observed.ndim != 1 or predicted.ndim != 1:
        raise ValueError("observed and predicted must be 1-d energy sequences")
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(predicted)):
        raise ValueError("finite energies required")
    if len(observed) == 0 or len(predicted) == 0:
        return MatchResult(
            pairs=(),
            extra_roots=tuple(range(len(predicted))),
            missing_levels=tuple(range(len(observed))),
            distances=(),
        )
    if tolerance is None:
        span = float(observed.max() - observed.min())
        tolerance = 0.1 * span if span > 0 else 0.05
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("positive finite tolerance required")
    if overlaps is not None:
        overlaps = np.asarray(overlaps, dtype=float)
        if (
            overlaps.shape != (len(observed), len(predicted))
            or not np.all(np.isfinite(overlaps))
            or np.any(overlaps < 0)
        ):
            raise ValueError(
                "overlaps must be a finite non-negative "
                "(n_observed, n_predicted) matrix"
            )
        cost = -overlaps + 1e-6 * np.abs(observed[:, None] - predicted[None, :]) / tolerance
    else:
        cost = np.abs(observed[:, None] - predicted[None, :])

    # Solve a square assignment after adding one dummy column for every
    # observation and one dummy row for every predicted root.  A finite edge
    # is allowed only within tolerance.  The unmatched penalty is larger than
    # every real edge, so a feasible match always beats leaving both endpoints
    # unmatched; this gives maximum cardinality first and minimum cost second.
    distances_matrix = np.abs(observed[:, None] - predicted[None, :])
    # Admit an exact tolerance-boundary pair when subtraction rounded one ulp
    # above the caller's decimal value, while keeping the requested tolerance
    # meaningful at the scale of the energies.
    roundoff = 8.0 * np.finfo(float).eps * max(
        1.0,
        float(np.max(np.abs(observed))),
        float(np.max(np.abs(predicted))),
        tolerance,
    )
    allowed = distances_matrix <= tolerance + roundoff
    n_observed, n_predicted = len(observed), len(predicted)
    # Equal-cost assignments receive a tiny non-separable index bias.  It makes
    # common ties stable without changing the meaningful energy/overlap
    # objective at ordinary numerical precision.
    tie_break = 1e-12 * (
        np.arange(n_observed, dtype=float)[:, None]
        - np.arange(n_predicted, dtype=float)[None, :]
    ) ** 2
    real_cost = cost + tie_break
    finite_cost = real_cost[allowed]
    min_cost = float(np.min(finite_cost)) if finite_cost.size else 0.0
    max_cost = float(np.max(finite_cost)) if finite_cost.size else 0.0
    max_abs_cost = max(abs(min_cost), abs(max_cost))
    # Make cardinality lexicographically dominant over the secondary cost.  A
    # lower-cardinality assignment can rearrange up to min(n_observed,
    # n_predicted) real edges, so a one-pair cardinality gain must outweigh the
    # full possible secondary-cost range of such a rearrangement.
    secondary_range = max_cost - min_cost
    unmatched_penalty = (
        (min(n_observed, n_predicted) + 1.0) * (secondary_range + 1.0)
        + max_abs_cost
        + 1.0
    )
    size = n_observed + n_predicted
    forbidden_cost = unmatched_penalty * (size + 1.0) + max_abs_cost + 1.0
    assignment_cost = np.full((size, size), forbidden_cost, dtype=float)
    assignment_cost[:n_observed, :n_predicted] = np.where(
        allowed, real_cost, forbidden_cost
    )
    assignment_cost[:n_observed, n_predicted:] = unmatched_penalty
    assignment_cost[n_observed:, :n_predicted] = unmatched_penalty
    assignment_cost[n_observed:, n_predicted:] = 0.0
    assigned_rows, assigned_columns = linear_sum_assignment(assignment_cost)

    pairs: list[tuple[int, int]] = []
    used_observed: set[int] = set()
    used_predicted: set[int] = set()
    for i, j in zip(assigned_rows, assigned_columns):
        if i >= n_observed or j >= n_predicted or not allowed[i, j]:
            continue
        pairs.append((int(i), int(j)))
        used_observed.add(int(i))
        used_predicted.add(int(j))
    pairs.sort()
    distances = [float(distances_matrix[i, j]) for i, j in pairs]
    history = []
    if previous is not None:
        changed = tuple(pairs) != tuple(previous.pairs)
        history.append(f"previous match present; changed={changed}")
    return MatchResult(
        pairs=tuple(pairs),
        extra_roots=tuple(j for j in range(len(predicted)) if j not in used_predicted),
        missing_levels=tuple(i for i in range(len(observed)) if i not in used_observed),
        distances=tuple(distances),
        history=tuple(history),
    )


_GROUPED_MATCHING_STRATEGIES = frozenset({"grouped"})
_INDEPENDENT_MATCHING_STRATEGIES = frozenset({"nearest"})


def _same_group_label(left, right) -> bool:
    """Compare explicit group labels without requiring them to be hashable."""
    try:
        equal = left == right
    except Exception:
        return left is right
    if isinstance(equal, (bool, np.bool_)):
        return bool(equal)
    # Array-valued equality is not a usable grouping label.  Treat it as a
    # distinct label instead of allowing ``if array`` to raise ambiguously.
    return False


def _observation_groups(observations):
    """Return first-seen ``(label, indices)`` groups from explicit labels.

    Grouped matching requires every observation to carry a non-null, hashable
    label.  A caller that wants an explicit singleton can provide a unique label.
    In particular, frame, irrep, and row are not folded into a key because
    equal-looking metadata do not prove that two predictors use the same
    quantization condition.
    """
    groups: list[tuple[object | None, list[int]]] = []
    for index, observation in enumerate(observations):
        label = observation.group
        if label is None:
            raise ValueError(
                "grouped matching requires an explicit non-null Observation.group "
                f"label for observation index {index}"
            )
        try:
            hash(label)
        except TypeError as error:
            raise ValueError("Observation.group labels must be hashable") from error
        for existing_label, indices in groups:
            if _same_group_label(label, existing_label):
                indices.append(index)
                break
        else:
            groups.append((label, [index]))
    return groups


@dataclass
class JointFitProblem:
    """A joint fit over observations with a per-condition root predictor.

    ``predict_roots(theta, observation)`` returns candidate root energies for a
    quantization condition, already restricted to the physical window. When
    every observation has an explicit ``group`` label and no strategy is given,
    labels define quantization groups and the API performs one-to-one matching
    inside each group. Each group must share one complete candidate root list;
    its predictor is called once with the first observation. Different labels
    are independent groups, even when their predicted energies coincide.

    When labels are absent, callers must explicitly choose
    ``matching_strategy="nearest"`` (or a callable) to request independent
    per-observation selection. Such selection can reuse roots. Group labels may
    not be combined with ``nearest`` or a callable because those strategies
    cannot enforce the declared group's one-to-one constraint. The API never
    infers groups from frame, irrep, or row.

    A callable ``matching_strategy`` retains the historical
    ``(observed_energy, roots) -> int | None`` per-observation contract and is
    available only when no observation groups are declared.
    """

    observations: tuple[Observation, ...]
    predict_roots: object
    matching_strategy: object | None = None
    tolerance: float | None = None
    match_history: list = field(default_factory=list)

    def __post_init__(self):
        """Resolve the safe default and reject ambiguous matching contracts."""
        if not self.observations:
            raise ValueError("JointFitProblem requires at least one observation")

        has_groups = [observation.group is not None for observation in self.observations]
        if self.matching_strategy is None:
            if all(has_groups):
                _observation_groups(self.observations)
                self.matching_strategy = "grouped"
            elif any(has_groups):
                missing = [index for index, present in enumerate(has_groups) if not present]
                raise ValueError(
                    "ambiguous matching groups: provide a non-null Observation.group "
                    f"for every observation, or remove all group labels and choose "
                    "matching_strategy='nearest' for independent per-observation matching; "
                    f"missing labels at observation indices {missing!r}"
                )
            else:
                raise ValueError(
                    "JointFitProblem requires explicit matching groups or an explicit "
                    "matching_strategy; provide Observation.group labels for one-to-one "
                    "grouped matching, or set matching_strategy='nearest' to allow "
                    "independent per-observation selection and possible root reuse"
                )
            return

        if not callable(self.matching_strategy):
            if not isinstance(self.matching_strategy, str):
                raise ValueError(
                    "matching_strategy must be 'grouped', 'nearest', a callable, or None"
                )
            self.matching_strategy = self.matching_strategy.lower()
            supported = _GROUPED_MATCHING_STRATEGIES | _INDEPENDENT_MATCHING_STRATEGIES
            if self.matching_strategy not in supported:
                raise ValueError(f"unknown matching strategy {self.matching_strategy!r}")

        if self.matching_strategy == "grouped":
            _observation_groups(self.observations)
            return

        if any(has_groups):
            raise ValueError(
                "Observation.group labels declare quantization groups and require "
                "matching_strategy='grouped' so roots remain one-to-one within each "
                "group; per-observation strategies can reuse roots"
            )

    def predict(self, theta) -> np.ndarray:
        """Predicted energy for every observation, in observation order.

        A root that cannot be matched raises, so a candidate parameter point with
        a missing level is never silently scored as if it were complete.
        """
        if self._uses_grouped_matching:
            return self._predict_grouped(theta)
        values = []
        for observation in self.observations:
            roots = np.atleast_1d(
                np.asarray(self.predict_roots(theta, observation), dtype=float)
            ).ravel()
            if roots.size == 0:
                raise ValueError(f"no predicted roots for level {observation.level_id!r}")
            selected = self._select(observation.energy, roots)
            if selected is None:
                raise ValueError(
                    f"level {observation.level_id!r} at {observation.energy!r} has no predicted "
                    f"root within tolerance {self.tolerance!r}"
                )
            values.append(float(roots[selected]))
            self.match_history.append(
                match_levels(
                    [observation.energy],
                    roots,
                    tolerance=self.tolerance,
                ).as_dict()
            )
        return np.asarray(values)

    @property
    def _uses_grouped_matching(self) -> bool:
        return isinstance(self.matching_strategy, str) and self.matching_strategy.lower() in _GROUPED_MATCHING_STRATEGIES

    def _predict_grouped(self, theta) -> np.ndarray:
        """Match each explicit group once and restore observation order."""
        groups = _observation_groups(self.observations)
        predicted = np.empty(len(self.observations), dtype=float)
        for group_number, (label, indices) in enumerate(groups):
            representative = self.observations[indices[0]]
            roots = np.atleast_1d(
                np.asarray(self.predict_roots(theta, representative), dtype=float)
            ).ravel()
            if roots.size == 0:
                raise ValueError(
                    f"no predicted roots for quantization group {label!r}"
                )
            if not np.all(np.isfinite(roots)):
                raise ValueError(
                    f"non-finite predicted roots for quantization group {label!r}"
                )
            observed = np.asarray(
                [self.observations[index].energy for index in indices], dtype=float
            )
            result = match_levels(observed, roots, tolerance=self.tolerance)
            self.match_history.append(
                {
                    "group": label if label is not None else f"observation:{indices[0]}",
                    "group_number": int(group_number),
                    "observation_indices": [int(index) for index in indices],
                    "matching": result.as_dict(),
                }
            )
            if result.missing_levels:
                missing = [
                    self.observations[indices[local_index]].level_id
                    for local_index in result.missing_levels
                ]
                raise ValueError(
                    f"no injective predicted-root assignment for quantization group "
                    f"{label!r}; missing levels {missing!r} within tolerance "
                    f"{self.tolerance!r}"
                )
            for local_index, root_index in result.pairs:
                predicted[indices[local_index]] = roots[root_index]
        return predicted

    def _select(self, energy: float, roots: np.ndarray):
        if callable(self.matching_strategy):
            index = self.matching_strategy(energy, roots)
            return None if index is None else int(index)
        if self.matching_strategy != "nearest":
            raise ValueError(f"unknown matching strategy {self.matching_strategy!r}")
        # ``scipy`` may hand a 0-d array for a scalar parameter vector; flatten so
        # a single-root predictor behaves like every other one.
        roots = np.atleast_1d(np.asarray(roots, dtype=float)).ravel()
        if roots.size == 0 or not np.all(np.isfinite(roots)):
            return None
        tolerance = self.tolerance
        if tolerance is None:
            tolerance = max(0.05, 0.1 * abs(energy))
        index = int(np.argmin(np.abs(roots - energy)))
        if abs(roots[index] - energy) > tolerance:
            return None
        return index


def fit_joint(
    problem: JointFitProblem,
    initial,
    *,
    covariance: np.ndarray,
    energies: np.ndarray,
    bounds=(-np.inf, np.inf),
    **solver_options,
) -> dict:
    """Correlated joint fit using the package's whitened-residual solver.

    ``covariance`` must be the full covariance of ``energies`` *in the same order
    as* ``problem.observations``. The solver performs whitening and an SVD local
    covariance estimate; this wrapper supplies the joint predictor and records
    matching history. Complete explicit ``Observation.group`` labels select
    one-to-one grouped matching by default. Otherwise callers must explicitly
    choose a per-observation strategy, which may reuse roots.
    """

    @dataclass(frozen=True)
    class _Data:
        energies_at: np.ndarray
        covariance_at2: np.ndarray

    data = _Data(energies_at=np.asarray(energies, dtype=float), covariance_at2=np.asarray(covariance, dtype=float))
    n = len(problem.observations)
    if data.energies_at.shape != (n,):
        raise ValueError(f"energies must have one entry per observation ({n}), got shape {data.energies_at.shape}")
    if data.covariance_at2.shape != (n, n):
        raise ValueError(
            f"covariance must be the ({n}, {n}) covariance of the observations, got shape {data.covariance_at2.shape}"
        )
    result = fit_energy_model(data, problem.predict, initial, bounds=bounds, **solver_options)
    result["diagnostics"]["matching"] = list(problem.match_history)
    result["diagnostics"]["n_observations"] = len(problem.observations)
    result["diagnostics"]["matching_strategy"] = (
        "callable" if callable(problem.matching_strategy) else problem.matching_strategy
    )
    result["diagnostics"]["matching_grouping"] = (
        "explicit Observation.group labels; every observation has a group label"
        if problem._uses_grouped_matching
        else "explicit per-observation strategy; candidate roots may be reused"
    )
    result["diagnostics"]["matching_root_uniqueness"] = (
        "enforced within explicit groups" if problem._uses_grouped_matching else "not enforced"
    )
    return result
