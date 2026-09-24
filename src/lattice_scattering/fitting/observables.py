"""Typed correlated-observable data and generalized least-squares fits.

The data container and residual whitening are model-agnostic.  The phase-fit
helper is one concrete consumer; later mass or phase models can reuse the same
ordered covariance handling without entering the finite-volume energy-fit path.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from numbers import Integral

import numpy as np
from scipy.linalg import solve_triangular


def _real_vector(name: str, value) -> np.ndarray:
    if _contains_boolean(value) or np.iscomplexobj(value):
        raise ValueError(f"{name} must be real and contain no booleans")
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or not len(result) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite nonempty vector")
    return np.array(result, dtype=float, copy=True)


def _contains_boolean(value) -> bool:
    """Detect Python/NumPy booleans before numeric coercion turns them into 0/1."""
    try:
        entries = np.asarray(value, dtype=object).flat
        return any(isinstance(entry, (bool, np.bool_)) for entry in entries)
    except (TypeError, ValueError):
        return False


def _real_scalar(name: str, value) -> float:
    if isinstance(value, (bool, np.bool_)) or np.iscomplexobj(value) or np.ndim(value) != 0:
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


def _labels(labels, count: int, name: str) -> tuple | None:
    if labels is None:
        return None
    try:
        result = tuple(labels)
        if len(result) != count:
            raise ValueError(f"{name} must have one entry per observation")
        if len(set(result)) != len(result):
            raise ValueError(f"{name} must contain unique labels")
    except TypeError as error:
        raise ValueError(f"{name} must contain unique hashable labels") from error
    return result


def _notes(notes) -> tuple[str, ...]:
    if isinstance(notes, str):
        notes = (notes,)
    try:
        result = tuple(notes)
    except TypeError as error:
        raise ValueError("notes must be strings") from error
    if any(not isinstance(note, str) or not note.strip() for note in result):
        raise ValueError("notes must be nonempty strings")
    return result


@dataclass(frozen=True, init=False)
class ObservableDataset:
    """Ordered real observations with a positive-definite aligned covariance.

    ``coordinates[i]``, ``values[i]`` and covariance row/column ``i`` always
    describe the same observation.  Optional labels make that contract
    checkable when covariance arrives from a separately ordered source.  When
    pointwise errors are used, :meth:`from_pointwise_errors` records the
    diagonal-covariance assumption explicitly.
    """

    coordinates: np.ndarray
    values: np.ndarray
    covariance: np.ndarray
    labels: tuple | None
    covariance_assumption: str
    notes: tuple[str, ...]

    def __init__(
        self,
        coordinates,
        values,
        covariance,
        *,
        labels=None,
        covariance_labels=None,
        covariance_assumption="full_covariance_provided",
        notes=(),
    ) -> None:
        x = _real_vector("coordinates", coordinates)
        y = _real_vector("values", values)
        if len(x) != len(y):
            raise ValueError("coordinates and values must have equal lengths")
        if _contains_boolean(covariance) or np.iscomplexobj(covariance):
            raise ValueError("covariance must be real and contain no booleans")
        cov = np.asarray(covariance, dtype=float)
        if cov.shape != (len(x), len(x)) or not np.all(np.isfinite(cov)):
            raise ValueError("covariance must be a finite square matrix matching the observations")
        scale = max(1.0, float(np.max(np.abs(cov))))
        if not np.allclose(cov, cov.T, rtol=1e-12, atol=1e-14 * scale):
            raise ValueError("covariance must be symmetric and ordered with the observations")
        cov = np.array((cov + cov.T) / 2.0, dtype=float, copy=True)
        try:
            np.linalg.cholesky(cov)
        except np.linalg.LinAlgError as error:
            raise ValueError("covariance must be positive definite") from error

        observation_labels = _labels(labels, len(x), "labels")
        covariance_order = _labels(covariance_labels, len(x), "covariance_labels")
        if observation_labels is None and covariance_order is not None:
            raise ValueError("covariance_labels require observation labels")
        if observation_labels is not None:
            if covariance_order is None:
                covariance_order = observation_labels
            if covariance_order != observation_labels:
                raise ValueError("covariance_labels must exactly match the ordered observation labels")
        if covariance_assumption not in (
            "full_covariance_provided",
            "diagonal_from_pointwise_errors",
        ):
            raise ValueError("unknown covariance assumption")

        x.setflags(write=False)
        y.setflags(write=False)
        cov.setflags(write=False)
        object.__setattr__(self, "coordinates", x)
        object.__setattr__(self, "values", y)
        object.__setattr__(self, "covariance", cov)
        object.__setattr__(self, "labels", observation_labels)
        object.__setattr__(self, "covariance_assumption", covariance_assumption)
        object.__setattr__(self, "notes", _notes(notes))

    @classmethod
    def from_pointwise_errors(
        cls, coordinates, values, errors, *, labels=None, notes=()
    ) -> "ObservableDataset":
        """Build the explicitly diagonal covariance implied by pointwise errors."""
        sigma = _real_vector("errors", errors)
        if np.any(sigma <= 0.0):
            raise ValueError("pointwise errors must be positive")
        x = _real_vector("coordinates", coordinates)
        y = _real_vector("values", values)
        if len(x) != len(y) or len(sigma) != len(x):
            raise ValueError("coordinates, values and errors must have equal lengths")
        return cls(
            x,
            y,
            np.diag(sigma**2),
            labels=labels,
            covariance_assumption="diagonal_from_pointwise_errors",
            notes=notes,
        )

    @classmethod
    def from_pairs(
        cls,
        observations,
        covariance,
        *,
        labels=None,
        covariance_labels=None,
        notes=(),
    ) -> "ObservableDataset":
        """Build an ordered dataset from ``(coordinate, value)`` pairs."""
        try:
            pairs = tuple(tuple(pair) for pair in observations)
        except TypeError as error:
            raise ValueError("observations must be ordered coordinate/value pairs") from error
        if not pairs or any(len(pair) != 2 for pair in pairs):
            raise ValueError("observations must be a nonempty sequence of coordinate/value pairs")
        return cls(
            [pair[0] for pair in pairs],
            [pair[1] for pair in pairs],
            covariance,
            labels=labels,
            covariance_labels=covariance_labels,
            notes=notes,
        )

    @classmethod
    def from_pairs_with_errors(
        cls, observations, errors, *, labels=None, notes=()
    ) -> "ObservableDataset":
        """Build ordered pairs while recording the diagonal-error assumption."""
        try:
            pairs = tuple(tuple(pair) for pair in observations)
        except TypeError as error:
            raise ValueError("observations must be ordered coordinate/value pairs") from error
        if not pairs or any(len(pair) != 2 for pair in pairs):
            raise ValueError("observations must be a nonempty sequence of coordinate/value pairs")
        return cls.from_pointwise_errors(
            [pair[0] for pair in pairs],
            [pair[1] for pair in pairs],
            errors,
            labels=labels,
            notes=notes,
        )

    def take(self, indices, *, note=None) -> "ObservableDataset":
        """Select positional rows and their matching covariance rows/columns."""
        try:
            selected = tuple(indices)
        except TypeError as error:
            raise ValueError("indices must be a nonempty sequence of distinct integers") from error
        if not selected or any(
            isinstance(index, bool) or not isinstance(index, Integral) for index in selected
        ):
            raise ValueError("indices must be a nonempty sequence of distinct integers")
        selected = tuple(int(index) for index in selected)
        if len(set(selected)) != len(selected) or any(
            index < 0 or index >= len(self.coordinates) for index in selected
        ):
            raise ValueError("indices must be distinct and within the observation range")
        labels = None if self.labels is None else tuple(self.labels[index] for index in selected)
        notes = self.notes + (() if note is None else _notes(note))
        index_array = np.asarray(selected, dtype=int)
        return ObservableDataset(
            self.coordinates[index_array],
            self.values[index_array],
            self.covariance[np.ix_(index_array, index_array)],
            labels=labels,
            covariance_labels=labels,
            covariance_assumption=self.covariance_assumption,
            notes=notes,
        )

    def select(self, labels, *, note=None) -> "ObservableDataset":
        """Select labeled observations in the requested order, preserving covariance."""
        if self.labels is None:
            raise ValueError("label selection requires observation labels")
        try:
            requested = tuple(labels)
        except TypeError as error:
            raise ValueError("labels must be a nonempty sequence of known unique labels") from error
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("labels must be a nonempty sequence of known unique labels")
        position = {label: index for index, label in enumerate(self.labels)}
        if any(label not in position for label in requested):
            raise ValueError("selection contains an unknown observation label")
        return self.take(tuple(position[label] for label in requested), note=note)


@dataclass(frozen=True)
class LinearObservableFit:
    """Generalized least-squares solution for a linear observable model."""

    coefficients: np.ndarray
    coefficient_covariance: np.ndarray
    predicted_values: np.ndarray
    residuals: np.ndarray
    chi2: float
    dof: int
    coefficient_names: tuple[str, ...]
    covariance_assumption: str
    notes: tuple[str, ...]


@dataclass(frozen=True)
class PhaseObservableFit:
    """Physical ``rho_0(s)=A+B*s`` fit and derived ``(g,s_r)`` covariance."""

    linear_fit: LinearObservableFit
    g: float
    s_r: float
    covariance_g_s_r: np.ndarray

    @property
    def A(self) -> float:
        return float(self.linear_fit.coefficients[0])

    @property
    def B(self) -> float:
        return float(self.linear_fit.coefficients[1])

    @property
    def covariance_A_B(self) -> np.ndarray:
        return self.linear_fit.coefficient_covariance

    @property
    def chi2(self) -> float:
        return self.linear_fit.chi2

    @property
    def dof(self) -> int:
        return self.linear_fit.dof

    @property
    def g_error(self) -> float:
        return sqrt(float(self.covariance_g_s_r[0, 0]))

    @property
    def s_r_error(self) -> float:
        return sqrt(float(self.covariance_g_s_r[1, 1]))

    def khat_inverse_at(self, s_at2) -> float:
        r"""Return the product-convention inverse amplitude analytically.

        The source observable is ``rho_0=(s_r-s)/g**2`` while the product uses
        ``Khat**-1=2 rho_0``.  This consumer path is finite at ``s=s_r`` and
        returns its analytic zero without evaluating the bare-pole ``K``.
        """
        s = _real_scalar("s_at2", s_at2)
        return 2.0 * (self.s_r - s) / self.g**2


def whitened_observable_residual(dataset: ObservableDataset, predictions) -> np.ndarray:
    """Whiten ``predictions - data`` with the dataset's ordered covariance."""
    if not isinstance(dataset, ObservableDataset):
        raise ValueError("ObservableDataset required")
    predicted = _real_vector("predictions", predictions)
    if len(predicted) != len(dataset.values):
        raise ValueError("predictions must match the observation count")
    chol = np.linalg.cholesky(dataset.covariance)
    return solve_triangular(chol, predicted - dataset.values, lower=True)


def fit_linear_observable(
    dataset: ObservableDataset,
    design_matrix,
    *,
    coefficient_names=None,
) -> LinearObservableFit:
    """Fit a linear model by generalized least squares.

    The supplied design matrix is row-aligned with ``dataset``.  This API is
    reusable for any linear observable parameterization; nonlinear models can
    reuse :func:`whitened_observable_residual` directly.
    """
    if not isinstance(dataset, ObservableDataset):
        raise ValueError("ObservableDataset required")
    if _contains_boolean(design_matrix) or np.iscomplexobj(design_matrix):
        raise ValueError("design_matrix must be real and contain no booleans")
    design = np.asarray(design_matrix, dtype=float)
    if (
        design.ndim != 2
        or design.shape[0] != len(dataset.values)
        or design.shape[1] == 0
        or design.shape[1] > design.shape[0]
        or not np.all(np.isfinite(design))
    ):
        raise ValueError("finite design_matrix with one row per observation and no more columns than rows required")
    parameter_count = design.shape[1]
    if coefficient_names is None:
        names = tuple(f"c{index}" for index in range(parameter_count))
    else:
        names = tuple(coefficient_names)
        if (
            len(names) != parameter_count
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("coefficient_names must uniquely name every design column")

    chol = np.linalg.cholesky(dataset.covariance)
    whitened_design = solve_triangular(chol, design, lower=True)
    whitened_values = solve_triangular(chol, dataset.values, lower=True)
    left, singular, right = np.linalg.svd(whitened_design, full_matrices=False)
    threshold = max(whitened_design.shape) * np.finfo(float).eps * singular[0]
    if len(singular) < parameter_count or singular[-1] <= threshold:
        raise ValueError("linear observable design is rank deficient")
    coefficients = right.T @ ((left.T @ whitened_values) / singular)
    coefficient_covariance = (right.T / singular**2) @ right
    predicted = design @ coefficients
    residual = predicted - dataset.values
    whitened = solve_triangular(chol, residual, lower=True)
    chi2 = float(whitened @ whitened)
    arrays = (coefficients, coefficient_covariance, predicted, residual)
    for array in arrays:
        array.setflags(write=False)
    return LinearObservableFit(
        coefficients=coefficients,
        coefficient_covariance=coefficient_covariance,
        predicted_values=predicted,
        residuals=residual,
        chi2=chi2,
        dof=len(dataset.values) - parameter_count,
        coefficient_names=names,
        covariance_assumption=dataset.covariance_assumption,
        notes=dataset.notes,
    )


def fit_phase_observable(dataset: ObservableDataset) -> PhaseObservableFit:
    r"""Fit ordered ``rho_0(s)=A+B*s`` data and propagate ``(g,s_r)`` covariance.

    Physical source parameters are ``g=sqrt(-1/B)`` and ``s_r=-A/B``.  Fits
    with ``B >= 0`` are rejected rather than mapped to an imaginary coupling.
    """
    if not isinstance(dataset, ObservableDataset):
        raise ValueError("ObservableDataset required")
    s = dataset.coordinates
    if np.any(s <= 0.0) or np.any(np.diff(s) <= 0.0):
        raise ValueError("phase-observable s coordinates must be positive and strictly increasing")
    design = np.column_stack((np.ones(len(s)), s))
    fit = fit_linear_observable(dataset, design, coefficient_names=("A", "B"))
    A, B = (float(value) for value in fit.coefficients)
    if B >= 0.0:
        raise ValueError("physical phase-observable fit requires B < 0")
    g = sqrt(-1.0 / B)
    s_r = -A / B
    jacobian = np.array(
        [
            [0.0, 0.5 * g**3],
            [-1.0 / B, A / B**2],
        ],
        dtype=float,
    )
    covariance_g_s_r = jacobian @ fit.coefficient_covariance @ jacobian.T
    covariance_g_s_r = (covariance_g_s_r + covariance_g_s_r.T) / 2.0
    if not isfinite(g) or not isfinite(s_r) or not np.all(np.isfinite(covariance_g_s_r)):
        raise ValueError("phase-observable parameter transform is nonfinite")
    covariance_g_s_r.setflags(write=False)
    return PhaseObservableFit(fit, g, s_r, covariance_g_s_r)


__all__ = [
    "LinearObservableFit",
    "ObservableDataset",
    "PhaseObservableFit",
    "fit_linear_observable",
    "fit_phase_observable",
    "whitened_observable_residual",
]
