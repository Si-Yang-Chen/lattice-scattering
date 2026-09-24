"""Correlated direct-observable fit for the 1107.5023 pi-pi NLO model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lattice_scattering.amplitudes.pipi_chiral import (
    PipiChiralPhaseObservablePredictor,
    PipiNLOChiral,
)

from .observables import (
    LinearObservableFit,
    ObservableDataset,
    fit_linear_observable,
    whitened_observable_residual,
)

__all__ = ["PipiChiralObservableFit", "fit_pipi_chiral_observables"]


@dataclass(frozen=True)
class PipiChiralObservableFit:
    """Fit result in the source convention and in the dataset's explicit row order."""

    linear_fit: LinearObservableFit
    amplitude: PipiNLOChiral
    coordinates: np.ndarray
    labels: tuple | None
    predicted_values: np.ndarray
    residuals: np.ndarray
    whitened_residuals: np.ndarray
    chi2: float

    @property
    def C1(self) -> float:
        return float(self.linear_fit.coefficients[0])

    @property
    def C2(self) -> float:
        return float(self.linear_fit.coefficients[1])

    @property
    def C4(self) -> float:
        return float(self.linear_fit.coefficients[2])

    @property
    def parameter_covariance(self) -> np.ndarray:
        return self.linear_fit.coefficient_covariance

    @property
    def dof(self) -> int:
        return self.linear_fit.dof

    @property
    def covariance_assumption(self) -> str:
        return self.linear_fit.covariance_assumption


def fit_pipi_chiral_observables(
    dataset: ObservableDataset,
    *,
    pion_mass: float,
    decay_constant: float,
) -> PipiChiralObservableFit:
    r"""Fit ordered ``k cot(delta)/m_pi`` rows with their full covariance.

    ``dataset.coordinates`` are physical ``s_phys`` values in the same units
    as ``pion_mass``; ``dataset.values`` are dimensionless source observables.
    The NLO prediction is affine in ``C1/C2/C4`` because the Eq. (20) inverse
    is truncated at first order in ``t_NLO``. The fixed zero-LEC term is
    removed, then the accepted generic correlated GLS implementation fits the
    three design columns without reordering observations or covariance rows.
    """
    if not isinstance(dataset, ObservableDataset):
        raise ValueError("ObservableDataset required")
    if dataset.covariance_assumption != "full_covariance_provided":
        raise ValueError("pi-pi chiral fits require an explicitly provided full covariance matrix")

    baseline_model = PipiNLOChiral(
        pion_mass=pion_mass,
        decay_constant=decay_constant,
        C1=0.0,
        C2=0.0,
        C4=0.0,
    )
    coordinates = dataset.coordinates
    baseline = PipiChiralPhaseObservablePredictor(baseline_model).predict(coordinates)

    basis_models = (
        PipiNLOChiral(pion_mass, decay_constant, 1.0, 0.0, 0.0),
        PipiNLOChiral(pion_mass, decay_constant, 0.0, 1.0, 0.0),
        PipiNLOChiral(pion_mass, decay_constant, 0.0, 0.0, 1.0),
    )
    design = np.column_stack(
        [
            PipiChiralPhaseObservablePredictor(model).predict(coordinates) - baseline
            for model in basis_models
        ]
    )
    centered_dataset = ObservableDataset(
        coordinates,
        dataset.values - baseline,
        dataset.covariance,
        labels=dataset.labels,
        covariance_labels=dataset.labels,
        covariance_assumption=dataset.covariance_assumption,
        notes=dataset.notes,
    )
    linear_fit = fit_linear_observable(
        centered_dataset,
        design,
        coefficient_names=("C1", "C2", "C4"),
    )
    amplitude = PipiNLOChiral(
        pion_mass=pion_mass,
        decay_constant=decay_constant,
        C1=float(linear_fit.coefficients[0]),
        C2=float(linear_fit.coefficients[1]),
        C4=float(linear_fit.coefficients[2]),
    )
    predicted = PipiChiralPhaseObservablePredictor(amplitude).predict(coordinates)
    residuals = predicted - dataset.values
    whitened = whitened_observable_residual(dataset, predicted)
    chi2 = float(whitened @ whitened)

    output_coordinates = np.array(coordinates, dtype=float, copy=True)
    predicted = np.array(predicted, dtype=float, copy=True)
    residuals = np.array(residuals, dtype=float, copy=True)
    whitened = np.array(whitened, dtype=float, copy=True)
    for values in (output_coordinates, predicted, residuals, whitened):
        values.setflags(write=False)
    return PipiChiralObservableFit(
        linear_fit=linear_fit,
        amplitude=amplitude,
        coordinates=output_coordinates,
        labels=dataset.labels,
        predicted_values=predicted,
        residuals=residuals,
        whitened_residuals=whitened,
        chi2=chi2,
    )
