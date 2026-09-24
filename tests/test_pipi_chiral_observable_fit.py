"""Independent correlated-fit oracle for the direct 1107.5023 observable."""

from cmath import log as complex_log
from math import log, pi, sqrt

import numpy as np
import pytest

from lattice_scattering.fitting import (
    ObservableDataset,
    fit_pipi_chiral_observables,
)


def _independent_kcot_over_mpi(s_phys_rows, m_pi, f_pi, C1, C2, C4):
    """Evaluate the printed Eq. (10)/(20) expressions without product code."""
    predicted = []
    for s_phys in s_phys_rows:
        k = sqrt(s_phys / 4.0 - m_pi**2)
        k2 = k * k
        m2 = m_pi * m_pi
        f2 = f_pi * f_pi
        t_lo = -(m2 + 2.0 * k2) / (8.0 * pi * f2)
        u = sqrt(k2 / (k2 + m2))
        v = sqrt((k2 + m2) / k2)
        L_s = complex_log((u - 1.0) / (u + 1.0))
        L_t = log((v - 1.0) / (v + 1.0))

        t_nlo = -m2**2 / f2**2 * (C1 - 31.0 / (384.0 * pi**3))
        t_nlo += (k2 / f2) * (m2 / f2) * (
            301.0 / (1152.0 * pi**3) - C2 / (128.0 * pi**2) - C1 / 2.0
        )
        t_nlo += (k2**2 / f2**2) * (
            14.0 / (45.0 * pi**3)
            - (19.0 * C1 / 8.0 - 9.0 * C2 / (512.0 * pi**2) + 216.0 * pi * C4)
        )
        t_nlo -= (
            (3.0 * m2**2 / 32.0 + 5.0 * m2 * k2 / 12.0 + 5.0 * k2**2 / 9.0)
            / (4.0 * pi**3 * f2**2)
            * log(m2 / f2)
        )
        t_nlo += (
            (m2**2 / 4.0 + m2 * k2 + k2**2)
            / (16.0 * pi**3 * f2**2)
            * u
            * L_s
        )
        t_nlo += (
            (3.0 * m2**2 / 16.0 + 7.0 * m2 * k2 / 9.0 + 11.0 * k2**2 / 18.0)
            / (8.0 * pi**3 * f2**2)
            * v
            * L_t
        )
        t_nlo -= (
            m2**2
            / (128.0 * pi**3 * f2**2)
            * (1.0 + 13.0 * m2 / (12.0 * k2))
            * L_t**2
        )

        eq20 = (
            sqrt(1.0 + k2 / m2)
            * (1.0 / t_lo - t_nlo / t_lo**2)
            + 1j * k / m_pi
        )
        assert eq20.imag == pytest.approx(0.0, abs=3e-13)
        predicted.append(eq20.real)
    return np.asarray(predicted)


def test_correlated_pipi_fit_preserves_rows_and_matches_independent_gls_oracle():
    m_pi, f_pi = 0.31, 0.47
    truth = np.asarray([0.12, -0.21, 0.004])
    # The source row order is intentionally non-monotone and is not changed by
    # either the fit design or the covariance whitening.
    q_rows = np.asarray([0.67, 0.12, 0.41, 0.88, 0.29, 0.54])
    coordinates = 4.0 * m_pi**2 * (1.0 + q_rows**2)
    labels = ("row-5", "row-1", "row-4", "row-6", "row-2", "row-3")
    residual_offset = np.asarray([0.00015, -0.00012, 0.00020, -0.00018, 0.00006, -0.00008])
    covariance = 0.0004 * np.eye(len(coordinates)) + 0.00005 * np.ones(
        (len(coordinates), len(coordinates))
    )
    generated_values = _independent_kcot_over_mpi(coordinates, m_pi, f_pi, *truth)
    dataset = ObservableDataset(
        coordinates,
        generated_values + residual_offset,
        covariance,
        labels=labels,
        covariance_labels=labels,
    )

    fit = fit_pipi_chiral_observables(
        dataset,
        pion_mass=m_pi,
        decay_constant=f_pi,
    )

    baseline = _independent_kcot_over_mpi(coordinates, m_pi, f_pi, 0.0, 0.0, 0.0)
    design = np.column_stack(
        [
            _independent_kcot_over_mpi(coordinates, m_pi, f_pi, *basis) - baseline
            for basis in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        ]
    )
    chol = np.linalg.cholesky(covariance)
    whitened_design = np.linalg.solve(chol, design)
    whitened_values = np.linalg.solve(chol, dataset.values - baseline)
    expected_parameters = np.linalg.lstsq(whitened_design, whitened_values, rcond=None)[0]
    fit_parameters = np.asarray([fit.C1, fit.C2, fit.C4])
    # Compare the residual path at the same fitted point. The accepted SVD GLS
    # solve and NumPy's independent lstsq oracle can differ by a few ulps in
    # coefficients; whitening a small residual amplifies that optimizer
    # difference by ||L^-1||. Parameter agreement is checked separately above.
    # Evaluate the source formulas independently, then whiten predictions and
    # data separately so their near-equal unwhitened values are not subtracted
    # before the covariance transform.
    independent_fit_predictions = _independent_kcot_over_mpi(
        coordinates, m_pi, f_pi, *fit_parameters
    )
    expected_residuals = independent_fit_predictions - dataset.values
    expected_whitened = (
        np.linalg.solve(chol, independent_fit_predictions)
        - np.linalg.solve(chol, dataset.values)
    )

    assert (fit.C1, fit.C2, fit.C4) == pytest.approx(expected_parameters, rel=2e-9, abs=2e-10)
    assert fit.coordinates == pytest.approx(coordinates, rel=0.0, abs=0.0)
    assert fit.labels == labels
    assert fit.predicted_values == pytest.approx(independent_fit_predictions, rel=2e-10, abs=2e-11)
    assert fit.residuals == pytest.approx(expected_residuals, rel=2e-9, abs=2e-11)
    assert fit.whitened_residuals == pytest.approx(expected_whitened, rel=2e-9, abs=2e-11)
    assert fit.chi2 == pytest.approx(float(expected_whitened @ expected_whitened), rel=2e-9, abs=2e-12)
    assert fit.dof == len(coordinates) - 3
    assert fit.covariance_assumption == "full_covariance_provided"


def test_pipi_correlated_fit_requires_explicit_full_covariance():
    dataset = ObservableDataset.from_pointwise_errors(
        [0.5, 0.6, 0.7, 0.8],
        [0.1, 0.2, 0.3, 0.4],
        [0.01, 0.01, 0.01, 0.01],
    )
    with pytest.raises(ValueError, match="explicitly provided full covariance"):
        fit_pipi_chiral_observables(dataset, pion_mass=0.31, decay_constant=0.47)
