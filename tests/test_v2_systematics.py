"""P7: systematic-error propagation, model comparison and replica accounting.

Statistical and systematic uncertainties are kept separate here by design; the
tests check that separation rather than a single combined number.
"""

from __future__ import annotations

import numpy as np
import pytest

from lattice_scattering.data.replicas import ReplicaTable
from lattice_scattering.fitting import (
    JointFitProblem,
    Observation,
    compare_models,
    fit_replicas,
    perturb_and_refit,
    replica_summary,
)


def _linear_problem(slope: float, intercept: float):
    """A one-parameter fit through ``n`` observations with a known relation.

    ``predict_roots(theta, obs)`` returns a single root at
    ``theta[0] * obs.frame + intercept``, so the truth is exactly recoverable.
    """
    def predict_roots(theta, observation):
        return np.array([theta[0] * observation.frame + intercept])

    return predict_roots


def _observations(count: int = 5, span: float = 1.0, energies=None):
    """Observations whose ``frame`` is the abscissa and ``energy`` the datum."""
    frames = [span * (i + 1) / count for i in range(count)]
    if energies is None:
        energies = [0.0] * count
    return tuple(
        Observation(
            level_id=f"L{i}",
            frame=frames[i],
            frame_id=f"F{i}",
            irrep="A1",
            row=0,
            energy=float(energies[i]),
            group=f"F{i}",
        )
        for i in range(count)
    )


# ---------------------------------------------------------------------------
# systematics
# ---------------------------------------------------------------------------

def test_perturb_and_refit_propagates_nuisance_spread():
    """A nuisance mass shift must move the fitted slope, and be reported as spread."""
    truth_slope = 0.25
    intercept = 0.40
    frames = [0.2 * (i + 1) for i in range(5)]
    energies = np.array([truth_slope * f + intercept for f in frames])
    observations = _observations(count=5, span=1.0, energies=energies)
    covariance = np.eye(len(observations)) * 1e-8

    def factory(shift):
        # The nuisance shift rescales the intercept of the model relation.
        return JointFitProblem(
            observations=observations,
            predict_roots=_linear_problem(1.0, intercept + float(shift[0])),
            tolerance=0.05,
        )

    result = perturb_and_refit(
        factory,
        np.array([truth_slope]),
        covariance=covariance,
        energies=energies,
        nuisance_covariance=np.array([[0.01**2]]),
        n_draws=24,
        bounds=([0.0], [1.0]),
    )
    assert result.n_draws == 24
    assert result.n_failed == 0, result.failed
    # The slope is insensitive to an intercept shift, so its systematic spread
    # must be tiny compared with the nuisance size.
    assert result.systematic_covariance.shape == (1, 1)
    # The slope is insensitive to an intercept shift: its systematic spread must be
    # far below the nuisance scale (sigma = 0.01) and below the statistical error.
    assert result.systematic_covariance[0, 0] <= 1e-3
    assert "statistical covariance is reported separately" in result.scope


def test_perturb_and_refit_reports_failed_draws():
    """A draw that cannot be fit is recorded, never silently dropped."""
    frames = [0.2 * (i + 1) for i in range(5)]
    energies = np.array([0.25 * f + 0.40 for f in frames])
    observations = _observations(count=5, span=1.0, energies=energies)
    covariance = np.eye(len(observations)) * 1e-8

    def factory(shift):
        # Every draw gets a wildly wrong intercept, so no draw can be matched.
        intercept = 0.40 + 10.0
        return JointFitProblem(
            observations=observations,
            predict_roots=_linear_problem(1.0, intercept),
            tolerance=1e-3,
        )

    with pytest.raises(RuntimeError, match="systematic covariance cannot be estimated"):
        perturb_and_refit(
            factory, np.array([0.25]), covariance=covariance, energies=energies,
            nuisance_covariance=np.array([[1.0]]), n_draws=8, bounds=([0.0], [1.0]),
        )


def test_perturb_and_refit_requires_a_square_nuisance_covariance():
    observations = _observations()
    energies = np.array([0.25 * o.frame + 0.40 for o in observations])
    with pytest.raises(ValueError, match="square matrix"):
        perturb_and_refit(
            lambda shift: JointFitProblem(observations=observations, predict_roots=_linear_problem(1.0, 0.4)),
            np.array([0.25]), covariance=np.eye(5), energies=energies,
            nuisance_covariance=np.zeros((2, 3)), n_draws=4,
        )


# ---------------------------------------------------------------------------
# model comparison
# ---------------------------------------------------------------------------

def test_compare_models_penalises_a_nuisance_parameter():
    """A parameter that cannot improve the fit must cost AIC, not reduce chi2.

    The second parameter enters with a **fixed, resolvable** coefficient so the
    model is identifiable and the comparison isolates the AIC penalty.  A
    genuinely non-identifiable parameter is not silently absorbed;
    ``compare_models`` records it as failed (companion test below).
    """
    intercept = 0.40
    frames = [0.2 * (i + 1) for i in range(5)]
    energies = np.array([0.25 * f + intercept for f in frames])
    observations = _observations(count=5, span=1.0, energies=energies)
    covariance = np.eye(len(observations)) * 1e-8

    good = JointFitProblem(
        observations=observations, predict_roots=_linear_problem(1.0, intercept), tolerance=0.05
    )

    def two_parameter(theta, observation):
        # ``theta[1]`` multiplies a resolvable regressor, but the data prefer it
        # at zero, so the extra parameter only costs AIC.
        return np.array([theta[0] * observation.frame + intercept + 0.01 * theta[1] * observation.frame])

    wasteful = JointFitProblem(
        observations=observations, predict_roots=two_parameter, tolerance=0.05
    )
    comparison = compare_models(
        [
            ("one-parameter", good, np.array([0.24]), ([0.0], [1.0])),
            ("two-parameter", wasteful, np.array([0.24, 0.0]), ([0.0, -1.0], [1.0, 1.0])),
        ],
        covariance=covariance,
        energies=energies,
    )
    assert comparison.labels == ("one-parameter", "two-parameter")
    assert comparison.n_observations == len(observations)
    # The extra parameter cannot improve chi2 materially, so AIC must prefer the
    # model with fewer parameters.
    assert comparison.chi2[1] <= comparison.chi2[0] + 1e-6
    assert comparison.weights[0] > comparison.weights[1]
    assert abs(sum(comparison.weights) - 1.0) <= 1e-12


def test_compare_models_reports_a_non_identifiable_parameter_as_failed():
    """A parameter the data cannot constrain must fail loudly, not be absorbed."""
    frames = [0.2 * (i + 1) for i in range(5)]
    energies = np.array([0.25 * f + 0.40 for f in frames])
    observations = _observations(count=5, span=1.0, energies=energies)
    covariance = np.eye(len(observations)) * 1e-8
    good = JointFitProblem(
        observations=observations, predict_roots=_linear_problem(1.0, 0.40), tolerance=0.05
    )

    def non_identifiable(theta, observation):
        # ``theta[1]`` does not appear at all: the residual Jacobian is singular.
        return np.array([theta[0] * observation.frame + 0.40])

    broken = JointFitProblem(
        observations=observations, predict_roots=non_identifiable, tolerance=1e-3
    )
    comparison = compare_models(
        [
            ("good", good, np.array([0.24]), ([0.0], [1.0])),
            ("non-identifiable", broken, np.array([0.24, 0.0]), ([0.0, -1.0], [1.0, 1.0])),
        ],
        covariance=covariance,
        energies=energies,
    )
    assert comparison.labels == ("good",)
    assert len(comparison.failed) == 1 and "non-identifiable" in comparison.failed[0]
    assert abs(sum(comparison.weights) - 1.0) <= 1e-12


def test_compare_models_records_failed_candidates():
    frames = [0.2 * (i + 1) for i in range(5)]
    energies = np.array([0.25 * f + 0.40 for f in frames])
    observations = _observations(count=5, span=1.0, energies=energies)
    covariance = np.eye(len(observations)) * 1e-8
    good = JointFitProblem(
        observations=observations, predict_roots=_linear_problem(1.0, 0.40), tolerance=0.05
    )
    broken = JointFitProblem(
        observations=observations,
        predict_roots=lambda theta, observation: np.array([observation.frame + 10.0]),
        tolerance=1e-6,
    )
    comparison = compare_models(
        [
            ("good", good, np.array([0.24]), ([0.0], [1.0])),
            ("broken", broken, np.array([0.24]), ([0.0], [1.0])),
        ],
        covariance=covariance, energies=energies,
    )
    assert comparison.labels == ("good",)
    assert len(comparison.failed) == 1 and "broken" in comparison.failed[0]
    # Weights are normalised over the converged models only, and that is explicit.
    assert abs(sum(comparison.weights) - 1.0) <= 1e-12


# ---------------------------------------------------------------------------
# replicas
# ---------------------------------------------------------------------------

def _table() -> ReplicaTable:
    values = np.array([[0.20, 0.40], [0.21, 0.41], [0.19, 0.39], [0.205, 0.405]])
    return ReplicaTable(values, ("s0", "s1", "s2", "s3"), ("p0", "p1"), "jackknife", "ens")


def test_replica_summary_reports_mean_only_for_a_complete_set():
    table = _table()

    def fit_sample(sample_id, mapping):
        return np.array([mapping["p0"], mapping["p1"]])

    result = fit_replicas(table, fit_sample, parameter_ids=("p0", "p1"))
    summary = replica_summary(result)
    assert summary["complete"] is True
    assert summary["n_success"] == 4 and summary["n_failed"] == 0
    assert summary["replica_mean"] is not None
    assert len(summary["replica_covariance"]) == 2
    np.testing.assert_allclose(summary["replica_mean"], table.values.mean(axis=0), atol=1e-12)


def test_replica_summary_refuses_a_mean_when_samples_failed():
    """A failed sample must suppress the mean rather than be replaced."""
    table = _table()

    def fit_sample(sample_id, mapping):
        if sample_id == "s2":
            raise ValueError("deliberate failure")
        return np.array([mapping["p0"], mapping["p1"]])

    result = fit_replicas(table, fit_sample, parameter_ids=("p0", "p1"))
    summary = replica_summary(result)
    assert summary["complete"] is False
    assert summary["n_success"] == 3 and summary["n_failed"] == 1
    assert summary["failed_samples"] == ["s2"]
    assert "deliberate failure" in summary["failure_reasons"]["s2"]
    assert summary["replica_mean"] is None
    assert summary["replica_covariance"] is None


def test_replica_summary_requires_a_fit_result():
    with pytest.raises(ValueError, match="fit_replicas result"):
        replica_summary({"not": "a result"})
