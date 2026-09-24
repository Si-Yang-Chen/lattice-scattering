"""P5: joint correlated fitting and level matching.

Acceptance is **synthetic recovery**: levels are generated from a known truth
amplitude by the P4 finite-volume kernel, then fitted back with a joint fit over
several frames.  The truth parameters must be recovered and the local covariance
must be consistent with the observed scatter.
"""

from __future__ import annotations

import numpy as np
import pytest

from lattice_scattering.kinematics import LatticeFrame
from lattice_scattering.amplitudes import PolePolynomialK
from lattice_scattering.finite_volume.coupled_s import coupled_s_roots
from lattice_scattering.fitting import (
    JointFitProblem,
    Observation,
    fit_joint,
    match_levels,
)

MASSES = np.array([[0.33281, 0.33281], [0.34424, 0.34424]])
FRAMES = ((16, (0, 0, 0)), (16, (0, 0, 1)), (20, (0, 0, 0)))
WINDOW = (0.67, 0.75)
XI = 3.444


def _truth_model(couplings=(0.35, 0.20, -0.05), background=(-0.45, -0.62, -0.18)):
    return PolePolynomialK(
        mass=0.72,
        couplings=np.array(couplings),
        coefficients=(np.diag(background),),
    )


def _observations(model) -> tuple[Observation, ...]:
    out = []
    for sites, d in FRAMES:
        frame = LatticeFrame(sites, XI, d)
        p2 = float(sum(x * x for x in frame.momentum_at))
        window = ((WINDOW[0] ** 2 + p2) ** 0.5, (WINDOW[1] ** 2 + p2) ** 0.5)
        roots = coupled_s_roots(frame, MASSES, model, "simple", window, samples=40)
        for index, energy in enumerate(roots):
            out.append(
                Observation(
                    level_id=f"L{sites}{d}{index}",
                    frame=frame,
                    frame_id=f"L{sites}_{d}",
                    irrep="A1g" if d == (0, 0, 0) else "A1",
                    row=0,
                    energy=float(energy),
                    group=(f"L{sites}_{d}", "A1g" if d == (0, 0, 0) else "A1", 0),
                )
            )
    return tuple(out)


def _predict_roots(theta, observation):
    """Predict roots for one observation from a parameter vector ``theta``.

    Parameters: the two pole couplings and the two diagonal background entries.
    """
    couplings = np.array([theta[0], theta[1]])
    background = np.diag([theta[2], theta[3]])
    model = PolePolynomialK(mass=0.72, couplings=couplings, coefficients=(background,))
    frame = observation.frame
    p2 = float(sum(x * x for x in frame.momentum_at))
    window = ((WINDOW[0] ** 2 + p2) ** 0.5, (WINDOW[1] ** 2 + p2) ** 0.5)
    return coupled_s_roots(frame, MASSES, model, "simple", window, samples=40)


# ---------------------------------------------------------------------------
# level matching
# ---------------------------------------------------------------------------

def test_match_levels_pairs_within_tolerance():
    result = match_levels([0.70, 0.72, 0.74], [0.7001, 0.7199, 0.7402])
    assert result.pairs == ((0, 0), (1, 1), (2, 2))
    assert result.extra_roots == ()
    assert result.missing_levels == ()
    assert max(result.distances) <= 1e-3


def test_match_levels_reports_extras_and_missing_instead_of_dropping():
    """Unmatched entries must be reported, never silently discarded."""
    result = match_levels([0.70, 0.72], [0.7001, 0.72, 0.90], tolerance=0.01)
    assert result.n_matched == 2
    assert result.extra_roots == (2,)
    assert result.missing_levels == ()


def test_match_levels_reports_missing_levels():
    result = match_levels([0.70, 0.72, 0.95], [0.7001, 0.7199], tolerance=0.01)
    assert result.missing_levels == (2,)
    assert result.extra_roots == ()


def test_match_levels_uses_overlaps_when_given():
    """A high-overlap partner wins even when it is not the nearest in energy."""
    observed = [0.700]
    predicted = [0.7005, 0.7004]
    overlaps = np.array([[0.2, 0.9]])
    result = match_levels(observed, predicted, overlaps=overlaps, tolerance=0.01)
    assert result.pairs == ((0, 1),)


def test_match_levels_rejects_negative_overlap_magnitudes():
    with pytest.raises(ValueError, match="finite non-negative"):
        match_levels([0.70], [0.70], overlaps=np.array([[-0.1]]), tolerance=0.01)


def test_match_distances_follow_sorted_observation_pairs():
    result = match_levels([0.70, 0.90], [0.9001, 0.7002], tolerance=0.01)
    assert result.pairs == ((0, 1), (1, 0))
    assert np.allclose(result.distances, (0.0002, 0.0001))


def test_match_levels_records_history():
    first = match_levels([0.70, 0.72], [0.70, 0.72])
    second = match_levels([0.70, 0.72], [0.70, 0.72], previous=first)
    assert second.history and "previous match present" in second.history[0]


def test_match_levels_uses_global_assignment_for_dense_candidates():
    """A locally nearest pair must not consume the only candidate for another level."""
    result = match_levels([0.00, 0.10], [0.09, 0.20], tolerance=0.11)
    assert result.pairs == ((0, 0), (1, 1))
    np.testing.assert_allclose(result.distances, (0.09, 0.10))


def test_match_levels_maximizes_cardinality_before_distance():
    """A feasible complete assignment wins over a shorter zero-distance match."""
    result = match_levels([0.0, 1.0, 2.0], [0.0, 2.0, 4.0], tolerance=2.0)
    assert result.pairs == ((0, 0), (1, 1), (2, 2))
    assert result.missing_levels == ()


def _grouped_observation(level_id, energy, group):
    # Keep frame/irrep metadata identical in these tests.  The explicit group
    # label, rather than inferred metadata, determines which roots are shared.
    return Observation(
        level_id=level_id,
        frame="same-frame",
        frame_id="same-frame",
        irrep="A1",
        row=0,
        energy=energy,
        group=group,
    )


def test_grouped_joint_matching_rejects_duplicate_root_reuse():
    observations = (
        _grouped_observation("a", 0.70, "condition"),
        _grouped_observation("b", 0.70, "condition"),
    )
    problem = JointFitProblem(
        observations=observations,
        predict_roots=lambda theta, observation: np.array([0.70]),
        tolerance=0.01,
    )
    assert problem.matching_strategy == "grouped"
    with pytest.raises(ValueError, match="injective predicted-root assignment"):
        problem.predict(np.array([0.0]))


def test_unlabelled_observations_require_an_explicit_matching_strategy():
    observations = (
        _grouped_observation("a", 0.70, None),
        _grouped_observation("b", 0.70, None),
    )
    with pytest.raises(ValueError, match="requires explicit matching groups or an explicit matching_strategy"):
        JointFitProblem(
            observations=observations,
            predict_roots=lambda theta, observation: np.array([0.70]),
            tolerance=0.01,
        )


def test_partial_group_labels_are_rejected_as_ambiguous():
    observations = (
        _grouped_observation("a", 0.70, "condition"),
        _grouped_observation("b", 0.71, None),
    )
    with pytest.raises(ValueError, match="ambiguous matching groups.*missing labels"):
        JointFitProblem(
            observations=observations,
            predict_roots=lambda theta, observation: np.array([0.70, 0.71]),
        )


def test_explicit_nearest_strategy_can_intentionally_reuse_roots():
    observations = (
        _grouped_observation("a", 0.70, None),
        _grouped_observation("b", 0.70, None),
    )
    problem = JointFitProblem(
        observations=observations,
        predict_roots=lambda theta, observation: np.array([0.70]),
        matching_strategy="nearest",
        tolerance=0.01,
    )
    np.testing.assert_allclose(problem.predict(np.array([0.0])), [0.70, 0.70])


def test_grouped_joint_matching_calls_each_explicit_condition_once_and_restores_order():
    observations = (
        _grouped_observation("a", 0.70, "condition-a"),
        _grouped_observation("b", 0.71, "condition-a"),
        _grouped_observation("c", 0.70, "condition-b"),
    )
    calls = []

    def predict_roots(theta, observation):
        calls.append(observation.level_id)
        if observation.group == "condition-a":
            return np.array([0.70, 0.71])
        return np.array([0.70])

    problem = JointFitProblem(
        observations=observations,
        predict_roots=predict_roots,
        matching_strategy="grouped",
        tolerance=0.01,
    )
    np.testing.assert_allclose(problem.predict(np.array([0.0])), [0.70, 0.71, 0.70])
    # The two conditions happen to have the same frame/irrep metadata, but are
    # evaluated independently because their explicit labels differ.
    assert calls == ["a", "c"]
    assert problem.match_history[-2]["observation_indices"] == [0, 1]
    assert problem.match_history[-1]["observation_indices"] == [2]


def test_grouped_matching_requires_explicit_labels_instead_of_inference():
    observations = (
        _grouped_observation("a", 0.70, None),
        _grouped_observation("b", 0.70, None),
    )
    with pytest.raises(ValueError, match="explicit non-null Observation.group"):
        JointFitProblem(
            observations=observations,
            predict_roots=lambda theta, observation: np.array([0.70]),
            matching_strategy="grouped",
            tolerance=0.01,
        )


def test_explicit_group_cannot_be_bypassed_by_per_observation_strategy():
    observations = (
        _grouped_observation("a", 0.70, "condition"),
        _grouped_observation("b", 0.70, "condition"),
    )
    with pytest.raises(ValueError, match="Observation.group labels declare quantization groups"):
        JointFitProblem(
            observations=observations,
            predict_roots=lambda theta, observation: np.array([0.70]),
            matching_strategy="nearest",
        )


# ---------------------------------------------------------------------------
# joint fit: synthetic recovery
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_joint_fit_recovers_the_truth_parameters():
    """External recovery: the generating parameters come back within errors."""
    truth = (0.35, 0.20, -0.45, -0.62)
    model = _truth_model(couplings=truth[:2], background=truth[2:])
    observations = _observations(model)
    assert len(observations) >= 6, "need several levels for a joint fit"
    energies = np.array([o.energy for o in observations])
    # Correlated 0.5% errors, correlated within a frame.
    frames = np.array([o.frame_id for o in observations])
    covariance = np.zeros((len(observations),) * 2)
    for i in range(len(observations)):
        for j in range(len(observations)):
            covariance[i, j] = (0.005 * energies[i]) ** 2 * (
                1.0 if i == j else (0.2 if frames[i] == frames[j] else 0.0)
            )
    covariance += np.eye(len(observations)) * 1e-10
    problem = JointFitProblem(
        observations=observations, predict_roots=_predict_roots, tolerance=0.01
    )
    result = fit_joint(
        problem,
        np.array([0.30, 0.25, -0.40, -0.55]),
        covariance=covariance,
        energies=energies,
        bounds=([0.1, 0.1, -1.0, -1.0], [0.6, 0.6, 0.0, 0.0]),
    )
    assert result["status"] == "converged"
    recovered = np.array(result["parameters"])
    errors = np.sqrt(np.diag(np.array(result["covariance"])))
    # Every truth parameter within 4 sigma (the truth is the generating point).
    for value, true_value, error in zip(recovered, truth, errors):
        assert abs(value - true_value) <= 4 * error + 1e-6, (value, true_value, error)
    assert result["chi2"] / max(result["dof"], 1) < 25.0
    assert result["diagnostics"]["matching"], "the matching history must be recorded"


@pytest.mark.slow
def test_joint_fit_is_better_than_a_deliberately_wrong_amplitude():
    """The fit must reduce chi2 relative to a far-off starting amplitude."""
    truth = (0.35, 0.20, -0.45, -0.62)
    model = _truth_model(couplings=truth[:2], background=truth[2:])
    observations = _observations(model)
    energies = np.array([o.energy for o in observations])
    covariance = np.eye(len(observations)) * (0.003) ** 2
    problem = JointFitProblem(
        observations=observations, predict_roots=_predict_roots, tolerance=0.01
    )
    result = fit_joint(
        problem,
        np.array([0.12, 0.50, -0.90, -0.20]),
        covariance=covariance,
        energies=energies,
        bounds=([0.05, 0.05, -1.5, -1.5], [0.7, 0.7, 0.0, 0.0]),
    )
    assert result["chi2"] < 1e4


def test_joint_fit_requires_one_covariance_entry_per_observation():
    observations = (
        Observation(level_id="a", frame=None, frame_id="f", irrep="A1", row=0, energy=0.70),
        Observation(level_id="b", frame=None, frame_id="f", irrep="A1", row=0, energy=0.72),
        Observation(level_id="c", frame=None, frame_id="f", irrep="A1", row=0, energy=0.74),
    )
    problem = JointFitProblem(
        observations=observations, predict_roots=lambda t, o: [o.energy],
        matching_strategy="nearest",
    )
    with pytest.raises(ValueError, match="covariance must be the"):
        fit_joint(problem, np.array([1.0]), covariance=np.eye(2), energies=np.array([0.7, 0.72, 0.74]))


def test_unmatchable_level_raises_instead_of_being_silently_scored():
    """A parameter point with no matching root must fail loudly."""

    def no_roots(theta, observation):
        return np.array([observation.energy + 0.5])

    observations = (
        Observation(level_id="a", frame=None, frame_id="f", irrep="A1", row=0, energy=0.70),
        Observation(level_id="b", frame=None, frame_id="f", irrep="A1", row=0, energy=0.72),
        Observation(level_id="c", frame=None, frame_id="f", irrep="A1", row=0, energy=0.74),
    )
    problem = JointFitProblem(
        observations=observations, predict_roots=no_roots,
        matching_strategy="nearest", tolerance=0.001,
    )
    with pytest.raises(ValueError, match="no predicted root within tolerance"):
        problem.predict(np.array([1.0]))
