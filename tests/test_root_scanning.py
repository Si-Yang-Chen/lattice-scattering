"""Root-scan failure policy and pole-boundary regressions."""

from __future__ import annotations

import numpy as np
import pytest

from lattice_scattering.finite_volume import Q2_MAX, RootScanError, quantization_roots
from lattice_scattering.finite_volume import jls_matrix
from lattice_scattering.kinematics import LatticeFrame, two_body_point
from lattice_scattering.symmetry.groups import oh_group


FRAME = LatticeFrame(24, 1.0, (0, 0, 0))
MASSES = ((0.3, 0.3),)
SECTORS = ((0, 0),)
BLOCKS = {0: lambda _s: np.array([[1.0]])}


def _scan(*, samples=8, window=(0.6, 0.9)):
    return quantization_roots(
        window,
        FRAME,
        MASSES,
        SECTORS,
        BLOCKS,
        group=oh_group(),
        irrep="A1g",
        samples=samples,
        root_method="determinant",  # historical path regressions; spectral cases have their own suite
    )


def test_domain_failure_is_not_silently_skipped(monkeypatch):
    """A bad matrix point must abort instead of returning a partial level set."""

    monkeypatch.setattr(jls_matrix, "_free_poles", lambda *args: [])

    def fail(*args, **kwargs):
        raise ValueError("q2 outside supported range")

    monkeypatch.setattr(jls_matrix, "row_matrix", fail)

    with pytest.raises(RootScanError, match=r"root scan failed .*q2 outside") as caught:
        _scan(samples=4)
    error = caught.value
    assert isinstance(error, ValueError)
    assert error.kind == "evaluation"
    assert error.energy is not None
    assert isinstance(error.cause, ValueError)


def _moving_q2_upper_energy():
    frame = LatticeFrame(24, 1.2, (1, 1, 0))
    mass = 0.3
    momentum_squared = sum(component**2 for component in frame.momentum_at)
    k_squared = Q2_MAX * (2.0 * np.pi / frame.length_at) ** 2
    boundary = float(np.sqrt(momentum_squared + 4.0 * (mass**2 + k_squared)))
    return frame, mass, boundary


def _constant_scan(monkeypatch, frame, mass, window):
    monkeypatch.setattr(jls_matrix, "_free_poles", lambda *args: [])
    monkeypatch.setattr(
        jls_matrix,
        "row_matrix",
        lambda *args, **kwargs: np.array([[1.0]]),
    )
    return quantization_roots(
        window,
        frame,
        ((mass, mass),),
        SECTORS,
        BLOCKS,
        group=oh_group(),
        irrep="A1g",
        samples=4,
    )


def test_scan_accepts_upper_window_endpoint_just_inside_zeta_domain(monkeypatch):
    frame, mass, boundary = _moving_q2_upper_energy()
    supported_high = float(np.nextafter(boundary, 0.0))
    point = two_body_point(
        energy_lab_at=supported_high,
        mass1_at=mass,
        mass2_at=mass,
        frame=frame,
    )
    assert point.q_squared <= Q2_MAX
    assert _constant_scan(monkeypatch, frame, mass, (0.61, supported_high)) == ()


def test_out_of_domain_sliver_past_upper_window_endpoint_fails_preflight(monkeypatch):
    """The endpoint padding must not hide an unsupported sliver at the window end."""

    frame, mass, boundary = _moving_q2_upper_energy()
    unsupported_high = float(np.nextafter(boundary, np.inf))
    assert two_body_point(
        energy_lab_at=unsupported_high,
        mass1_at=mass,
        mass2_at=mass,
        frame=frame,
    ).q_squared > Q2_MAX
    monkeypatch.setattr(jls_matrix, "_free_poles", lambda *args: [])
    monkeypatch.setattr(
        jls_matrix,
        "row_matrix",
        lambda *args, **kwargs: np.array([[1.0]]),
    )

    with pytest.raises(RootScanError, match="q2=.*outside supported range") as caught:
        quantization_roots(
            (0.61, unsupported_high),
            frame,
            ((mass, mass),),
            SECTORS,
            BLOCKS,
            group=oh_group(),
            irrep="A1g",
            samples=4,
        )
    assert caught.value.kind == "domain"
    assert caught.value.energy == unsupported_high
    assert caught.value.interval == (0.61, unsupported_high)
    assert isinstance(caught.value.cause, ValueError)


@pytest.mark.parametrize("root", [0.6, 0.9])
def test_scan_includes_roots_at_closed_window_endpoints(monkeypatch, root):
    monkeypatch.setattr(jls_matrix, "_free_poles", lambda *args: [])
    monkeypatch.setattr(
        jls_matrix,
        "row_matrix",
        lambda energy, *args, **kwargs: np.array([[energy - root]]),
    )
    assert _scan(samples=4, window=(0.6, 0.9)) == (root,)


def test_split_edge_does_not_hide_root_inside_padded_boundary_band(monkeypatch):
    breakpoint = 0.75
    root = breakpoint + 5e-9
    monkeypatch.setattr(jls_matrix, "_free_poles", lambda *args: [])
    monkeypatch.setattr(
        jls_matrix,
        "row_matrix",
        lambda energy, *args, **kwargs: np.array([[energy - root]]),
    )
    result = quantization_roots(
        (0.6, 0.9),
        FRAME,
        MASSES,
        SECTORS,
        BLOCKS,
        group=oh_group(),
        irrep="A1g",
        samples=8,
        breakpoints_at2=(breakpoint**2,),
    )
    assert result == pytest.approx((root,), abs=1e-12)


def test_nonfinite_determinant_is_a_scan_failure(monkeypatch):
    """A NaN matrix result cannot be interpreted as an empty scan cell."""

    monkeypatch.setattr(jls_matrix, "_free_poles", lambda *args: [])
    monkeypatch.setattr(
        jls_matrix,
        "row_matrix",
        lambda *args, **kwargs: np.array([[np.nan]]),
    )

    with pytest.raises(RootScanError, match="non-finite root determinant") as caught:
        _scan(samples=4)
    assert caught.value.kind == "determinant"
    assert caught.value.energy is not None
    assert isinstance(caught.value.cause, ValueError)


def test_finite_logdet_scaling_is_continuous_and_stays_positive(monkeypatch):
    """Softplus scaling has no zero discontinuity and protects underflow."""

    from scipy import optimize

    monkeypatch.setattr(jls_matrix, "_free_poles", lambda *args: [])
    probe_logs = {0.76: -1000.0, 0.77: -1e-6, 0.78: 0.0, 0.79: 1e-6}

    def fake_row(energy, *args, **kwargs):
        # The first field supplies opposite endpoint signs; the second lets
        # the fake slogdet probe the magnitude transform at selected values.
        return np.array([[energy - 0.65, probe_logs.get(float(energy), -1.0)]])

    monkeypatch.setattr(jls_matrix, "row_matrix", fake_row)
    monkeypatch.setattr(
        jls_matrix.np.linalg,
        "slogdet",
        lambda value: (np.sign(value[0, 0]), float(value[0, 1])),
    )
    transformed = []

    def inspect_bracket(function, left, right, **kwargs):
        transformed.extend(function(point) for point in probe_logs)
        return 0.65

    monkeypatch.setattr(optimize, "brentq", inspect_bracket)
    assert _scan(samples=4) == (0.65,)

    smallest = float(np.nextafter(0.0, 1.0))
    expected = [
        max(float(np.logaddexp(0.0, log_abs)), smallest)
        for log_abs in probe_logs.values()
    ]
    assert transformed[0] == smallest
    assert np.allclose(transformed, expected, rtol=0.0, atol=0.0)
    assert transformed[1] < transformed[2] < transformed[3]


def test_refinement_failure_is_not_downgraded_to_a_skipped_bracket(monkeypatch):
    """A refinement error must not be swallowed by the ValueError fallback."""

    from scipy import optimize

    monkeypatch.setattr(jls_matrix, "_free_poles", lambda *args: [])
    monkeypatch.setattr(
        jls_matrix,
        "row_matrix",
        lambda energy, *args, **kwargs: np.array([[energy - 0.65]]),
    )

    def fail_refinement(*args, **kwargs):
        raise RootScanError("interior determinant evaluation failed", energy=0.65)

    monkeypatch.setattr(optimize, "brentq", fail_refinement)
    with pytest.raises(RootScanError, match="interior determinant evaluation failed"):
        _scan(samples=4)


def test_free_pole_split_does_not_turn_a_pole_sign_flip_into_a_root(monkeypatch):
    """The automatic free-pole edge remains an exclusion boundary."""

    pole = 0.75
    monkeypatch.setattr(jls_matrix, "_free_poles", lambda *args: [pole])

    def pole_sign_flip(energy, *args, **kwargs):
        # This deliberately has no zero: the sign change belongs to the pole
        # itself and must be separated by the scanner's free-pole edge.
        return np.array([[-1.0 if energy < pole else 1.0]])

    monkeypatch.setattr(jls_matrix, "row_matrix", pole_sign_flip)
    assert _scan(samples=8) == ()


def test_tangent_zero_remains_an_explicit_sign_change_limitation(monkeypatch):
    """A bounded sign-change scan does not invent an even-multiplicity root."""

    monkeypatch.setattr(jls_matrix, "_free_poles", lambda *args: [])
    tangent = 0.75

    monkeypatch.setattr(
        jls_matrix,
        "row_matrix",
        lambda energy, *args, **kwargs: np.array([[(energy - tangent) ** 2]]),
    )
    # The grid deliberately does not land on ``tangent``.  All sampled signs
    # are positive, so there is no mathematically justified bracket from which
    # to refine a root.
    assert _scan(samples=8) == ()
