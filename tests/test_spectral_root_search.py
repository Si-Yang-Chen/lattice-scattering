"""Analytic spectra challenge sign-only scans, degeneracies and pole handling."""
import numpy as np
import pytest

from lattice_scattering.finite_volume import quantization_roots, RootScanError, matrix_root_residual
from lattice_scattering.finite_volume import jls_matrix
from lattice_scattering.kinematics import LatticeFrame
from lattice_scattering.symmetry.groups import oh_group


def scan(monkeypatch, matrix, **settings):
    monkeypatch.setattr(jls_matrix, "row_matrix", lambda e, *a, **kw: matrix(e))
    monkeypatch.setattr(jls_matrix, "_free_poles", lambda *a: [])
    return quantization_roots((0.6, 0.9), LatticeFrame(24, 1, (0, 0, 0)),
                              ((0.3, 0.3),), ((0, 0),), {0: lambda s: np.eye(1)},
                              group=oh_group(), irrep="A1g", samples=8, **settings)


@pytest.mark.parametrize("subdivisions", [0, 1, 2])
def test_tangent(monkeypatch, subdivisions):
    assert scan(monkeypatch, lambda e: np.diag([(e-0.75123)**2, 1.0]),
                subdivisions=subdivisions) == pytest.approx([0.75123], abs=2e-12)


@pytest.mark.parametrize("gap", [0, 1e-6, 0.01])
def test_rotating_basis_degenerate_and_dense_roots(monkeypatch, gap):
    def matrix(e):
        angle = 8*e
        u = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        return u @ np.diag([e-0.75123, e-0.75123-gap]) @ u.T
    expected = [0.75123] if gap == 0 else [0.75123, 0.75123+gap]
    assert scan(monkeypatch, matrix) == pytest.approx(expected, abs=2e-12)


def test_pair_in_one_scalar_cell(monkeypatch):
    roots = (0.751, 0.7511)
    assert scan(monkeypatch, lambda e: np.array([[(e-roots[0])*(e-roots[1])]])) == pytest.approx(roots, abs=2e-12)


def test_oppositely_crossing_eigenvalues(monkeypatch):
    assert scan(monkeypatch, lambda e: np.diag([e-0.75123, 0.75123-e])) == pytest.approx([0.75123], abs=2e-12)


def test_positive_minimum_and_small_constant_are_not_roots(monkeypatch):
    assert scan(monkeypatch, lambda e: np.diag([(e-0.75)**2+1e-6, 1e-14])) == ()


def test_undeclared_discontinuity_rejected_by_residual(monkeypatch):
    with pytest.raises(RootScanError, match="residual"):
        scan(monkeypatch, lambda e: np.array([[-1.0 if e < 0.753 else 1.0]]))


def test_declared_model_pole_is_not_a_root(monkeypatch):
    assert scan(monkeypatch, lambda e: np.array([[1/(e-0.753)]]), breakpoints_at2=[0.753**2]) == ()


def test_near_free_pole_outside_exclusion_band(monkeypatch):
    monkeypatch.setattr(jls_matrix, "row_matrix", lambda e, *a, **kw: np.array([[(e-0.75001)/(e-0.75)]]))
    monkeypatch.setattr(jls_matrix, "_free_poles", lambda *a: [0.75])
    result = quantization_roots((0.6, 0.9), LatticeFrame(24, 1, (0,0,0)), ((0.3,0.3),),
                               ((0,0),), {0: lambda s: np.eye(1)}, group=oh_group(), irrep="A1g", samples=8)
    assert result == pytest.approx([0.75001], abs=1e-12)


def test_residual_and_nonhermitian_validation():
    assert matrix_root_residual(np.diag([1e-8, 100])) == pytest.approx(1e-10)
    with pytest.raises(ValueError, match="non-Hermitian"):
        matrix_root_residual([[1, 2], [0, 1]])


@pytest.mark.parametrize("settings", [{"subdivisions": True}, {"subdivisions": 9}, {"residual_tol": 0}, {"root_method": "guess"}])
def test_invalid_options(monkeypatch, settings):
    with pytest.raises(ValueError):
        scan(monkeypatch, lambda e: np.eye(1), **settings)


def test_spectral_refinement_failure_is_fatal(monkeypatch):
    from lattice_scattering.finite_volume import root_search
    def fail(*a, **kw):
        raise RuntimeError("refiner failed")
    monkeypatch.setattr(root_search, "brentq", fail)
    with pytest.raises(RootScanError, match="refiner failed"):
        scan(monkeypatch, lambda e: np.array([[e-0.75123]]))


def test_large_finite_diagonal_does_not_overflow_during_symmetrization(monkeypatch):
    assert scan(monkeypatch, lambda e: np.array([[1e308]])) == ()


def test_nonfinite_eigensolver_result_is_fatal(monkeypatch):
    monkeypatch.setattr(np.linalg, "eigvalsh", lambda m: np.array([np.nan]))
    with pytest.raises(RootScanError, match="non-finite quantization eigenvalues"):
        scan(monkeypatch, lambda e: np.eye(1))
