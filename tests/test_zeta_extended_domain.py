"""Independent scalar Ewald/Yukawa references and extended-boost regressions."""
from itertools import product

import numpy as np
import pytest
from scipy.integrate import quad

from lattice_scattering.finite_volume import harmonic_zeta_wide, row_matrix, quantization_roots, matrix_root_residual, RootScanError
from lattice_scattering.finite_volume.zeta import GAMMA_MAX
from lattice_scattering.kinematics import LatticeFrame, two_body_point
from lattice_scattering.symmetry import little_group


def scalar_reference(q2, d, gamma, alpha, radius=14):
    """Independent scalar Ewald expression, scalar quad and explicit Y00.

    Uses splitting t0=0.37, outside the production split domain, and fixed
    cubes. Does not call production grids/harmonics/Poisson helpers.
    """
    t0 = 0.37
    n = np.array(list(product(range(-radius, radius+1), repeat=3)), dtype=float)
    unit = np.array(d)/np.linalg.norm(d)
    shifted = n - alpha*np.array(d)
    r = shifted + (1/gamma-1)*np.outer(shifted@unit, unit)
    delta = np.sum(r*r, axis=1) - q2
    direct = np.sum(np.exp(-t0*delta)/delta)/np.sqrt(4*np.pi)
    images = n[np.any(n != 0, axis=1)]
    w = images + (gamma-1)*np.outer(images@unit, unit)
    w2 = np.sum(w*w, axis=1)
    phases = np.cos(2*np.pi*alpha*(images@np.array(d)))
    def integrand(t):
        if t == 0:
            return 0.
        return np.sum(phases*np.exp(t*q2-np.pi**2*w2/t))/t**1.5
    image = quad(integrand, 0, t0, epsabs=1e-12, epsrel=1e-12)[0]
    zero = quad(lambda v: 2*np.expm1(q2*v*v)/(v*v) if v else 2*q2,
                0, np.sqrt(t0), epsabs=1e-12, epsrel=1e-12)[0] - 2/np.sqrt(t0)
    return direct + gamma*np.pi/2*(image+zero)


def yukawa_reference(kappa, d, gamma, alpha):
    """Closed-form Fourier transform at q2=-kappa^2; no heat splitting."""
    n = np.array(list(product(range(-8, 9), repeat=3)), dtype=float)
    n = n[np.any(n != 0, axis=1)]
    unit = np.array(d)/np.linalg.norm(d)
    w = n + (gamma-1)*np.outer(n@unit, unit)
    length = np.linalg.norm(w, axis=1)
    images = np.sum(np.cos(2*np.pi*alpha*(n@np.array(d)))*np.exp(-2*np.pi*kappa*length)/length)
    return -gamma*np.pi**1.5*kappa + gamma*np.sqrt(np.pi)/2*images


@pytest.mark.parametrize("gamma", [1.5, 1.5137903686, 1.52])
@pytest.mark.parametrize("q2", [0.0135778014, -1.3, 3.7])
def test_independent_scalar_reference(gamma, q2):
    kwargs = dict(d=(0,0,2), gamma=gamma, alpha=0.5)
    reference = scalar_reference(q2, **kwargs)
    assert scalar_reference(q2, **kwargs, radius=18) == pytest.approx(reference, abs=2e-11)
    for split in (0.5, 1., 2.):
        value = harmonic_zeta_wide(q2, 0, split=split, **kwargs)
        assert value.values[0] == pytest.approx(reference, rel=2e-10, abs=2e-10)
        assert value.diagnostics['stable']


@pytest.mark.parametrize("d,alpha", [((0,0,2), 0.5), ((1,2,3), 0.17), ((6,0,0), 1.)])
def test_negative_q2_yukawa_oracle(d, alpha):
    kwargs = dict(d=d, alpha=alpha, gamma=1.52)
    for kappa in (1., np.sqrt(48.)):
        result = harmonic_zeta_wide(-kappa*kappa, 0, **kwargs)
        assert result.values[0] == pytest.approx(yukawa_reference(kappa, **kwargs), rel=3e-11, abs=3e-11)


@pytest.mark.slow
@pytest.mark.parametrize("ell", range(13))
@pytest.mark.parametrize("d,q2,alpha", [((0,0,2), 0.0135778014, 0.5), ((1,2,3), -3., 0.17), ((6,0,0), 4., 0.83)])
def test_all_degrees_split_and_cutoff_stability(ell, d, q2, alpha):
    kwargs = dict(d=d, gamma=1.52, alpha=alpha)
    a = harmonic_zeta_wide(q2, ell, split=0.5, **kwargs)
    b = harmonic_zeta_wide(q2, ell, split=2., cutoff=12, **kwargs)
    np.testing.assert_allclose(a.values, b.values, rtol=2e-9, atol=2e-8)
    assert a.diagnostics['stable'] and b.diagnostics['stable']


def test_source_condition_and_full_window_root_scan():
    # Source-inspired kinematics only; synthetic constant inverse, no paper fit.
    frame = LatticeFrame(48, 1., (0,0,2))
    mass = 5.48/48
    energy = 0.34872
    point = two_body_point(energy_lab_at=energy, mass1_at=mass, mass2_at=mass, frame=frame)
    assert 1.5 < point.gamma < GAMMA_MAX
    kwargs = dict(group=little_group(frame.d), irrep="A1")
    # Force a synthetic root at the old blocked condition, then check independent
    # scan density and its actual matrix residual rather than just root count.
    zero = {0: lambda s: np.zeros((1,1))}
    box = -row_matrix(energy, frame, ((mass,mass),), ((0,0),), zero, **kwargs)[0,0].real
    amplitude = {0: lambda s: np.array([[box/frame.length_at*(2*np.pi)]])}
    # R is multiplied by length/(2pi) in scale weighting for S waves.
    for samples in (8, 17):
        roots = quantization_roots((0.348, 0.3495), frame, ((mass,mass),), ((0,0),), amplitude,
                                   samples=samples, **kwargs)
        assert roots == pytest.approx([energy], abs=2e-12)
        for root in roots:
            matrix = row_matrix(root, frame, ((mass,mass),), ((0,0),), amplitude, **kwargs)
            assert matrix_root_residual(matrix) < 1e-10
            np.testing.assert_allclose(matrix, matrix.conj().T, atol=1e-12)
    with pytest.raises(RootScanError, match="gamma"):
        quantization_roots((0.34, 0.3495), frame, ((mass,mass),), ((0,0),), amplitude, **kwargs)


def test_extended_upper_bound_still_rejects():
    with pytest.raises(ValueError, match="gamma"):
        harmonic_zeta_wide(0.1, 0, d=(0,0,2), gamma=np.nextafter(GAMMA_MAX, np.inf), alpha=0.5)


@pytest.mark.slow
@pytest.mark.parametrize("d", [(0,0,0), (0,0,1), (0,1,1), (1,1,1), (0,0,2)])
def test_five_frame_synthetic_windows_including_previously_blocked_frame(d):
    frame = LatticeFrame(48, 1., d)
    mass = 5.48/48
    momentum2 = sum(x*x for x in frame.momentum_at)
    low = 0.348 if d == (0,0,2) else np.sqrt(momentum2 + (2*mass+0.0005)**2)
    high = np.sqrt(momentum2+0.35**2)
    kwargs = dict(group=little_group(d), irrep="A1g" if d == (0,0,0) else "A1")
    amplitudes = {0: lambda s: np.array([[0.2]])}
    results = []
    for samples in (8, 19):
        roots = quantization_roots((low, high), frame, ((mass,mass),), ((0,0),), amplitudes,
                                   samples=samples, **kwargs)
        assert roots
        results.append(roots)
        for root in roots:
            matrix = row_matrix(root, frame, ((mass,mass),), ((0,0),), amplitudes, **kwargs)
            assert matrix_root_residual(matrix) < 1e-10
    assert results[0] == pytest.approx(results[1], abs=2e-12)
