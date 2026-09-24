"""Independent on-shell and geometric checks; no legacy imports or outputs."""
from math import pi, sqrt

import pytest

from lattice_scattering.kinematics import LatticeFrame, shifted_vector, two_body_point


@pytest.mark.parametrize('d', [(0, 0, 0), (0, 0, 1), (1, 1, 0), (1, -2, 3)])
def test_recover_on_shell_relative_momentum(d):
    frame = LatticeFrame(24, 3.5, d)
    m1, m2, k = .31, .47, .12
    ecm = sqrt(m1*m1+k*k)+sqrt(m2*m2+k*k)
    energy = sqrt(ecm*ecm+sum(p*p for p in frame.momentum_at))
    point = two_body_point(energy_lab_at=energy, mass1_at=m1, mass2_at=m2, frame=frame)
    assert point.k_squared_at2 == pytest.approx(k*k, abs=1e-15)
    assert point.s_at2 == pytest.approx(ecm*ecm)
    assert point.gamma == pytest.approx(energy/ecm)
    swapped = two_body_point(energy_lab_at=energy, mass1_at=m2, mass2_at=m1, frame=frame)
    assert swapped.k_squared_at2 == pytest.approx(point.k_squared_at2)
    assert swapped.alpha == pytest.approx(1-point.alpha)


def test_equal_mass_subthreshold_and_threshold():
    frame = LatticeFrame(20, 2)
    for energy in [.8, 1., 1.2]:
        p = two_body_point(energy_lab_at=energy, mass1_at=.5, mass2_at=.5, frame=frame)
        assert p.k_squared_at2 == pytest.approx(energy**2/4-.25)
        assert p.alpha == .5
        assert p.gamma == 1
        assert shifted_vector((1, 2, 3), frame=frame, point=p) == (1, 2, 3)


def test_boost_preserves_transverse_and_compresses_parallel():
    frame = LatticeFrame(20, 2, (1, 1, 0))
    point = two_body_point(energy_lab_at=1.2, mass1_at=.3, mass2_at=.4, frame=frame)
    r = shifted_vector((3, 1, 2), frame=frame, point=point)
    assert r[2] == 2
    assert r[0]-r[1] == pytest.approx(2)
    assert r[0]+r[1] == pytest.approx((4-2*point.alpha)/point.gamma)
    assert frame.momentum_at[0] == pytest.approx(2*pi/40)


@pytest.mark.parametrize('sites,xi,d', [(0, 1, (0, 0, 0)), (20, 0, (0, 0, 0)), (20, 1, (.1, 0, 0)), (True, 1, (0, 0, 0))])
def test_reject_invalid_frame(sites, xi, d):
    with pytest.raises(ValueError):
        LatticeFrame(sites, xi, d)


@pytest.mark.parametrize('energy', [0, float('nan'), .01])
def test_reject_invalid_lab_energy(energy):
    with pytest.raises(ValueError):
        two_body_point(energy_lab_at=energy, mass1_at=.2, mass2_at=.3,
                       frame=LatticeFrame(16, 1, (1, 0, 0)))
