import numpy as np
import pytest
from lattice_scattering.symmetry.solid_harmonics import solid_harmonics


def test_low_degree_cartesian_normalization():
    v=np.array([[.2,-.7,1.1],[0,0,0],[1,2,3.]])
    x,y,z=v.T;r2=np.sum(v*v,axis=1)
    np.testing.assert_allclose(solid_harmonics(v,0)[:,0],1/np.sqrt(4*np.pi))
    np.testing.assert_allclose(solid_harmonics(v,1)[:,2],-np.sqrt(3/(8*np.pi))*(x+1j*y))
    np.testing.assert_allclose(solid_harmonics(v,2)[:,2],np.sqrt(5/(16*np.pi))*(3*z*z-r2))
    np.testing.assert_allclose(solid_harmonics(v,4)[:,4],3/(16*np.sqrt(np.pi))*(35*z**4-30*z*z*r2+3*r2*r2))


@pytest.mark.parametrize('ell',range(1,9))
def test_addition_theorem_homogeneity_parity_and_origin(ell):
    v=np.random.default_rng(47).normal(size=(7,3));a=solid_harmonics(v,ell)
    expected=(2*ell+1)/(4*np.pi)*np.sum(v*v,axis=1)**ell
    np.testing.assert_allclose(np.sum(abs(a)**2,axis=1),expected,rtol=2e-14)
    np.testing.assert_allclose(solid_harmonics(-2*v,ell),(-2)**ell*a,rtol=2e-14)
    np.testing.assert_array_equal(solid_harmonics([[0,0,0]],ell),0)


def test_invalid_degree_and_budget():
    from lattice_scattering.symmetry.solid_harmonics import MAX_DEGREE
    for degree in (-1,MAX_DEGREE+1,True,.5):
        with pytest.raises(ValueError):solid_harmonics([[0,0,0]],degree)
    with pytest.raises(ValueError,match='budget'):solid_harmonics([[1,2,3]],4,max_values=8)


@pytest.mark.parametrize('ell',[1,2,4,8])
def test_active_rotation_uses_conjugate_representation(ell):
    from lattice_scattering.symmetry.rotations import spin_rotation,quaternion_rotation
    q=np.array([.7,.2,-.3,.4]);q/=np.linalg.norm(q)
    v=np.random.default_rng(9).normal(size=(5,3));v/=np.linalg.norm(v,axis=1)[:,None]
    rotated=solid_harmonics(v@quaternion_rotation(q).T,ell)
    expected=solid_harmonics(v,ell)@spin_rotation(q,2*ell).conj().T
    np.testing.assert_allclose(rotated,expected,atol=3e-14)
