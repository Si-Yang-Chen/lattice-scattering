import numpy as np
import pytest
from lattice_scattering.symmetry.harmonic_products import conjugate_product_coefficients
from lattice_scattering.symmetry.solid_harmonics import solid_harmonics


@pytest.mark.parametrize('left,right',[(0,0),(0,2),(1,1),(1,2),(2,2),(3,4),(4,4)])
def test_product_reconstruction_away_from_quadrature_grid(left,right):
    vectors=np.random.default_rng(53).normal(size=(11,3))
    vectors/=np.linalg.norm(vectors,axis=1)[:,None]
    coefficients,labels=conjugate_product_coefficients(left,right)
    target=np.concatenate([solid_harmonics(vectors,ell) for ell in range(left+right+1)],axis=1)
    direct=solid_harmonics(vectors,left).conj()[:,:,None]*solid_harmonics(vectors,right)[:,None,:]
    np.testing.assert_allclose(np.einsum('ijk,qk->qij',coefficients,target),direct,atol=3e-14)
    if left==right:
        np.testing.assert_allclose(coefficients[:,:,0],np.eye(2*left+1)/np.sqrt(4*np.pi),atol=2e-14)


def test_budget_and_degree():
    from lattice_scattering.symmetry.solid_harmonics import MAX_DEGREE
    with pytest.raises(ValueError):conjugate_product_coefficients(MAX_DEGREE+1,0)
    with pytest.raises(ValueError,match='budget'):conjugate_product_coefficients(2,2,max_entries=1)
