import numpy as np
import pytest
from lattice_scattering.symmetry.angular_momentum import coupled_j_basis,total_j_projectors
from lattice_scattering.symmetry.rotations import spin_rotation


def test_singlet_phase_and_triplet():
    singlet=coupled_j_basis(1,1,0)
    np.testing.assert_allclose(singlet[:,0],[0,-1/np.sqrt(2),1/np.sqrt(2),0],atol=1e-15)
    triplet=coupled_j_basis(1,1,2)
    np.testing.assert_allclose(triplet,[[1,0,0],[0,1/np.sqrt(2),0],[0,1/np.sqrt(2),0],[0,0,1]],atol=1e-15)


@pytest.mark.parametrize('a,b',[(0,3),(2,1),(4,2),(4,3)])
def test_coupled_basis_projectors_and_rotation(a,b):
    q=np.array([.7,.2,-.3,.4]);q/=np.linalg.norm(q)
    rotation=np.kron(spin_rotation(q,a),spin_rotation(q,b))
    projectors=total_j_projectors(a,b)
    for j,projector in projectors.items():
        basis=coupled_j_basis(a,b,j)
        np.testing.assert_allclose(basis@basis.conj().T,projector,atol=1e-13)
        np.testing.assert_allclose(rotation@basis,basis@spin_rotation(q,j),atol=1e-13)
        assert not basis.flags.writeable


def test_coupled_basis_rejects_invalid_j_and_budget():
    with pytest.raises(ValueError):coupled_j_basis(2,1,2)
    with pytest.raises(ValueError):coupled_j_basis(4,3,3,max_dimension=19)
    with pytest.raises(ValueError):coupled_j_basis(True,1,0)
