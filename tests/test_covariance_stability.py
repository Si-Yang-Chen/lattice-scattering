import numpy as np
import pytest
from lattice_scattering.fitting.covariance import covariance_from_jacobian


def test_rotated_ill_conditioned_information_matches_known_covariance():
    angle=.6
    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    left=np.array([[1.,0.],[0.,1.],[1.,0.],[0.,1.]])/np.sqrt(2)
    jacobian=left@np.diag([1.,1e-8])@rotation.T
    actual,condition=covariance_from_jacobian(jacobian)
    expected=rotation@np.diag([1.,1e16])@rotation.T
    assert actual==pytest.approx(expected,rel=2e-8)
    assert condition==pytest.approx(1e8,rel=1e-8)


@pytest.mark.parametrize('jacobian',[np.ones((4,2)),np.zeros((4,2)),np.ones((1,2))])
def test_rank_deficiency_does_not_become_a_truncated_pseudoinverse(jacobian):
    with pytest.raises(RuntimeError,match='rank-deficient'):
        covariance_from_jacobian(jacobian)
