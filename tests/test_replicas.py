import numpy as np
import pytest
from lattice_scattering.data.replicas import ReplicaTable


def test_delete_one_means_reproduce_covariance_of_mean():
    raw=np.array([[1.,3.],[2.,5.],[4.,1.],[7.,2.]])
    replicas=(raw.sum(axis=0)-raw)/(len(raw)-1)
    table=ReplicaTable(replicas,('a','b','c','d'),('x','y'),'jackknife','ensemble-A')
    assert table.covariance()==pytest.approx(np.cov(raw,rowvar=False,ddof=1)/len(raw),abs=1e-14)


def test_pairing_reorders_samples_and_observables():
    a=ReplicaTable([[1.,2.],[3.,6.],[5.,8.]],('a','b','c'),('x','y'),'bootstrap','E')
    b=ReplicaTable([[8.,5.],[2.,1.],[6.,3.]],('c','a','b'),('y','x'),'bootstrap','E')
    delta=a.paired_difference(b)
    assert delta.covariance()==pytest.approx(np.zeros((2,2)))
    assert a.covariance()==pytest.approx(np.cov(a.values,rowvar=False,ddof=1))
    wrong=ReplicaTable(b.values,b.sample_ids,b.observable_ids,'bootstrap','other')
    with pytest.raises(ValueError,match='matching ensemble'):
        a.paired_difference(wrong)


def test_no_central_row_or_duplicate_identity_is_inferred():
    with pytest.raises(ValueError,match='unique'):
        ReplicaTable([[1],[2]],('same','same'),('x',),'jackknife','E')
    with pytest.raises(ValueError,match='kind'):
        ReplicaTable([[1],[2]],('a','b'),('x',),'unknown','E')
