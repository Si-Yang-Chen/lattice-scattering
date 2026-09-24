import numpy as np
import pytest
from lattice_scattering.data.replicas import ReplicaTable
from lattice_scattering.fitting.replicas import fit_replicas


def test_correlated_nuisance_parameters_remain_paired():
    # A mass and energy share a fluctuation, cancelled by the fitted difference.
    table=ReplicaTable([[2.,1.],[3.,2.],[4.,3.]],('a','b','c'),('energy','mass'),'jackknife','E')
    output=fit_replicas(table,lambda sample,row:[row['energy']-row['mass']],parameter_ids=('difference',))
    assert output['complete']
    assert output['covariance']==[[0.]]
    assert output['central_estimate'] is None
    assert [r['sample_id'] for r in output['records']]==list(table.sample_ids)


def test_failed_replica_prevents_covariance_and_keeps_later_samples():
    table=ReplicaTable([[1.],[2.],[3.]],('a','b','c'),('energy',),'bootstrap','E')
    def fit(sample,row):
        if sample=='b': raise RuntimeError('root left its allowed interval')
        return [row['energy']]
    output=fit_replicas(table,fit,parameter_ids=('p',))
    assert not output['complete'] and output['covariance'] is None
    assert [r['status'] for r in output['records']]==['success','failed','success']
    assert output['records'][1]['error']=='root left its allowed interval'


def test_programming_errors_are_not_silenced():
    table=ReplicaTable([[1.],[2.]],('a','b'),('energy',),'bootstrap','E')
    with pytest.raises(KeyError):
        fit_replicas(table,lambda sample,row:[row['misspelled']],parameter_ids=('p',))
