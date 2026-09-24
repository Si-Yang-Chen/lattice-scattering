"""Identity-preserving real replica tables; central estimates are separate."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ReplicaTable:
    values: np.ndarray
    sample_ids: tuple
    observable_ids: tuple
    kind: str
    ensemble: str

    def __post_init__(self):
        if self.kind not in ('jackknife','bootstrap') or not isinstance(self.ensemble,str) or not self.ensemble:
            raise ValueError('explicit resampling kind and ensemble identity required')
        if np.iscomplexobj(self.values):
            raise ValueError('split complex observables into real components explicitly')
        a=np.array(self.values,dtype=float,copy=True)
        samples=tuple(self.sample_ids);observables=tuple(self.observable_ids)
        if a.ndim!=2 or a.shape!=(len(samples),len(observables)) or len(samples)<2 or not len(observables) or not np.all(np.isfinite(a)):
            raise ValueError('finite sample-by-observable table with at least two samples required')
        for ids in (samples,observables):
            if any(not isinstance(x,str) or not x for x in ids) or len(set(ids))!=len(ids):
                raise ValueError('nonempty unique string identities required')
        a.setflags(write=False)
        object.__setattr__(self,'values',a)
        object.__setattr__(self,'sample_ids',samples)
        object.__setattr__(self,'observable_ids',observables)

    def covariance(self):
        """Unweighted delete-one jackknife or ordinary bootstrap covariance.

        Does not interpret a row as a central estimate. Unequal block sizes
        and weighted resampling require a different estimator.
        """
        n=len(self.sample_ids)
        delta=self.values-self.values.mean(axis=0)
        factor=(n-1)/n if self.kind=='jackknife' else 1/(n-1)
        return factor*(delta.T@delta)

    def paired_difference(self,other):
        """Align by IDs and preserve correlations; refuse different ensembles."""
        if self.kind!=other.kind or self.ensemble!=other.ensemble or set(self.sample_ids)!=set(other.sample_ids) or set(self.observable_ids)!=set(other.observable_ids):
            raise ValueError('paired comparison requires matching ensemble, kind and identities')
        samples={key:i for i,key in enumerate(other.sample_ids)}
        observables={key:i for i,key in enumerate(other.observable_ids)}
        aligned=other.values[np.ix_([samples[x] for x in self.sample_ids],[observables[x] for x in self.observable_ids])]
        return ReplicaTable(self.values-aligned,self.sample_ids,self.observable_ids,self.kind,self.ensemble)
