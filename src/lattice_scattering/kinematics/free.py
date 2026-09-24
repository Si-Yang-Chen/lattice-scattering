"""Finite, budgeted enumeration of labelled noninteracting two-body states."""
from itertools import product
from math import ceil, isfinite, pi, sqrt
from numbers import Integral
from . import LatticeFrame, _positive


def free_two_body_states(*, mass1_at, mass2_at, frame, energy_max_at, max_vectors):
    """All labelled momentum pairs (n,d-n) with lab energy <= energy_max_at.

    Uses continuum dispersion in temporal lattice units, as the interacting
    kinematics. No irrep projection, exchange symmetrization or spin counting.
    Equal energies retain separate momentum labels; no tolerance clustering.
    A conservative cube follows from E1 <= Emax-m2, with one-cell rounding pad.
    Rejects an insufficient budget before enumerating any momentum vectors.
    """
    m1=_positive(mass1_at,'mass1_at');m2=_positive(mass2_at,'mass2_at')
    maximum=_positive(energy_max_at,'energy_max_at')
    if not isinstance(frame,LatticeFrame):
        raise ValueError('LatticeFrame required')
    if isinstance(max_vectors,bool) or not isinstance(max_vectors,Integral) or max_vectors<1:
        raise ValueError('positive integer vector budget required')
    if maximum<m1+m2:
        return {'states':[], 'examined_vectors':0, 'cube_radius':0}
    pmax2=max(0.,(maximum-m2-m1)*(maximum-m2+m1))
    bound=frame.length_at*sqrt(pmax2)/(2*pi)
    if not isfinite(bound):
        raise ValueError('nonfinite enumeration bound')
    radius=ceil(bound)+1
    count=(2*radius+1)**3
    if count>max_vectors:
        raise ValueError(f'free-spectrum budget requires {count} vectors')
    unit=2*pi/frame.length_at
    states=[]
    for n in product(range(-radius,radius+1),repeat=3):
        other=tuple(int(d-v) for d,v in zip(frame.d,n))
        e1=sqrt(m1*m1+unit*unit*sum(v*v for v in n))
        e2=sqrt(m2*m2+unit*unit*sum(v*v for v in other))
        energy=e1+e2
        if energy<=maximum:
            states.append({'n1':n,'n2':other,'energy_lab_at':energy})
    states.sort(key=lambda row:(row['energy_lab_at'],row['n1']))
    return {'states':states,'examined_vectors':count,'cube_radius':radius}
