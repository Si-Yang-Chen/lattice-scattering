"""Axial moving-frame Z_l0 and spinless A1 S/P block.

Heat-kernel formula: arXiv:1707.05817v1 Eq.7-10. Only d=(0,0,dz)
and ell=0,1,2 are exposed; this is not an arbitrary-spin implementation.
"""
from functools import lru_cache
import numpy as np
from scipy.integrate import quad_vec


@lru_cache(maxsize=8)
def _grid(cutoff):
    return (np.indices((2*cutoff+1,)*3)-cutoff).reshape(3,-1).T.astype(float)


def _harmonics(v):
    r2=np.sum(v*v,axis=1)
    z=v[:,2]
    return np.array([np.full(len(v),1/np.sqrt(4*np.pi)),
                     np.sqrt(3/(4*np.pi))*z,
                     np.sqrt(5/(16*np.pi))*(3*z*z-r2)])


def axial_zeta(q2, *, gamma, alpha, dz=1, split=1., cutoff=10):
    """Return real Z00,Z10,Z20 for r=(nx,ny,(nz-alpha*dz)/gamma).

    Real q² in [-4,4], gamma in [1,1.5], alpha in [0,1], |dz|<=2.
    Exact poles raise; nearby poles retain their divergence.
    """
    if not all(np.isfinite(x) for x in (q2,gamma,alpha,split)):
        raise ValueError('nonfinite moving geometry')
    if not (-4<=q2<=4 and 1<=gamma<=1.5 and 0<=alpha<=1 and .5<=split<=2):
        raise ValueError('outside supported moving geometry domain')
    if isinstance(dz,bool) or not isinstance(dz,int) or abs(dz)>2 or (dz==0 and gamma!=1):
        raise ValueError('invalid axial frame')
    if isinstance(cutoff,bool) or not isinstance(cutoff,int) or cutoff<10:
        raise ValueError('cutoff must be integer >=10')
    # Pad the direct cube for the compressed, shifted longitudinal direction.
    # A fixed cube is insufficient at large gamma/shift and small split.
    direct_cutoff=max(cutoff,int(np.ceil(abs(alpha*dz)+gamma*np.sqrt(40/split+max(q2,0)))))
    r=_grid(direct_cutoff).copy()
    r[:,2]=(r[:,2]-alpha*dz)/gamma
    delta=np.sum(r*r,axis=1)-q2
    if np.any(delta==0):
        raise ValueError('free-spectrum pole')
    direct=np.sum(_harmonics(r)*np.exp(-split*delta)/delta,axis=1)
    n=_grid(cutoff)
    nonzero=np.any(n!=0,axis=1)
    w=n[nonzero].copy()
    phase=np.exp(2j*np.pi*alpha*dz*w[:,2])
    w[:,2]*=gamma
    w2=np.sum(w*w,axis=1)
    weighted=_harmonics(w)*phase
    ell=np.arange(3)
    def integrand(v):
        if v==0:
            return np.array([2*q2,0,0,0],dtype=complex)
        images=np.sum(weighted*np.exp(q2*v*v-np.pi**2*w2/(v*v)),axis=1)*2/v**(2*ell+2)
        return np.concatenate(([2*np.expm1(q2*v*v)/(v*v)],images))
    integrals=quad_vec(integrand,0,np.sqrt(split),epsabs=1e-11,epsrel=1e-11)[0]
    value=direct+gamma*(1j**ell)*np.pi**(ell+1.5)*integrals[1:]
    value[0]+=gamma*np.pi*(-2/np.sqrt(split)+integrals[0])/2
    if np.max(np.abs(value.imag))>1e-9:
        raise ArithmeticError('unexpected imaginary Z_l0')
    return value.real


def axial_sp_box(q2, *, gamma, alpha, dz=1, **settings):
    """q-scaled real symmetric A1 ell<=1 block after a basis rephasing.

    Diagonal inverse K entries are q*cot(delta0), q³*cot(delta1).
    dz=0 is rejected: S and P belong to different parity irreps at rest.
    """
    if dz==0:
        raise ValueError('rest S/P waves belong to separate parity irreps')
    z0,z1,z2=axial_zeta(q2,gamma=gamma,alpha=alpha,dz=dz,**settings)
    return np.array([[z0,z1],[z1,q2*z0+2*z2/np.sqrt(5)]])/(gamma*np.pi**1.5)
