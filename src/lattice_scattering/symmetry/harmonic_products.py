"""Finite spherical-harmonic product expansion by exact-order angular quadrature."""
from numbers import Integral
import numpy as np
from .solid_harmonics import MAX_DEGREE, solid_harmonics


def conjugate_product_coefficients(ell_left,ell_right,*,max_entries=1000000):
    """Expand conj(Y_left,m) Y_right,n = sum C[m,n,L,M] Y_L,M.

    Returns coefficients (2*l+1,2*r+1,(l+r+1)^2) and explicit (L,M)
    column labels. Degrees up to ``MAX_DEGREE`` are supported.
    Gauss-Legendre times equispaced azimuth quadrature is exact at the
    required polynomial/Fourier order, up to floating-point roundoff.
    These are angular coefficients, not a finite-volume normalization.
    """
    for ell in (ell_left,ell_right):
        if isinstance(ell,bool) or not isinstance(ell,Integral) or not 0<=ell<=MAX_DEGREE:
            raise ValueError(f'integer input harmonic degrees 0..{MAX_DEGREE} required')
    maximum=ell_left+ell_right
    labels=tuple((ell,m) for ell in range(maximum+1) for m in range(-ell,ell+1))
    count=(2*ell_left+1)*(2*ell_right+1)*len(labels)
    if isinstance(max_entries,bool) or not isinstance(max_entries,Integral) or max_entries<count:
        raise ValueError('harmonic product exceeds entry budget')
    z,weights=np.polynomial.legendre.leggauss(maximum+1)
    phi=2*np.pi*np.arange(2*maximum+1)/(2*maximum+1)
    radius=np.sqrt(1-z*z)
    vectors=np.stack(np.broadcast_arrays(radius[:,None]*np.cos(phi),radius[:,None]*np.sin(phi),z[:,None]),axis=-1).reshape(-1,3)
    weights=np.repeat(weights, len(phi))*2*np.pi/len(phi)
    left=solid_harmonics(vectors,ell_left);right=solid_harmonics(vectors,ell_right)
    target=np.concatenate([solid_harmonics(vectors,ell) for ell in range(maximum+1)],axis=1)
    coefficients=np.einsum('qi,qj,qk,q->ijk',left.conj(),right,target.conj(),weights,optimize=True)
    # Exact selection rules suppress roundoff in forbidden sectors.
    for i,m in enumerate(range(-ell_left,ell_left+1)):
        for j,n in enumerate(range(-ell_right,ell_right+1)):
            for k,(ell,order) in enumerate(labels):
                if order!=n-m or ell<abs(ell_left-ell_right) or (ell+ell_left+ell_right)%2:
                    coefficients[i,j,k]=0
    coefficients.setflags(write=False)
    return coefficients,labels
