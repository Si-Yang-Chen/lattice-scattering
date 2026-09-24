"""Complex regular solid harmonics r^ell Y_ell,m with Condon--Shortley phase."""
from math import factorial, pi, sqrt
from numbers import Integral
import numpy as np

#: Highest polynomial degree this recurrence is validated for.
#:
#: The recurrence itself is general, but the floating-point accuracy of the
#: upward recursion degrades with degree and the ``factorial`` normalisation
#: overflows for very large ``ell``; 12 was reached without loss of accuracy in
#: high-degree covariance checks within the supported finite-volume domain.
MAX_DEGREE = 12


def solid_harmonics(vectors, ell, *, max_values=1000000):
    """Return (n,2*ell+1), m=-ell..ell, for finite real Cartesian vectors.

    Polynomial recurrence avoids spherical-coordinate singularities at the
    origin. The supported degree is 0..``MAX_DEGREE``; output allocation is
    budgeted.  This supplies harmonics only, not finite-volume sums or a box
    matrix.
    """
    if isinstance(ell,bool) or not isinstance(ell,Integral) or not 0<=ell<=MAX_DEGREE:
        raise ValueError(f'integer harmonic degree in 0..{MAX_DEGREE} required')
    if np.iscomplexobj(vectors): raise ValueError('finite real Cartesian vectors required')
    v=np.asarray(vectors,dtype=float)
    if v.ndim!=2 or v.shape[1]!=3 or not np.all(np.isfinite(v)):
        raise ValueError('finite real Cartesian vectors with shape (n,3) required')
    if isinstance(max_values,bool) or not isinstance(max_values,Integral) or max_values<0 or len(v)*(2*ell+1)>max_values:
        raise ValueError('solid harmonic output exceeds value budget')
    result=np.empty((len(v),2*ell+1),dtype=complex)
    x,y,z=v.T;r2=np.sum(v*v,axis=1)
    diagonal=np.ones(len(v),dtype=complex)
    with np.errstate(over='raise',invalid='raise'):
        for m in range(ell+1):
            if m: diagonal=-(2*m-1)*(x+1j*y)*diagonal
            lower=diagonal
            if ell==m: polynomial=lower
            else:
                upper=(2*m+1)*z*lower
                for degree in range(m+2,ell+1):
                    lower,upper=upper,((2*degree-1)*z*upper-(degree+m-1)*r2*lower)/(degree-m)
                polynomial=upper
            value=sqrt((2*ell+1)/(4*pi)*factorial(ell-m)/factorial(ell+m))*polynomial
            result[:,ell+m]=value
            result[:,ell-m]=(-1)**m*value.conj()
    if not np.all(np.isfinite(result)):raise ValueError('nonfinite solid harmonic result')
    return result
