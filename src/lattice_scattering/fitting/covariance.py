"""Local unscaled covariance without squaring the Jacobian condition number."""
import numpy as np


def covariance_from_jacobian(jacobian):
    if np.iscomplexobj(jacobian):raise ValueError('real residual Jacobian required')
    jacobian=np.asarray(jacobian,dtype=float)
    if jacobian.ndim!=2 or not all(jacobian.shape) or not np.all(np.isfinite(jacobian)):
        raise ValueError('finite nonempty residual Jacobian required')
    _,singular,vh=np.linalg.svd(jacobian,full_matrices=False)
    threshold=max(jacobian.shape)*np.finfo(float).eps*singular[0]
    if len(singular)<jacobian.shape[1] or singular[-1]<=threshold:
        raise RuntimeError('unidentifiable coupled fit: rank-deficient residual Jacobian')
    scaled=vh.T/singular
    covariance=scaled@scaled.T
    if not np.all(np.isfinite(covariance)):
        raise RuntimeError('nonfinite local covariance')
    return covariance,float(singular[0]/singular[-1])
