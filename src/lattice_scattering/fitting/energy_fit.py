"""Shared correlated energy fit and explicit convergence diagnostics."""
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import least_squares
from .covariance import covariance_from_jacobian


class FitConvergenceError(RuntimeError):
    """A resumable last iterate, not an accepted fit or covariance estimate."""
    def __init__(self,result):
        self.result=result
        super().__init__('failed spectrum fit: '+result['diagnostics']['optimizer_message'])


def whitened_residual_floor(covariance_at2,energy_tolerance):
    """Norm of the whitened residual caused by the forward model's own precision.

    A local root finder returns an energy accurate only to its bracket tolerance, so
    the whitened residual cannot fall below the norm of that tolerance propagated
    through the same Cholesky factor used to whiten the fit.  This is a bound on the
    forward model, not a tolerance chosen to make a fit appear to converge.
    """
    covariance=np.asarray(covariance_at2,dtype=float)
    if covariance.ndim!=2 or covariance.shape[0]!=covariance.shape[1] or not len(covariance):
        raise ValueError('square nonempty covariance matrix required')
    if np.iscomplexobj(energy_tolerance) or np.ndim(energy_tolerance)!=0 or not np.isfinite(energy_tolerance) or energy_tolerance<=0:
        raise ValueError('positive finite energy tolerance required')
    chol=np.linalg.cholesky(covariance)
    return float(np.linalg.norm(solve_triangular(chol,float(energy_tolerance)*np.ones(len(covariance)),lower=True)))


def fit_energy_model(data,predict,initial,*,bounds,jacobian_scheme='2-point',difference_step=None,
                     max_evaluations=None,xtol=1e-11,ftol=1e-11,gtol=1e-8,positive_dof=True,covariance_audit=None,
                     residual_floor=None):
    if covariance_audit is not None and (not isinstance(covariance_audit,dict) or
            set(covariance_audit)!={'parameter_scales','relative_steps','max_evaluations'}):
        raise ValueError('covariance_audit requires parameter_scales, relative_steps, max_evaluations')
    if jacobian_scheme not in ('2-point','3-point'):
        raise ValueError('Jacobian scheme must be 2-point or 3-point')
    if difference_step is not None and (np.iscomplexobj(difference_step) or np.ndim(difference_step)!=0 or not np.isfinite(difference_step) or difference_step<=0):
        raise ValueError('positive finite relative difference step required')
    if max_evaluations is not None and (not isinstance(max_evaluations,(int,np.integer)) or isinstance(max_evaluations,bool) or max_evaluations<1):
        raise ValueError('positive integer evaluation budget required')
    if residual_floor is not None and (np.iscomplexobj(residual_floor) or np.ndim(residual_floor)!=0 or not np.isfinite(residual_floor) or residual_floor<0):
        raise ValueError('non-negative finite residual floor required')
    if np.iscomplexobj(initial):raise ValueError('real initial parameters required')
    initial=np.asarray(initial,dtype=float)
    if initial.ndim!=1 or not len(initial) or not np.all(np.isfinite(initial)):
        raise ValueError('finite parameter vector required')
    count=len(initial);levels=len(data.energies_at)
    if levels<count or (positive_dof and levels==count):
        raise ValueError('more levels than fitted parameters required')
    chol=np.linalg.cholesky(data.covariance_at2)
    def residual(theta):
        return solve_triangular(chol,predict(theta)-data.energies_at,lower=True)
    fit=least_squares(residual,initial,bounds=bounds,x_scale='jac',xtol=xtol,ftol=ftol,gtol=gtol,
                      jac=jacobian_scheme,diff_step=difference_step,max_nfev=max_evaluations)
    diagnostics={'active_bounds':fit.active_mask.tolist(),'optimizer_status':int(fit.status),
        'optimizer_message':fit.message,'optimizer_optimality':float(fit.optimality),
        'function_evaluations':int(fit.nfev),'jacobian_evaluations':None if fit.njev is None else int(fit.njev),
        'jacobian_scheme':jacobian_scheme,'relative_difference_step':difference_step,
        'max_function_evaluations':max_evaluations,'tolerances':{'xtol':xtol,'ftol':ftol,'gtol':gtol},
        'residual_norm':float(np.linalg.norm(fit.fun)),
        'residual_floor':None if residual_floor is None else float(residual_floor)}
    # An optimizer can exhaust its budget without satisfying gtol/ftol/xtol when the
    # forward model itself is only accurate to a finite tolerance: the residual then
    # plateaus at that precision and no criterion expressed in residual or step units
    # can be met.  Such a state is accepted only when the residual has demonstrably
    # reached the model's precision floor -- a physical bound, not a loosened tolerance.
    at_floor=(residual_floor is not None and diagnostics['residual_norm']<=residual_floor)
    result={'status':'converged','parameters':fit.x.tolist(),'chi2':float(fit.fun@fit.fun),'dof':levels-count,'diagnostics':diagnostics}
    diagnostics['convergence_reason']='optimizer' if fit.success else ('forward-model-precision-floor' if at_floor else None)
    if not fit.success and not at_floor:
        result.update(status='not_converged',predicted_energies_at=(data.energies_at+chol@fit.fun).tolist())
        diagnostics['covariance_scope']='no covariance: optimizer did not converge and the residual exceeds the forward-model precision floor'
        raise FitConvergenceError(result)
    covariance,condition=covariance_from_jacobian(fit.jac)
    errors=np.sqrt(np.diag(covariance))
    result.update(predicted_energies_at=predict(fit.x).tolist(),covariance=covariance.tolist())
    diagnostics.update(jacobian_condition=condition,parameter_correlation=(covariance/np.outer(errors,errors)).tolist(),
        covariance_method='SVD of whitened residual Jacobian; no singular-value truncation',
        covariance_scope=('local linear estimate at a residual accepted as lying within the forward-model '
                          'precision floor; does not include parameter-bound truncation, root tolerance or model uncertainty'
                          if at_floor and not fit.success else
                          'local linear estimate; does not include parameter-bound truncation or model uncertainty'))
    if covariance_audit is not None:
        from .derivatives import audit_covariance_steps
        try:
            audit=audit_covariance_steps(residual,fit.x,**covariance_audit)
            audit['status']='completed'
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as error:
            audit={'status':'failed','error':str(error),
                   'scope':'Fit retained; covariance audit did not complete. Optimizer covariance remains unverified by this audit.'}
        diagnostics['covariance_audit']=audit
    return result
