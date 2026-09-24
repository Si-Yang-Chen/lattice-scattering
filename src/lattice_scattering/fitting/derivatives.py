"""Budgeted fixed-point centered-difference covariance diagnostics."""
import numpy as np
from .covariance import covariance_from_jacobian


def audit_covariance_steps(residual, parameters, *, parameter_scales, relative_steps,
                           max_evaluations):
    """Audit a whitened residual; steps = explicit scales * relative_steps.

    No refitting, bound clipping, automatic convergence claim or covariance
    replacement. Scales must be positive even for zero-valued parameters.
    Requires 2 * number_of_parameters * number_of_steps residual calls.
    The callback must be deterministic and defined at all displaced points.
    """
    def vector(value, name):
        if np.iscomplexobj(value):
            raise ValueError(f'{name} must be real')
        value = np.asarray(value, dtype=float)
        if value.ndim != 1 or not len(value) or not np.all(np.isfinite(value)):
            raise ValueError(f'{name} must be a finite nonempty vector')
        return value.copy()
    theta = vector(parameters, 'parameters')
    scales = vector(parameter_scales, 'parameter_scales')
    steps = vector(relative_steps, 'relative_steps')
    if scales.shape != theta.shape or np.any(scales <= 0):
        raise ValueError('positive matching parameter scales required')
    if len(steps) < 2 or np.any(steps <= 0) or np.any(np.diff(steps) >= 0):
        raise ValueError('at least two strictly decreasing positive steps required')
    required = 2 * len(theta) * len(steps)
    if isinstance(max_evaluations, bool) or not isinstance(max_evaluations, (int, np.integer)) or max_evaluations < required:
        raise ValueError(f'evaluation budget must cover {required} calls')
    # Validate all displacements before spending the callback budget.
    displaced = []
    for relative in steps:
        h = relative * scales
        plus = theta[None, :] + np.diag(h)
        minus = theta[None, :] - np.diag(h)
        if not np.all(np.isfinite(plus)) or not np.all(np.isfinite(minus)) or np.any(np.diag(plus) == theta) or np.any(np.diag(minus) == theta):
            raise ValueError('unrepresentable finite-difference displacement')
        displaced.append((h, plus, minus))
    rows = []
    previous = None
    shape = None
    for relative, (h, plus, minus) in zip(steps, displaced):
        columns = []
        for index in range(len(theta)):
            high = vector(residual(plus[index].copy()), 'residual')
            low = vector(residual(minus[index].copy()), 'residual')
            if shape is None:
                shape = high.shape
            if high.shape != shape or low.shape != shape:
                raise ValueError('residual dimension changed')
            columns.append((high - low) / (2 * h[index]))
        jacobian = np.column_stack(columns)
        covariance, condition = covariance_from_jacobian(jacobian)
        errors = np.sqrt(np.diag(covariance))
        rows.append({'relative_step': float(relative), 'absolute_steps': h.tolist(),
                     'condition': condition, 'standard_errors': errors.tolist(),
                     'covariance': covariance.tolist(),
                     'singular_values': np.linalg.svd(jacobian, compute_uv=False).tolist(),
                     'max_relative_error_change_from_previous': None if previous is None else float(np.max(abs(errors / previous - 1)))})
        previous = errors
    return {'parameters': theta.tolist(), 'parameter_scales': scales.tolist(),
            'evaluations': required, 'steps': rows,
            'scope': 'Fixed-point unscaled local covariance; no refit, bound truncation, replica or model uncertainty.'}
