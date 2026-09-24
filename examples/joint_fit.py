"""Synthetic correlated fit with a complete-group root matching check."""
import numpy as np
from lattice_scattering.fitting import JointFitProblem, Observation, fit_joint, match_levels

positions = (0.2, 0.4, 0.6, 0.8, 1.0)
observations = tuple(
    Observation(
        level_id=f"L{i}", frame=None, frame_id="toy", irrep="A1", row=0,
        energy=0.4 + 0.25 * x, group="shared-toy-condition",
    )
    for i, x in enumerate(positions)
)
energies = np.array([item.energy for item in observations])


def predict_roots(theta, _observation):
    # All five toy observations share this complete candidate root set.
    # Replace it with quantization_roots for a physical fit.
    return [0.4 + theta[0] * x for x in positions]


problem = JointFitProblem(observations, predict_roots, tolerance=0.1)
result = fit_joint(
    problem, [0.2], covariance=np.eye(len(observations)) * 1e-4,
    energies=energies, bounds=([0.0], [1.0]),
)
print(f"status={result['status']} parameter={result['parameters'][0]:.6f}")

predicted = predict_roots(result["parameters"], observations[0])
check = match_levels(energies, predicted, tolerance=0.01)
print(f"matched={check.n_matched} missing={len(check.missing_levels)}")
