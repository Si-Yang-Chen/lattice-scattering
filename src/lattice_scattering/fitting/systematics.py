"""Systematic-error propagation and model comparison (P7).

The layer keeps *statistical* and *systematic* uncertainties separate, as the
project's evidence discipline requires:

* :func:`perturb_and_refit` propagates shifts in nuisance inputs (hadron
  masses, anisotropy) by covariant Gaussian sampling, refitting each draw and
  returning the **spread of the fitted parameters**;
* :func:`compare_models` scores several model parameterisations on the same
  observations and reports an Akaike-style spread rather than a single winner.

Nothing here merges the two error sources into one Gaussian, and failed draws
are reported, not dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .joint import FitConvergenceError, JointFitProblem, fit_joint

__all__ = [
    "ModelComparison",
    "SystematicResult",
    "compare_models",
    "perturb_and_refit",
]


@dataclass(frozen=True)
class SystematicResult:
    """Outcome of a covariant nuisance-parameter scan.

    ``parameters`` is the central fit; ``systematic_covariance`` is the sample
    covariance of the refitted parameters over the draws.  ``failed`` records
    every draw that could not be refit, so the systematic band is never computed
    from a silently truncated sample.
    """

    parameters: np.ndarray
    systematic_covariance: np.ndarray
    n_draws: int
    n_failed: int
    failed: tuple = ()
    scope: str = (
        "systematic covariance from nuisance draws; statistical covariance is reported separately"
    )


def perturb_and_refit(
    problem_factory,
    central,
    *,
    covariance,
    energies,
    nuisance_covariance,
    n_draws: int = 32,
    seed: int = 0,
    bounds=(-np.inf, np.inf),
    **solver_options,
) -> SystematicResult:
    """Refit ``n_draws`` times with nuisance parameters drawn from their covariance.

    ``problem_factory(nuisance_vector)`` must return a
    :class:`~lattice_scattering.fitting.joint.JointFitProblem` built with the
    drawn nuisance values (masses, anisotropy, ...).  ``nuisance_covariance`` is
    the covariance of those nuisance parameters and ``central`` the starting
    point of every refit.  The returned systematic covariance is
    ``cov(draws) - nuisance_covariance_contribution`` is **not** attempted; it is
    the raw sample covariance of the refitted parameters, which is what a
    systematic band needs.
    """
    nuisance_covariance = np.asarray(nuisance_covariance, dtype=float)
    if nuisance_covariance.ndim != 2 or nuisance_covariance.shape[0] != nuisance_covariance.shape[1]:
        raise ValueError("nuisance_covariance must be a square matrix")
    if np.iscomplexobj(nuisance_covariance) or not np.all(np.isfinite(nuisance_covariance)):
        raise ValueError("finite real nuisance_covariance required")
    if isinstance(n_draws, bool) or not isinstance(n_draws, int) or n_draws < 2:
        raise ValueError("n_draws must be an integer >= 2")
    central = np.asarray(central, dtype=float)
    generator = np.random.default_rng(seed)
    draws = generator.multivariate_normal(
        np.zeros(nuisance_covariance.shape[0]), nuisance_covariance, size=n_draws
    )
    parameters: list[np.ndarray] = []
    failed: list[str] = []
    for index, shift in enumerate(draws):
        try:
            problem = problem_factory(shift)
            result = fit_joint(
                problem, central, covariance=covariance, energies=energies, bounds=bounds, **solver_options
            )
        except (FitConvergenceError, ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            failed.append(f"draw {index}: {type(error).__name__}: {error}")
            continue
        parameters.append(np.asarray(result["parameters"], dtype=float))
    if len(parameters) < 2:
        raise RuntimeError(
            f"only {len(parameters)} of {n_draws} nuisance draws converged; "
            "a systematic covariance cannot be estimated"
        )
    values = np.asarray(parameters)
    # ``np.cov`` returns a 0-d array for a single parameter; keep the shape a
    # caller can rely on.
    covariance_matrix = np.atleast_2d(np.cov(values, rowvar=False, ddof=1))
    return SystematicResult(
        parameters=np.asarray(central, dtype=float),
        systematic_covariance=covariance_matrix,
        n_draws=n_draws,
        n_failed=len(failed),
        failed=tuple(failed),
    )


@dataclass(frozen=True)
class ModelComparison:
    """Akaike-style comparison of several model parameterisations."""

    labels: tuple[str, ...]
    chi2: tuple[float, ...]
    n_parameters: tuple[int, ...]
    n_observations: int
    aic: tuple[float, ...]
    weights: tuple[float, ...]
    failed: tuple[str, ...] = field(default=())

    def as_dict(self) -> dict:
        return {
            "labels": list(self.labels),
            "chi2": list(self.chi2),
            "n_parameters": list(self.n_parameters),
            "n_observations": self.n_observations,
            "aic": list(self.aic),
            "weights": list(self.weights),
            "failed": list(self.failed),
        }


def compare_models(
    candidates,
    *,
    covariance,
    energies,
    bounds=(-np.inf, np.inf),
    **solver_options,
) -> ModelComparison:
    """Fit each candidate model and report an Akaike weight spread.

    ``candidates`` is a sequence of ``(label, problem, initial)`` triples, or of
    ``(label, problem, initial, bounds)`` quadruples when the candidates have
    different parameter counts (the shared ``bounds`` argument is then only a
    default for the triples).  Candidates whose fit fails are recorded in
    ``failed`` and excluded from the weights; the weights are therefore **not**
    normalised over the failed models, which is the honest reading of an
    incomplete model set.
    """
    labels: list[str] = []
    chi2: list[float] = []
    sizes: list[int] = []
    failed: list[str] = []
    n_observations = len(np.asarray(energies))
    for candidate in candidates:
        if len(candidate) == 4:
            label, problem, initial, candidate_bounds = candidate
        elif len(candidate) == 3:
            label, problem, initial = candidate
            candidate_bounds = bounds
        else:
            raise ValueError(
                "each candidate must be (label, problem, initial) or (label, problem, initial, bounds)"
            )
        try:
            result = fit_joint(
                problem, np.asarray(initial, dtype=float), covariance=covariance,
                energies=energies, bounds=candidate_bounds, **solver_options,
            )
        except (FitConvergenceError, ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            failed.append(f"{label}: {type(error).__name__}: {error}")
            continue
        labels.append(str(label))
        chi2.append(float(result["chi2"]))
        sizes.append(int(len(result["parameters"])))
    if not labels:
        raise RuntimeError("no candidate model converged; nothing to compare")
    chi2_array = np.asarray(chi2)
    aic = chi2_array + 2.0 * np.asarray(sizes)
    shifted = aic - aic.min()
    weights = np.exp(-0.5 * shifted)
    weights = weights / weights.sum()
    return ModelComparison(
        labels=tuple(labels),
        chi2=tuple(chi2),
        n_parameters=tuple(sizes),
        n_observations=n_observations,
        aic=tuple(float(a) for a in aic),
        weights=tuple(float(w) for w in weights),
        failed=tuple(failed),
    )
