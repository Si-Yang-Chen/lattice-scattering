# Changelog

## Unreleased — 0.2.0.dev0

- Add optional Matplotlib figure builders for finite-volume spectra,
  coupled-channel amplitude strengths, root diagnostics, local pole maps and
  covariance diagnostics. A reproducible `plot-demo` command writes synthetic
  results, PNGs and SVGs without claiming paper-data reproduction.

## 0.1.0 — initial research release

- Public two-body JLS finite-volume matrices, kinematics, symmetry projections,
  amplitude models, correlated fits, replicas, and CLI examples.
- Default spectral root search with sampled-extremum refinement and per-root
  matrix residual checks. The historical determinant search remains selectable.
- Zeta boost domain extended to `gamma <= 1.52`, with a corrected componentwise
  convergence stop criterion.
- Optional left-hand-cut diagnostics for a declared equal-mass, single-channel
  t-channel exchange in root, fit, and pole workflows.
- Explicit numerical-domain checks and documentation of finite sampling,
  physical conventions, and unsupported analyses.

The release contains product tests and synthetic examples. Paper-by-paper
screening inputs, mock runs, and campaign evidence are maintained outside the
public distribution.
