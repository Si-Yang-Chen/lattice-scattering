# Root-scan domain and limits

`quantization_roots` checks the full requested laboratory-energy window against
the moving-frame Zeta kinematic domain before it samples the determinant. For
each channel it checks both energy endpoints and, for unequal masses, the
possible interior stationary point
`s = |m1^2 - m2^2|` of `q^2(s)`. These points cover the extrema of `q^2`; the
boost `gamma` and shift `alpha` are monotone over the physical window. The
preflight also checks the pseudothreshold, momentum class, and split parameter.
An unsupported kinematic window raises `RootScanError` with the offending
laboratory energy, interval, kind, and original `ValueError` cause. Matrix or
determinant evaluation failures during sampling also stop the scan with their
energy and cause. A root-refinement failure stops the scan and reports its
bracket and cause.

Free poles and caller-provided model breakpoints split the scan into separate
intervals. The exact low and high scan-window endpoints are sampled unless an
endpoint is itself a free pole. Model-breakpoint edges use the nearest
representable energy on each side, avoiding a padding band at those splits.
The Zeta kernel can lose Hermiticity through cancellation immediately beside a
free pole, so a small one-sided numerical band around each free pole remains
excluded. A root inside that band is not guaranteed to be resolved. The
scanner no longer skips non-finite cells or falls back to a bisection that can
return no root after encountering a failed evaluation.

The default `root_method="eigenvalues"` follows the ordered eigenvalues of the
Hermitian row matrix. Ordered eigenvalues remain continuous at degeneracies;
this avoids ambiguous eigenvector assignment. Sign brackets are refined with
Brent's method. Sampled local minima/maxima trigger scalar extremum refinement,
which can locate tangent roots and split a cell containing two roots even when
the determinant has the same sign at both ends. Flat small nonzero eigenvalues
are not roots merely because they are small.

Each initial grid has `samples` points per split interval. `subdivisions=1`
inserts one midpoint in each cell; values `0..8` are accepted. Thus the initial
number of matrix evaluations is `(samples-1)*2**subdivisions+1`, before adaptive
refinement. Increasing this depth costs runtime and does not prove completeness.
The tests cover tangent, degenerate, opposite-slope, rotating-basis and dense
analytic roots, positive minima, model discontinuities and near-free-pole roots.

Every spectral candidate must satisfy
`min(abs(eigvalsh(M))) / max(1, max(abs(eigvalsh(M)))) <= residual_tol`
(default `1e-10`). `matrix_root_residual(M)` exposes this same backward residual.
The floor of 1 refers to the declared matrix normalization; it is not an energy
error bound, and a minimum below the tolerance can be indistinguishable from a
true tangent root numerically. The bracket solver automatically retries at
machine-level energy precision if `xtol` was insufficient for a steep root.
If the residual still fails, `RootScanError(kind="residual")` aborts the scan;
it never quietly drops the candidate. An undeclared pole/sign discontinuity
therefore cannot pass solely because a sign bracket converged in energy.

`root_method="determinant"` retains the historical sign-change-only algorithm
for comparisons. That path does not have the automatic matrix-residual gate;
check its results independently. Both methods return unique energies: roots
closer than `xtol` (or the floating-point merge floor) merge, so exact degeneracy
multiplicities are not returned as repeated levels.

This remains a finite-resolution search. Unseen oscillations, narrow extrema,
and roots inside a free-pole exclusion band can be missed. Compare sample
counts/subdivisions and inspect residuals; no general completeness guarantee is
made. Callback failures and strict `pole_policy="raise"` are still propagated.

The Zeta boost domain is now `1 <= gamma <= 1.52`, with the other joint bounds
unchanged (see [conventions](conventions.md)). Extension tests include independent
scalar Ewald and closed-channel Yukawa-image references, split/cutoff stability
through harmonic degree 12, and synthetic five-frame spectra including the
previously blocked `gamma=1.5137903686` condition. The per-component truncation
criterion now controls both adaptive refinement and final acceptance. These are
numerical tests at sampled conditions, not a rigorous uniform error bound over
the entire continuous domain or a reproduction of paper 2409.13197.

Optional `lhc_domain` performs a [left-hand-cut check](left-hand-cut-domain.md)
before scanning. It is available through Python and roots/fit CLI configurations.
Omitting it preserves existing calls but makes no left-hand-cut validity claim.
