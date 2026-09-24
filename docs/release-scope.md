# Release scope and scientific limits

Version 0.1.0 is a public **research software** release for selected two-body
finite-volume calculations. It can read declared spectra, evaluate supported
kinematics and JLS quantization matrices, search for levels, construct supported
amplitude models, and perform correlated fits. The CLI accepts constant blocks
or registered spinless S-wave amplitudes; more general JLS blocks use the Python
API. Inputs must follow the units and conventions in [conventions](conventions.md).

## Supported numerical domain

- The wide Zeta kernel accepts harmonic degree 0–12, real `q²` from -48 to 4,
  boost `gamma` from 1 to 1.52, `|d|² <= 36`, shift `alpha` from 0 to 1, and
  split parameter from 0.5 to 2. Other joint restrictions are checked by the
  API. These bounds are an implementation domain, not a uniform error theorem.
- The default root search follows ordered Hermitian eigenvalues, refines sampled
  sign changes and extrema, and checks every reported root's matrix residual.
  It can miss roots narrower than the scan resolution or inside an excluded
  free-pole band. Users should compare scan densities and verify returned roots.
- An optional left-hand-cut guard covers one declared equal-mass elastic
  channel with a single t-channel exchange. The user must supply the exchange
  mechanism and relevant masses. Missing declarations make no applicability
  claim; unequal-mass, coupled, unstable-particle, and other exchange mechanisms
  are outside this guard's implemented formula. See [the domain guide](left-hand-cut-domain.md).

The package does not implement a three-body quantization engine, complete
one-particle-exchange/LS treatment, or the modified finite-volume geometry for
generic finite-lattice-spacing dispersion. No general root-completeness or
paper-specific amplitude-convention claim is made. The capability boundary
applies to this version, not to all possible two-body analyses.

## Evidence and interpretation

The public test suite uses analytic references and synthetic spectra. A separate
screening exercise classified 100 unique research works: 63 required a missing
model or algorithm, 37 had no conflict within the stated mock-test scope, and
zero had a verified conflict. **No same-input comparison using published energy
levels and covariance matrices was performed.** Thus the zero-conflict count
does not establish agreement with any paper. The screening and per-work evidence
are kept in a separate private development archive, outside the public source
and release artifacts.

The intended citation is to this software version and its source revision.
Any scientific result should additionally state the chosen physical model,
normalization, truncation, units, scan settings, and data provenance.
