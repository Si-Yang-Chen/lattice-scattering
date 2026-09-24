# Left-hand-cut domain diagnostics

`EqualMassExchangeDomain` checks **one declared single-particle t-channel
exchange in one equal-mass elastic channel**. It does not implement OPE/LS
dynamics or a modified finite-volume quantization condition. In particular,
the first implementation does not derive unequal-mass u-channel DD* cuts,
unstable-particle or coupled-channel cuts. These require additional formulas.

The exchange angular denominator is `mu^2 + 2*k^2*(1-z)`. Its endpoint at
`z=-1` gives `k_L^2=-mu^2/4` and `s_L=4*m^2-mu^2`. On the chosen principal
left-hand sheet the cut extends along real `s <= s_L`. The supported mass
range is `0 < mu < 2*m`. See [2311.18793v2, Eq. (2) and Sec. 2.2](https://arxiv.org/html/2311.18793v2#S2.SS2).

```python
from dataclasses import asdict
from lattice_scattering.amplitudes import EqualMassExchangeDomain

domain = EqualMassExchangeDomain(
    mass=0.3, exchange_mass=0.1, exchange_allowed=True, coupling=1.0,
    mechanism="equal_mass_t_exchange", units="temporal_lattice",
)
report = domain.check((0.355, 0.365), path_s=[0.36, 0.362-0.001j], sheet="II")
print(asdict(report))
report.require_qc()       # reject excluded or unknown QC applicability
report.require_ere()      # reject outside/unknown exchange-limited ERE radius
report.require_path()     # reject crossing, unknown, or missing path
```

All windows and path vertices above are **CM squared energy**, not lab energy
or CM energy. Masses must share the declared units; squared quantities use
their square. The report labels those base units. GeV inputs are supported for
standalone diagnostics: `domain.to_temporal_lattice(at_gev_inverse)` converts
the masses using a supplied `a_t` in GeV^-1. Convert window/path values by
`a_t**2` separately. The numerical coupling is used only to decide zero versus
nonzero; this utility does not convert coupling conventions or compute amplitudes.

The report separates:

- `qc_status`: `excluded` at or below the branch point, `not_excluded` above
  it, or `unknown` when this formula cannot be applied. It includes the
  affected real interval and minimum distance in s to the branch point.
- `ere_status`: compares the maximum `|s-4*m^2|/mu^2` over the window and path
  with one. `inside_exchange_radius` is only a necessary exchange-based test;
  it does not bound truncation error or exclude closer amplitude singularities.
- `path_status`: tests every straight segment against the cut, not only its
  vertices. `I` and `II` label right-hand two-body sheets; unsupported sheet
  names yield `unknown`. Continuation across the left-hand cut is not supported.

Missing selection rules, coupling, mass or mechanism give `unknown`. Explicit
zero coupling or a forbidden exchange remove **that declared exchange** only.
The caller must inventory other relevant exchanges/cuts. Passing this diagnostic
does not certify the whole model, ERE accuracy or finite-volume accuracy:
neglected exponential corrections can grow even above but near `s_L`.

## Guards in numerical workflows

Pass `lhc_domain=domain` to `quantization_roots` or `coupled_s_roots` to check
the complete lab-energy window after converting it to s. The guard requires
one matching equal-mass channel and temporal-lattice units. Unknown or excluded
domains raise `LeftHandCutDomainError` before any matrix call. Existing calls
without the option continue to run and make no LHC applicability claim.

The `roots` CLI (constant blocks and registered amplitudes) accepts a
`lhc_domain` object with the constructor fields above. The `fit` CLI accepts
the same object in **each condition** that needs the check. The option does
not automatically impose an ERE restriction on arbitrary K-matrix models;
call `require_ere()` explicitly when using an ERE expansion.

`solve_poles(..., lhc_domain=domain)` checks the entire complex search rectangle
for an intersection with the declared cut before optimization. A rectangle
crossing the cut is rejected even if all seeds lie off it. This checks the
search region, not a caller's external continuation path from a reference
point: check that path separately with `domain.check(..., path_s=..., sheet=...)`.
The guard does not change the amplitude's sheet implementation.

Tests use the independent angular exchange denominator, branch-point/boundary
cases, unit conversion, straight-segment crossings, unknown mechanisms,
decoupling and pre-evaluation rejection in root, fit and pole workflows. No
paper classification or numerical reproduction is automatically promoted.
