# Inverse-K and reduced ERE models

This page covers three registered single-channel parameterizations:
`RationalInverseK`, `ReducedERE`, and `UnitaryChPTLO`. They evaluate inverse
forms directly, so a zero of an inverse amplitude can be represented without
first constructing the singular K matrix. The documented formulas and the
synthetic equation checks verify these implementations and their adapters;
they do not fit source-paper data or reproduce a paper spectrum.

## Shared energy units

Use one energy unit for all masses, momenta, and energies. For lattice inputs,
use temporal-lattice units: masses, momenta, and energies are `a_t E`, while
`s` and `scale_squared` are in `(a_t E)^2`. `s_ref` is also in `(a_t E)^2`.
The formulas below use natural units, `hbar = c = 1`; the numeric values in
the code are dimensionless in the chosen common unit.

## Rational inverse K

`RationalInverseK` implements

\[
K^{-1}(s)=\frac{N(x)}{D(x)},\qquad
x=\frac{s-s_{\rm ref}}{\texttt{scale\_squared}},
\]

where `numerator_coefficients` and `denominator_coefficients` list polynomial
coefficients in ascending powers of `x`. This generalizes the rational
inverse-K form in [arXiv:2102.04973v2, Eq. (8)](https://arxiv.org/html/2102.04973v2),
which writes the denominator with constant term one and uses powers of `s` or
a threshold-shifted dimensionless variable. Here both polynomials may have an
arbitrary nonzero normalization. Since `x` is dimensionless, the polynomial
coefficients are dimensionless in the chosen K convention.

The model returns `N/D` directly. A zero of `N` is a regular zero of `K^-1`
(a K pole); a zero of `D` is a singularity of the inverse and is returned by
`s_breakpoints()`. `scale_squared` must be positive, and `D` must not be the
zero polynomial.

## Reduced effective-range expansion

For partial wave `ell`, `ReducedERE` represents

\[
R_\ell(s)=k^{2\ell+1}\cot\delta_\ell
       =\sum_{n=0}^{N}c_n\,[k^2(s)]^n,
\qquad
k^2(s)=\frac{[s-(m_1+m_2)^2][s-(m_1-m_2)^2]}{4s}.
\]

This is the polynomial form of the ERE, whose leading terms are given in
[arXiv:2102.04973v2, Eq. (9)](https://arxiv.org/html/2102.04973v2). `coefficients`
are in ascending powers of `k^2`; coefficient `c_n` has energy dimension
`2*ell + 1 - 2*n`. `mass1` and `mass2` are positive energies, and `ell` is a
nonnegative integer.

`reduced(s)` and `jls_scale(s)` return the finite `R_ell` polynomial, including
at `k^2=0`. The hatted inverse view is

\[
\hat K^{-1}(s)=\rho\cot\delta_\ell
 =\frac{2}{\sqrt{s}}\,\frac{R_\ell(s)}{[k^2(s)]^\ell},
\qquad \rho=\frac{2k}{\sqrt{s}}.
\]

Thus `inverse(s)` can be singular at threshold for `ell > 0`, even when the
reduced polynomial remains finite. Use the reduced view for the JLS
`weighting="scale"` callback. The inverse view reports a threshold
singularity unless its numerator has a removable factor of `(k^2)^ell`.

## Unitarized D-pion LO ChPT

`UnitaryChPTLO` follows [arXiv:2102.04973v2, Eqs. (11)–(12)](https://arxiv.org/html/2102.04973v2)
in the arXiv HTML version. Define

\[
P(s)=3s^2-2s(m_D^2+m_\pi^2)-(m_D^2-m_\pi^2)^2,
\qquad
\mathcal V_{J=0}(s)=-\frac{P(s)}{4sF^2}.
\]

The inverse K used by the model is

\[
K^{-1}(s)=\left(-\frac{1}{16\pi}\mathcal V_{J=0}(s)\right)^{-1}
 +\frac{\alpha(\mu)}{\pi}
 +\frac{2}{\pi}\left[
      \frac{m_D}{m_\pi+m_D}\log\frac{m_D}{m_\pi}
      +\log\frac{m_\pi}{\mu}
   \right].
\]

The source treats `F` and `alpha(mu)` as fit parameters and fixes the
renormalization scale to `a_t*mu = 0.1645` (about 1000 MeV in physical units
for that ensemble). In this API, `decay_constant` is the source's `F`; it is a
positive energy parameter and is not fixed to the physical pion decay constant.
`alpha` is the dimensionless `alpha(mu)` at the scale supplied as `mu`. `mu`
is a positive energy in the same units as `pion_mass` and `heavy_mass`. The
code does not run `alpha` when `mu` changes, so keep the parameter paired with
the scale at which it was defined. In particular, for inputs matching that
paper's lattice convention, pass `mu=0.1645` along with the fitted
`alpha(0.1645)` and `F`.

`inverse(s)` returns the rational `K^-1` expression above; it refuses `s=0`,
where the source expression is undefined. `t_inverse(s)` adds the real-axis,
threshold-subtracted Chew–Mandelstam function `I(s)` from the package, with
`I((m_D+m_pi)^2)=0`. The masses, `F`, and `mu` all use the same energy unit;
`alpha` is dimensionless.

## Registered names

The registry names are `"rational-inverse-k"`, `"reduced-ere"`, and
`"unitarized-chpt-lo"`. Their public classes are exported by
`lattice_scattering.amplitudes`.
