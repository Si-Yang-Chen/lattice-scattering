# Chung and P33 resonance models

This guide documents the registered elastic amplitude models and their JLS
normalization adapter in the current `lattice_scattering` package. The formulas
and examples describe software conventions; they do not validate a paper's
parameter extraction, irrep mapping, or wave truncation, and they make no paper
reproduction claim. These APIs have no dependency on the former V1 package or
its import paths.

The Chung convention follows [arXiv:2006.14035v2, Eqs. (9)–(19)](https://arxiv.org/html/2006.14035v2).
The P33 Breit–Wigner convention follows [arXiv:2101.00689v2, Eqs. (32)–(34)](https://arxiv.org/html/2101.00689v2): Eqs. (32)–(33) give P33, while Eq. (34) is the accompanying S31 ERE. The tests check these formulas and the JLS adapter's normalization; they do not fit paper data or reproduce either paper.

## Models and registered names

Import `ChungBW` and `P33BW` from `lattice_scattering.amplitudes`. Their
registry keys are `"chung-bw"` and `"p33-bw"`; use those exact strings with
`build(...)`. `ChungK` and `P33K` are compatibility aliases for the class names.
For both models, `.k(s)` returns the hatted amplitude \(\hat K\) as a 1-by-1
matrix and `.inverse(s)` returns \(\hat K^{-1}\). Here `s` is center-of-mass
energy squared in temporal-lattice units. The method name `.k` follows the
amplitude protocol; it means \(\hat K\), not the center-of-mass momentum.

With \(\rho=2k/\sqrt{s}\), the hatted convention is
\(\hat K=\tan\delta/\rho\), so \(\hat K^{-1}=\rho\cot\delta\).

Use one consistent energy unit for masses, momenta, and energies. In this
package's lattice convention, enter them in temporal-lattice units (\(a_t E\))
and enter \(s\) in \((a_t E)^2\). The Chung `coupling` corresponds to the
paper's \(g_\ell^0\) and has energy units (enter \(a_t g_\ell^0\)); its
`range_parameter` has inverse-energy units (enter \(R_1/a_t\)). The P33
`coupling` is dimensionless in natural units. All these values are numeric in
the chosen common unit.

### Chung S and P waves

`ChungBW` accepts `pole_mass`, `coupling`, `mass1`, `mass2`, and `ell`. Use
`ell=0` or `ell=1`; `range_parameter` is required and positive for `ell=1`.
The finite real coupling must be nonzero, and both masses and the pole mass must
be positive. The model uses

\[
\hat K_\ell(s)=\frac{g^2 B_\ell^2(s)}{m_{\rm BW}^2-s},\qquad
B_0^2(s)=1,
\]

and for the P wave

\[
F_1^2(k)=\frac{2(kR)^2}{1+(kR)^2},\qquad
B_1^2(s)=\frac{F_1^2(k(s))}{F_1^2(k_\alpha)},\qquad
k_\alpha^2=k^2(m_{\rm BW}^2).
\]

Here `coupling` is \(g_\ell^0\) and `pole_mass` is \(m_\ell\) in the source
notation. The inverse is evaluated analytically as
\((m_{\rm BW}^2-s)/(g^2B_\ell^2(s))\), including its exact zero at the bare
pole. A Chung P-wave pole cannot coincide with a two-body branch point because
then \(F_1(k_\alpha)=0\); the constructor also rejects a pole at which the
Blatt-Weisskopf denominator vanishes. `ConformalMapK` is a separate registered
model: its `alpha` must be a
finite positive real scalar, with no upper bound, so `alpha=1.3` is valid. This
model parameter is distinct from the kinematic shift `TwoBodyPoint.alpha`.

### P33 Breit-Wigner

`P33BW` accepts `pole_mass`, `coupling`, `mass1`, and `mass2`, and represents a
P wave (`ell=1`). Its convention is

\[
\hat K_{P33}(s)=\frac{g^2 k^2}{12\pi(m_{\rm BW}^2-s)},\qquad
\hat K_{P33}^{-1}(s)=\frac{12\pi(m_{\rm BW}^2-s)}{g^2k^2}.
\]

The exact \(12\pi\) factor is part of the convention. A noncoincident bare pole
may lie below threshold; `P33BW` does not impose an above-threshold pole. The
constructor rejects a bare pole that coincides exactly with a two-body
threshold or pseudothreshold, where the pole-zero and inverse-singularity
contracts would coincide.

For example, the public constructors and registry use are:

```python
from lattice_scattering.amplitudes import ChungBW, P33BW, build

chung_p = ChungBW(
    pole_mass=0.9, coupling=0.7, mass1=0.25, mass2=0.35,
    ell=1, range_parameter=1.2,
)
p33 = P33BW(pole_mass=0.78, coupling=0.63, mass1=0.2, mass2=0.3)

same_chung = build(
    "chung-bw",
    {"pole_mass": 0.9, "coupling": 0.7, "mass1": 0.25,
     "mass2": 0.35, "ell": 1, "range_parameter": 1.2},
)
same_p33 = build(
    "p33-bw",
    {"pole_mass": 0.78, "coupling": 0.63, "mass1": 0.2, "mass2": 0.3},
)
```

## Adapt a model to a JLS block

`HattedKJLSAdapter(model, ell, twice_S, twice_J, mass1=None, mass2=None)`
binds one amplitude to one \((\ell,2S,2J)\) incidence. It checks the triangle
rule, agreement with `model.ell` when present, and agreement of supplied masses
with model masses. If the model does not expose both masses, supply `mass1` and
`mass2`. The wrapped model must provide `.k(s)` and `.inverse(s)`.

The adapter's `.reduced_inverse(s)` (also callable via `adapter(s)` or
`adapter.block(s)`) returns the scale-weighted one-by-one block

\[
R_{\rm scale}^{J}(s)=\frac{\sqrt{s}}{2}\,k^{2\ell}\hat K^{-1}(s)
                   =k^{2\ell+1}\cot\delta_\ell(s).
\]

The adapter applies the `rho=2k/sqrt(s)` convention exactly once. Its `.blocks(s)`
returns a mapping containing only its declared `twice_J`; it is a one-incidence
adapter, not a multi-J model. In the common all-J layout, pass one callable
adapter for every required J key. `row_matrix` and `quantization_roots` default
to `weighting="scale"`, so this is the direct adapter path.

This tested spinless moving-frame pattern uses a separate S and P adapter in
each allowed J block:

```python
from lattice_scattering.amplitudes import ChungBW, HattedKJLSAdapter
from lattice_scattering.finite_volume import row_matrix
from lattice_scattering.kinematics import LatticeFrame
from lattice_scattering.symmetry import double_cover, little_group

frame = LatticeFrame(24, 1.0, (0, 0, 1))
masses = ((0.2, 0.3),)
chung_s = ChungBW(pole_mass=0.75, coupling=0.5,
                 mass1=0.2, mass2=0.3, ell=0)
chung_p = ChungBW(pole_mass=0.82, coupling=0.65,
                 mass1=0.2, mass2=0.3, ell=1, range_parameter=1.1)
j_blocks = {
    0: HattedKJLSAdapter(chung_s, ell=0, twice_S=0, twice_J=0),
    2: HattedKJLSAdapter(chung_p, ell=1, twice_S=0, twice_J=2),
}
group = double_cover(little_group(frame.d))
row = row_matrix(
    0.79, frame, masses, ((0, 0), (1, 0)), j_blocks,
    group=group, irrep="A1", weighting="scale",
)
```

For a selective-J layout, match adapter incidences and keys exactly. For example,
N-pion sectors `((0, 1), (1, 1))` with
`selected_j_sectors={1: ((0, 0, 1),), 3: ((0, 1, 1),)}` use an S-wave adapter
bound to `(ell=0, twice_S=1, twice_J=1)` and a P33 adapter bound to
`(ell=1, twice_S=1, twice_J=3)`. The P31 incidence is absent from the active
space; do not supply a zero or large-inverse substitute. This describes an
explicit computational truncation and does not establish that a paper's
truncation is physically justified.

## Weighting, poles, and threshold behavior

Let \(\lambda=L/(2\pi)\). `weighting="scale"` multiplies a JLS matrix entry
between orbital waves \(\ell_a,\ell_b\) by
\(\lambda^{\ell_a+\ell_b+1}\). For a diagonal one-wave adapter this is
\(\lambda^{2\ell+1}R_{\rm scale}\).

`weighting="threshold"` instead applies \((2k)^{-\ell}\) on each side of the
supplied block. To convert one adapter's scale block, call
`adapter.threshold_block(s, scale=frame.length_at / (2*pi))`; it returns
\[
R_{\rm threshold}(s)=(2k)^{2\ell}\lambda^{2\ell+1}R_{\rm scale}(s),
\]
which the JLS threshold factors transform back to the same scale-weighted
matrix. Do not pass a scale block unchanged with `weighting="threshold"`; the
flag does not convert the callback. The threshold convention uses a real
above-threshold branch and is singular at `k=0` for `ell>0`; the helper raises
at exact `k^2=0` for those waves, including at the physical threshold and
pseudothreshold.

At an exact bare pole, `.k(s)` refuses the raw amplitude pole; `.inverse(s)`
returns its analytic zero. The inverse of either P-wave model is singular at
`k^2=0` (threshold or pseudothreshold) and reports that point. The adapter's
scale form uses an analytic `k^(2*ell) * inverse` cancellation for Chung and
P33, so its reduced inverse has a finite threshold limit. That finite scale
limit does not remove the threshold singularity of the `weighting="threshold"`
parameterization.

Use `adapter.s_breakpoints()` (or the wrapped model's `s_breakpoints()`) when
building `breakpoints_at2` for a root scan. The P-wave branch points belong in
the scan split; the regular inverse zero at a bare K pole does not. Scans still
need density-stability and residual checks, and are not proofs of complete
level coverage. The amplitude formulas, adapter checks, and synthetic
normalization oracles establish tested software behavior only; they do not
validate paper data or reproduce a paper spectrum.
