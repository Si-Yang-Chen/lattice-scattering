# Usage

## 1. Run the included examples

Run commands from the `lattice-scattering` repository root, where `pyproject.toml` has `name = "lattice-scattering"`. Install with `python -m pip install -e .`, then:

```sh
python -m lattice_scattering roots examples/roots.json
python -m lattice_scattering roots examples/roots-amplitude.json
python -m lattice_scattering models
python -m lattice_scattering fit examples/fit.json
python -m lattice_scattering validate-spectrum examples/spectrum.json
python examples/constant_k.py
python examples/spinful_jls.py
python examples/joint_fit.py
```

The CLI root example returns approximately `0.783040725816` and `0.932308172488` in laboratory-frame temporal-lattice energy units. Its output schema is `lattice-scattering-roots-result/v1`; the spectrum checker returns `lattice-scattering-validation/v1`. Neither output is a spectrum input file for the other command. The constant-K Python example uses a **different** reduced-inverse convention and returns approximately `0.745754046618` and `0.897352314410`. Its printed determinants are zero to numerical precision. The spin-1/2 S+P matrix example prints shape `(3, 3)` and a Hermiticity error near zero. The synthetic fit example prints `status=converged parameter=0.250000` and `matched=5 missing=0`.

`roots-amplitude.json` uses the energy-dependent `chiral-ere-pcotdelta` model and returns approximately `0.779388655060` and `0.923592921214`. `models` reports registered model names, descriptions, and required constructor fields. `fit examples/fit.json` fits the coupling of a synthetic pole-plus-background amplitude and recovers it near `0.3`; this checks the CLI workflow, not physical data or model selection.

If the installed console command is on `PATH`, `lattice-scattering roots ...` and `lattice-scattering validate-spectrum ...` are equivalent. On Windows PowerShell, if installation cannot be used and NumPy/SciPy are already installed, run `$env:PYTHONPATH = (Resolve-Path .\src).Path` from this repository root before the `python -m lattice_scattering` commands.

## 2. CLI root-scan configuration

[examples/roots.json](../examples/roots.json) is a complete constant-block input. The `roots` command also accepts a registry-backed energy-dependent amplitude configuration as in [examples/roots-amplitude.json](../examples/roots-amplitude.json). The two input forms are mutually exclusive: use either `j_blocks` or `amplitude`.

| JSON field | Requirement and meaning |
| --- | --- |
| `schema` | Required, exactly `lattice-scattering-roots/v1`. |
| `frame` | Required object with positive integer `spatial_sites`, positive `anisotropy = a_s/a_t`, and three-integer momentum `d` (defaults to `[0,0,0]`). |
| `channel_masses_at` | Required list of `[m1, m2]` pairs, one pair per channel, in temporal-lattice units. |
| `sectors` | Ordered list of distinct `[ell, twice_S]` pairs, shared by every channel; required unless `channel_sectors` is supplied. |
| `j_blocks` | Required for the constant-block form: mapping from stringified `twice_J` to a constant real-symmetric matrix with the order described below. Omit when using `amplitude`. |
| `amplitude` | Alternative to `j_blocks`: object with a registered `model` name, its constructor `parameters`, optional `phase_space` (`simple` or `chew-mandelstam`), and optional per-channel `subtractions`. |
| `irrep` | Required label in the double cover of this frame's little group. |
| `energy_window_at` | Required increasing `[low, high]` laboratory-frame energy interval in temporal-lattice units. |
| `row` | Optional zero-based irrep row; default `0`. |
| `intrinsic_parity` | Optional product of the two hadron intrinsic parities, `+1` or `-1`; default `+1`. Coupled channels passed here must use the same value. |
| `weighting` | Optional `scale` (default) or `threshold`; the blocks must already use the matching reduced-inverse convention. |
| `samples` | Optional scan points per pole-free interval; default `60`, minimum `2`. Increase it to test root stability. |
| `breakpoints_at2` | Optional real model singularities expressed as center-of-mass squared energies. They split the scan; free poles are split automatically. |
| `root_method` | `eigenvalues` (default), or historical `determinant` for comparison. |
| `subdivisions` | Initial grid bisection depth `0..8`, default `1`; spectral path only. |
| `residual_tol` | Spectral matrix backward-residual tolerance, default `1e-10`. |
| `xtol` | Positive root energy/merge tolerance, default `1e-13`; steep spectral roots may refine more tightly to pass the residual gate. |
| `lhc_domain` | Optional [equal-mass exchange declaration](left-hand-cut-domain.md); unknown/excluded requests fail before scanning. |

The CLI constructs `double_cover(little_group(frame.d))`. Discover its exact labels before choosing `irrep`:

```python
from lattice_scattering.symmetry import double_cover, little_group
group = double_cover(little_group((0, 0, 0)))
print([(rep.label, rep.dimension) for rep in group.irreps])
```

Printed literature irrep names may differ from these labels. Check the representation, dimension, row, and parity before mapping them.

### Explicit channel and J selection

The Python JLS APIs and `roots` CLI additionally accept:

| Option | Meaning |
| --- | --- |
| `channel_sectors` | One ordered `(ell, twice_S)` list per channel. In JSON, use nested arrays. Use `sectors=None` with this option (the CLI omits `sectors`); common and per-channel sector declarations are mutually exclusive. |
| `channel_intrinsic_parities` | One intrinsic parity product `+1` or `-1` per channel, replacing the common scalar parity for the explicit layout. |
| `selected_j_sectors` | Mapping from `twice_J` to the active `(channel_index, ell, twice_S)` incidences for that J. JSON keys are strings and values are arrays of triples. |

Without these options the existing common-layout, all-J behavior is retained. With explicit J selection, supply amplitude blocks for exactly the selected J keys. Their rows/columns follow channel order and each channel's sector order, restricted to active incidences; mapping-entry order does not define the matrix order. Duplicate/unknown incidences and invalid triangle couplings are rejected. For example, `{"1": [[0, 0, 1]], "3": [[0, 1, 1]]}` retains the S wave at twice-J=1 and P wave at twice-J=3 for one spin-1/2 S+P channel, excluding the P-wave incidence at twice-J=1. This describes an explicit truncation, not a derivation that it is appropriate for a particular paper.

The selected determinant is evaluated on the active irrep-row subspace, not by inserting zero or very large inverse amplitudes for omitted blocks. A selected space with no requested irrep row is an error. Cross-channel amplitude mixing must conserve total parity `intrinsic_parity[channel] * (-1)**ell`, so channels of opposite intrinsic parity may mix opposite orbital parities. Physical validity of any selected truncation, amplitude normalization, and paper irrep mapping still needs independent verification.

### JLS block layout

For the default all-J layout, the allowed `twice_J` values for each `(ell, twice_S)` run from `abs(2*ell - twice_S)` through `2*ell + twice_S` in steps of two. Supply **every** key in the union of those values, including allowed J values that a source analysis might omit. For a given J, include only participating sectors; matrix indices run by channel first, then by the original sector order. For one channel with `sectors = [[0,1],[1,1]]`, `j_blocks["1"]` is `2×2` (S and P), while `j_blocks["3"]` is `1×1` (P). With two channels those orders become `4` and `2`. With the common intrinsic parity default, matrix elements between opposite orbital parity sectors must be zero. [examples/spinful_jls.py](../examples/spinful_jls.py) applies this layout to a moving-frame spin-1/2 S+P matrix with `twice_J=1,3`. Its constants are an algebra demonstration, not a physical amplitude fit.

`core.enumerate_sectors` returns `Sector(channel_index, ell, twice_S)` objects for data-model bookkeeping. `quantization_roots` and the CLI instead require the **common** `(ell, twice_S)` pair list above; do not pass the `Sector` objects directly. Use the explicit channel options above for channel-dependent sectors or intrinsic parities.

### Registry-backed roots

Run `python -m lattice_scattering models` to discover models registered by `lattice_scattering.amplitudes`. Each `amplitude.parameters` object is passed to that registered constructor and checked there. Unknown model names and unknown config fields fail with a CLI error. `phase_space` defaults to `simple`; `chew-mandelstam` adds the package's real-axis CM term and accepts one `subtractions` value per channel. Known inverse-form singularities returned by `model.s_breakpoints()` are automatically added to the scan split points; explicit `breakpoints_at2` are merged with them.

The registry models expose channel-basis K matrices. This CLI route currently adapts that interface only for spinless S waves: common `sectors` must be omitted or exactly `[[0, 0]]`, and per-channel `channel_sectors` (if supplied) must contain `[[0, 0]]` for each channel. It constructs the reduced inverse for the selected JLS weighting:

| Weighting | S-wave block passed to JLS |
| --- | --- |
| `threshold` | `R^0(s) = (sqrt(s) L / 4π) K_eff^-1(s)` |
| `scale` | `R^0(s) = (sqrt(s) / 2) K_eff^-1(s)` |

and sends that energy-dependent block to the JLS root scanner. The threshold form is the established S-wave mapping used by `coupled_s_roots`; the scale form gives the same condition in the default scale normalization. Neither is a general higher-wave conversion. The model matrix channel dimension must match `channel_masses_at`; models whose parameters embed channel masses need consistent values in both locations. Do not use this route for a spinful/higher-wave model or select a convention without matching the model's normalization. Use the Python JLS API for amplitude-specific blocks outside this domain. The CLI does not infer paper-specific amplitude, unit, subtraction, or irrep conventions.

## 3. Energy-dependent amplitude and root scan

[examples/constant_k.py](../examples/constant_k.py) is an executable one-channel S-wave example. It constructs `ConstantK([[2.0]])`, computes `effective_inverse_k(..., "simple", s_at2)`, and maps that channel-basis inverse to the JLS reduced inverse via

`R^0(s) = sqrt(s) * frame.length_at / (4*pi) * effective_inverse_k(model, masses, "simple", s)`.

That mapping is checked for this S-wave convention. The example passes `{0: reduced_inverse}` to `quantization_roots` and uses `weighting="threshold"` in both root scanning and `row_matrix`. A Python JLS input may be a `{twice_J: callable(s_at2) -> real symmetric matrix}` mapping or a model object with `.blocks(s_at2)`. The callback argument is **center-of-mass squared energy**, while the returned roots are **laboratory-frame energy**.

Before changing the example's masses, frame, or energy window, preflight the Zeta domain. For its one-channel setup, a simple dense-window check is:

```python
import numpy as np
from lattice_scattering.kinematics import LatticeFrame, two_body_point
frame = LatticeFrame(24, 1.0, (0, 0, 0))
for energy in np.linspace(0.61, 0.95, 51):
    point = two_body_point(energy_lab_at=float(energy), mass1_at=0.3,
                           mass2_at=0.3, frame=frame)
    assert -48 <= point.q_squared <= 4
    assert 1 <= point.gamma <= 1.52
    assert 0 <= point.alpha <= 1
```

For multiple channels, check each mass pair. Also check `|d|² ≤ 36`, sector degree limits, and `row_matrix` at representative energies. This grid is an operational preflight, not a proof that every intermediate energy is valid.

For a different amplitude, derive its `R^J(s)` from its own K-matrix, phase-space, subtraction, units, and threshold convention. The S-wave formula above is not a general higher-wave conversion. Pass every singularity of the inverse model to `breakpoints_at2`; inspect `model.s_breakpoints()` where available. Confirm the same energy, sectors, group, irrep, parity, and weighting when calling `row_matrix` to check a returned root. The default scan follows ordered eigenvalues, refines sampled extrema/sign brackets and checks matrix residuals. Narrow unresolved zeros can still be missed; compare sample densities and subdivisions. See [root scan limits](root-scan-domain.md).

## 4. Spectrum JSON

[examples/spectrum.json](../examples/spectrum.json) is a complete `coupled-spectrum/v2` file. Its schema tag remains `v2` even though the package has no `lattice_scattering.v2` namespace. Required collections are `ensembles`, `hadrons`, `channels`, `frames`, `levels`, and `covariance`. Channels refer to hadron labels; levels refer to frame IDs. Covariance rows and columns follow **exactly the order of `levels`** and must be finite, symmetric, and positive definite. `units` and `irrep_map` are optional. Omitting `units` defaults to `temporal_lattice`; an explicit value must be exactly `"temporal_lattice"`. Other labels, null, and non-string declarations are rejected, both in JSON and in the `Spectrum` model. No unit conversion is inferred.

`python -m lattice_scattering validate-spectrum examples/spectrum.json` checks the schema and data-model invariants and reports one level and one channel. It rejects `examples/roots.json` and root-scan result JSON, which use different schema tags. It verifies the supported unit declaration but cannot prove that the supplied numbers actually have those units. It does **not** check that an irrep belongs to a group's character table, test a quantization condition, or fit an amplitude. In Python, use `lattice_scattering.io.load_spectrum(path)` and `dump_spectrum(spectrum, path)`.

## 5. Correlated fits and replicas

### Python fit API

[examples/joint_fit.py](../examples/joint_fit.py) runs a small **synthetic** fit to show the API. Its five toy observations share one complete candidate root set, which makes the final grouped `match_levels` check meaningful. A physical fit must replace its toy `predict_roots(theta, observation)` with a call to `quantization_roots` (or `coupled_s_roots` in its supported channel-basis domain) using that observation's frame, irrep, and row. Build one `Observation` per measured level, with `energy` in temporal-lattice units. `JointFitProblem` passes the current parameter vector and a representative observation to the predictor once per group in grouped mode, or once per observation when an explicit per-observation strategy is selected. Pass the full covariance and energy vector to `fit_joint` in exactly the same observation order, and set an explicit matching tolerance.

### Correlated-fit CLI

`python -m lattice_scattering fit examples/fit.json` provides a basic product path for correlated fits without Python glue. The input schema is `lattice-scattering-fit/v1`; the output schema is `lattice-scattering-fit-result/v1`. It uses the same registry-backed S-wave route described above and supports scalar fit parameters in a registered model's JSON constructor parameters.

| Fit field | Requirement and meaning |
| --- | --- |
| `schema` | Required, exactly `lattice-scattering-fit/v1`. |
| `amplitude` | Required registry amplitude object as described above. |
| `fit_parameters` | Required nonempty list of `{path, initial, bounds}`. `path` is a JSON Pointer within `amplitude.parameters`, such as `/couplings/0` or `/coefficients/0/1/1`. It must point to an existing numeric scalar. `initial` replaces that value at the optimizer's starting point. Bounds are an increasing finite `[lower, upper]` pair, and `initial` must lie inside them. |
| `conditions` | Required nonempty list. Each condition has a unique `id`, `frame`, `channel_masses_at`, `energy_window_at`, `irrep`, and nonempty `levels`; optional fields are `row`, `intrinsic_parity`, `channel_intrinsic_parities`, `weighting`, `samples`, `breakpoints_at2`, `root_method`, `subdivisions`, `residual_tol`, `xtol`, and `lhc_domain`. Each level has a globally unique `id` and positive `energy_at`. A condition identifies the complete candidate root list shared by its levels. |
| `covariance` | Required finite, symmetric, positive-definite matrix sized to all levels. Rows and columns follow the listed condition order, then each condition's level order. |
| `matching_tolerance` | Optional positive energy tolerance for the grouped one-to-one level assignment. |
| `solver` | Optional object with `jacobian_scheme`, `difference_step`, `max_evaluations`, `xtol`, `ftol`, and `gtol`. Unknown solver keys are rejected. |

The CLI uses grouped matching: one root list is calculated per condition and observed levels within that condition receive an injective assignment. Parameter bounds, convergence status, chi-squared, degrees of freedom, covariance, optimizer diagnostics, and matching history are returned. A nonconverged fit prints its structured last-iterate result and exits with status 1. The assignment minimizes the package's documented energy-distance matching cost within its tolerance; it is not jointly optimized against the full correlated chi-squared. Root-count changes, missing assignments, and model-domain failures abort the fit rather than silently dropping a level. The full covariance must remain ordered with the flattened levels list.

### Matching behavior

When every observation has an explicit, non-null, hashable `Observation.group`, `JointFitProblem` uses one-to-one grouped matching by default; `matching_strategy="grouped"` states the same intent explicitly. Use the same label only for observations that share the same complete candidate root list. Include ensemble, channel/model condition, frame, irrep, and row as needed to distinguish physical conditions, and give each singleton condition its own label. The program does not infer groups from metadata. The predictor is called once per group using its first observation, so it must return the full root list rather than restrict it to that individual level. Different labels remain independent: the same root energy may be assigned once in each group.

If group labels are absent or only partially supplied without a strategy, construction raises a clear ambiguity error. To request independent per-observation selection explicitly, set `matching_strategy="nearest"` (or a custom callable) and leave every group label unset. That strategy can reuse roots, so use it only when reuse across observations is intended. The API rejects `nearest` or a custom per-observation callback if any group is declared, preserving one-to-one matching inside every explicit quantization group.

Grouped matching restores results to the original observation/covariance order and raises if a group lacks enough eligible roots. `match_levels` uses a global assignment, maximizing the number of eligible matches before minimizing its energy-distance cost (or the documented overlap-based cost). This is not optimization of the assignment under the full correlated chi-squared. Inspect matched pairs, unmatched levels, distances, `fit_joint` convergence, chi-squared, degrees of freedom, and covariance. `problem.predict(...)` still contains selected energies, not the full root list; retain independent final checks of root coverage and matching stability.

For replicas, `lattice_scattering.data.replicas.ReplicaTable` holds aligned sample and observable identities. `lattice_scattering.fitting.fit_replicas(table, fit_sample, parameter_ids=...)` calls `fit_sample(sample_id, observable_mapping)` for each sample. Construct masses, anisotropy, spectrum, and fit weights consistently within that callback. `replica_summary` reports failures and withholds the ensemble mean/covariance when any sample failed. Checkpointed runs additionally need a stable `run_id`.
