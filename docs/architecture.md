# Architecture

The public package is `lattice_scattering`; there is no parallel versioned implementation. The `coupled-spectrum/v2` string is a data-schema tag, not a second Python package.

| Package | Responsibility |
| --- | --- |
| `core` | Hadron, channel, ensemble, level, spectrum, and channel-specific `Sector` data models |
| `io` | `coupled-spectrum/v2` JSON serialization and structural validation |
| `kinematics` | Real-axis two-body and free-state kinematics |
| `symmetry` | Little groups, double covers, representations, and subduction |
| `amplitudes` | K-matrix models, phase space, and complex-plane analysis |
| `finite_volume` | Zeta and box matrices, JLS assembly, and root scans |
| `fitting` | Correlated level fits, matching, systematics, and replicas |
| `data` | Replica data structures |

The forward path is: masses and frame → common JLS sector pairs → amplitude-specific reduced-inverse total-J blocks → irrep-row projection → finite-volume determinant → laboratory-frame roots. The `lattice-scattering` CLI supports constant reduced-inverse blocks, registered channel-basis S-wave models and correlated S-wave fits, registry discovery, and spectrum JSON validation. The registry-model adapter explicitly refuses non-S-wave sectors; spinful/higher-wave models, matrix checks, systematics, and replicas remain Python API workflows.

The `finite_volume.quantization_roots` API retains common `(ell, twice_S)` sectors and parity by default. Explicit channel layouts and selected-J incidences use `finite_volume/jls_channels.py` to construct the active irrep-row subspace. `fit_joint` uses grouped one-to-one assignment when every observation has an explicit group label; otherwise callers must explicitly choose a per-observation strategy. See [usage](usage.md) for executable examples and [conventions](conventions.md) for the numerical domain and scientific limits.

This repository incorporates the mathematical and fitting helpers the former V2 code imported. It requires no legacy package or CLI at runtime. Paper extraction and reproduction workflows are outside the public product distribution and are maintained in a separate private development archive.
