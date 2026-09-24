# lattice-scattering

`lattice-scattering` is a Python package for two-body finite-volume lattice
scattering. It provides spectrum data models, kinematics, symmetry projections,
amplitude models, JLS quantization matrices, root searches, correlated fits, and
replica analysis. The current release is **0.1.0**, an initial research software
release with a defined numerical and physical domain.

The package starts from extracted spectra or scattering observables. It does not
generate lattice gauge configurations or correlation functions. Its synthetic
tests check software behavior; they do not establish agreement with published
energy levels, covariances, or amplitudes. See [release scope](docs/release-scope.md)
and [conventions](docs/conventions.md) before drawing a physics conclusion.

## Install

Python 3.11 or newer is required. Install from a release wheel or the public
source checkout:

```sh
python -m pip install .
python -m lattice_scattering --help
```

The runtime dependencies are NumPy and SciPy. The command-line interface is
also available as `lattice-scattering` after installation.

## First run

From a source checkout, run the included synthetic examples:

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

The [usage guide](docs/usage.md) describes the JSON schemas, Python API, and
fit inputs. The [root-search guide](docs/root-scan-domain.md) and
[left-hand-cut guide](docs/left-hand-cut-domain.md) explain the supported domain
and checks. Amplitude model details are in [inverse models](docs/inverse-models.md),
[resonance models](docs/resonance-models.md), and
[pi-pi chiral API](docs/pipi-chiral-api.md).

## Verify and cite

```sh
python -m pip install -e ".[test]"
python -m pytest -q
python -m pytest -q -m slow
```

The [verification summary](docs/verification.md) states what these tests cover
and what remains unverified. To report a reproducible issue, include the package
version, Python/NumPy/SciPy versions, the minimal input, and the full error.
Citation details are in [CITATION.cff](CITATION.cff). This project is distributed
under the [MIT license](LICENSE).
