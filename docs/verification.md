# Verification of version 0.1.0

The reviewed public source was verified locally on 2026-09-25 with Windows,
Python 3.11.9, NumPy 1.26.4, and SciPy 1.11.2. The
[cross-platform CI run](https://github.com/Si-Yang-Chen/lattice-scattering/actions/runs/36038605726)
checks the portable product code on Ubuntu 24.04 and Windows:

| Check | Result | Meaning |
| --- | --- | --- |
| Local `python -m pytest -q --tb=short` | 377 passed; 76 slow tests deselected in 196.70 s | Default software regressions |
| Local `python -m pytest -m slow -q --tb=short` | 76 passed; 377 deselected in 590.37 s | Windows extended numerical checks |
| CI default tests: Ubuntu Python 3.11, 3.12, 3.13; Windows Python 3.11 | All four jobs passed | Cross-platform software regressions |
| CI `python -m pytest -m slow -q --tb=short` on Ubuntu 24.04/Python 3.11 | 76 passed; 377 deselected in 532.65 s | Extended numerical checks |
| `python -m build --wheel --sdist --no-isolation` | Passed | Wheel and source distribution built |
| `python -m twine check dist/*` | Both distributions passed | Package metadata and README rendering |
| `python tools/check_release_artifacts.py dist` | Passed | Member allowlist and license checked |
| Installed-wheel CLI from another working directory | Passed; synthetic roots `0.783040725816` and `0.932308172488` | Imports used the installed wheel, not the source tree |

The public suite collects 453 tests. Nine tests of a private historical Git
archive extractor were removed; two historical numerical comparisons now use
frozen matrices in the public test file. The tests include analytic root-search
cases, independent scalar Zeta references, split/cutoff comparisons, synthetic
multi-frame roots, left-hand-cut branch and
path checks, amplitude models, CLI flows, and correlated fitting. They do not
compare published energy levels or covariance matrices against the package.

For a release, run both suites against the exact public source revision, build
both wheel and source distribution, install the built wheel in a clean
environment, run the CLI examples, and record the artifact SHA-256 hashes.
The public CI workflow checks its declared Python/platform matrix; results on
other systems require their own verification.
