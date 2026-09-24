# Verification of version 0.1.0

The reviewed public source snapshot was verified locally on 2026-09-25 with
Windows, Python 3.11.9, NumPy 1.26.4, and SciPy 1.11.2:

| Check | Result | Meaning |
| --- | --- | --- |
| `python -m pytest -q --tb=short` | 386 passed; 76 slow tests deselected in 198.54 s | Default software regressions |
| `python -m pytest -m slow -q --tb=short` | 76 passed; 386 deselected in 589.43 s | Extended numerical checks |
| `python -m build --wheel --sdist --no-isolation` | Passed | Wheel and source distribution built |
| `python -m twine check dist/*` | Both distributions passed | Package metadata and README rendering |
| `python tools/check_release_artifacts.py dist` | Passed | Member allowlist and license checked |
| Installed-wheel CLI from another working directory | Passed; synthetic roots `0.783040725816` and `0.932308172488` | Imports used the installed wheel, not the source tree |

The combined suite collected 462 tests. The tests include analytic root-search
cases, independent scalar Zeta references,
split/cutoff comparisons, synthetic multi-frame roots, left-hand-cut branch and
path checks, amplitude models, CLI flows, and correlated fitting. They do not
compare published energy levels or covariance matrices against the package.

For a release, run both suites against the exact public source revision, build
both wheel and source distribution, install the built wheel in a clean
environment, run the CLI examples, and record the artifact SHA-256 hashes.
The public CI workflow checks its declared Python/platform matrix; results on
other systems require their own verification.
