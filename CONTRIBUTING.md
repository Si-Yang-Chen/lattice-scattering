# Contributing

Please open an issue with the expected and observed behavior, a minimal
reproducer, package and Python versions, and the relevant physical conventions.
For numerical root issues, include masses, frame, sectors, energy window,
scan settings, and any model breakpoints. Do not post unpublished paper data or
private campaign evidence in a public issue.

For code changes, run `python -m pytest -q` and, when changing numerical kernels,
`python -m pytest -m slow -q`. Add a focused regression for a new failure mode.
Update [usage](docs/usage.md), [conventions](docs/conventions.md), or the
[release scope](docs/release-scope.md) when public behavior or bounds change.
Synthetic regression tests should identify their inputs as synthetic; external
paper comparisons require a separate evidence review before a scientific claim.
