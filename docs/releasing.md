# Release procedure

The public repository is an independent, clean-history export. The private
development repository and its validation archive are separate; making a
branch private inside a public repository is not possible. Export only the
allowlisted product files and inspect the complete public Git history before
publishing it. Do not copy paper PDFs, individual-paper inputs, mock outputs,
registries, campaign state, or private workflow records into the public tree.

For each version:

1. Update `pyproject.toml`, `src/lattice_scattering/__init__.py`,
   `CITATION.cff`, `CHANGELOG.md`, and the scope and verification pages together.
2. Run default and slow tests, build the wheel and source distribution, and
   check package metadata. Repeat the wheel install and CLI smoke test in a
   clean environment. Keep dependency versions and the operating system in the
   release record.
3. Check the source distribution and wheel member lists. Confirm they contain
   the license and public package files, with no private evidence or credentials.
4. Commit the public source, wait for CI, and tag that exact commit as
   `v<version>`. Publish the release notes from `CHANGELOG.md` and attach the
   wheel, source distribution, and SHA-256 hashes.
5. Document any user-visible limitation or failed platform check in the
   release notes. A passing mock test cannot be described as a paper
   reproduction.

Do not reuse an older distribution file after source changes. Build final
artifacts from the tagged revision. Keep the private archive accessible only
to authorized collaborators and maintain its evidence manifest separately.
