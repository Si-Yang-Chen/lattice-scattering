"""Fail if a built distribution contains non-product archive material."""

from __future__ import annotations

import sys
import tarfile
import zipfile
from pathlib import Path


PUBLIC_DOCS = {
    "architecture.md",
    "conventions.md",
    "inverse-models.md",
    "left-hand-cut-domain.md",
    "pipi-chiral-api.md",
    "plotting.md",
    "release-scope.md",
    "releasing.md",
    "resonance-models.md",
    "root-scan-domain.md",
    "selective-j.md",
    "usage.md",
    "verification.md",
}
SOURCE_TOP_LEVEL = {
    "CHANGELOG.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "LICENSE",
    "MANIFEST.in",
    "PKG-INFO",
    "README.md",
    "docs",
    "examples",
    "pyproject.toml",
    "setup.cfg",
    "src",
    "tests",
}
FORBIDDEN_PARTS = {
    ".env",
    ".git",
    "broad-mock-100-2026-09-24",
    "developer",
    "paper_validation",
    "selected100-final.json",
    "registry.json",
}


def check_member(name: str, *, source: bool) -> None:
    parts = Path(name.replace("\\", "/")).parts
    if any(part in FORBIDDEN_PARTS or part.lower().endswith(".pdf") for part in parts):
        raise ValueError(f"private or non-product archive member: {name}")
    if not source:
        return
    relative = parts[1:]
    if not relative:
        return
    if relative[0] not in SOURCE_TOP_LEVEL:
        raise ValueError(f"unexpected source distribution member: {name}")
    if relative[0] == "docs" and (len(relative) != 2 or relative[1] not in PUBLIC_DOCS):
        raise ValueError(f"unreviewed document in source distribution: {name}")


def check_artifacts(directory: Path) -> None:
    wheels = sorted(directory.glob("*.whl"))
    sources = sorted(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise ValueError("expected one wheel and one source distribution")
    with zipfile.ZipFile(wheels[0]) as archive:
        members = archive.namelist()
        for name in members:
            check_member(name, source=False)
        if not any(name.endswith("/LICENSE") for name in members):
            raise ValueError("wheel does not contain LICENSE")
    with tarfile.open(sources[0], "r:gz") as archive:
        members = [member.name for member in archive.getmembers() if member.isfile()]
        for name in members:
            check_member(name, source=True)
        if not any(name.endswith("/LICENSE") for name in members):
            raise ValueError("source distribution does not contain LICENSE")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_release_artifacts.py DIST_DIRECTORY")
    check_artifacts(Path(sys.argv[1]))
    print("Distribution member audit passed")
