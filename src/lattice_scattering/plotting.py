"""Figures from declared scattering results; no paper data are bundled.

Plot builders consume numbers already computed by the physics APIs.  Every
figure requires a visible provenance label so synthetic examples cannot be
mistaken for a comparison with measured lattice energies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class SpectrumPoint:
    volume_sites: int
    energy_cm_at: float
    uncertainty_at: float | None = None
    channel: str | None = None
    used: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.volume_sites, bool) or not isinstance(self.volume_sites, int) or self.volume_sites <= 0:
            raise ValueError("volume_sites must be a positive integer")
        _finite_scalar(self.energy_cm_at, "energy_cm_at", positive=True)
        if self.uncertainty_at is not None:
            _finite_scalar(self.uncertainty_at, "uncertainty_at", nonnegative=True)


@dataclass(frozen=True)
class SpectrumPanel:
    label: str
    observations: tuple[SpectrumPoint, ...]
    predictions: tuple[SpectrumPoint, ...]
    free_references: tuple[SpectrumPoint, ...]
    thresholds_at: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.label.strip() or not self.predictions:
            raise ValueError("spectrum panel needs a label and model predictions")
        for collection in (self.observations, self.predictions, self.free_references):
            if not all(isinstance(point, SpectrumPoint) for point in collection):
                raise ValueError("spectrum entries must be SpectrumPoint values")
        for name, energy in self.thresholds_at.items():
            if not str(name).strip():
                raise ValueError("threshold label must be nonempty")
            _finite_scalar(energy, "threshold energy", positive=True)


@dataclass(frozen=True)
class PoleMarker:
    label: str
    s_at2: complex
    sheet: tuple[int, ...]
    residue: np.ndarray

    def __post_init__(self) -> None:
        if not self.label.strip() or not np.isfinite(self.s_at2):
            raise ValueError("pole marker needs a label and finite complex s")
        if not self.sheet or any(value not in (1, 2) for value in self.sheet):
            raise ValueError("pole sheet must give I/II for every channel")
        residue = np.asarray(self.residue, dtype=complex)
        if residue.shape != (len(self.sheet), len(self.sheet)) or not np.all(np.isfinite(residue)):
            raise ValueError("pole residue must be a finite square channel matrix")


def _finite_scalar(value, name: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite scalar")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not np.isfinite(number) or (positive and number <= 0) or (nonnegative and number < 0):
        raise ValueError(f"{name} is outside its finite domain")
    return number


def _vector(values, name: str, *, increasing: bool = False) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or not len(result) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a nonempty finite vector")
    if increasing and np.any(np.diff(result) <= 0):
        raise ValueError(f"{name} must be strictly increasing")
    return result


def _figure(figsize):
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as exc:
        raise RuntimeError('Plotting requires matplotlib; install "lattice-scattering[plot]"') from exc
    figure = Figure(figsize=figsize, layout="constrained", facecolor="white")
    FigureCanvasAgg(figure)
    return figure


def _provenance(figure, label: str) -> None:
    if not isinstance(label, str) or not label.strip():
        raise ValueError("a visible provenance label is required")
    figure.text(0.012, 0.995, label.strip(), ha="left", va="top", fontsize=8, color="#8f331f")


def spectrum_figure(panels: Sequence[SpectrumPanel], *, provenance: str):
    """Observed and predicted CM levels against volume, with free references.

    Free references are displayed as supplied; this function does not project
    them into an irrep or assign an interacting root to a continuous branch.
    """
    panels = tuple(panels)
    if not panels or not all(isinstance(panel, SpectrumPanel) for panel in panels):
        raise ValueError("at least one SpectrumPanel is required")
    figure = _figure((6.1 * len(panels), 5.0))
    axes = figure.subplots(1, len(panels), sharey=True, squeeze=False)[0]
    for ax, panel in zip(axes, panels):
        for index, (label, energy) in enumerate(panel.thresholds_at.items()):
            ax.axhline(energy, color=("#4373a3", "#799353", "#8a6aa1")[index % 3],
                       ls="--", lw=1, alpha=0.65, label=f"{label} threshold")
        if panel.free_references:
            ax.scatter([p.volume_sites for p in panel.free_references],
                       [p.energy_cm_at for p in panel.free_references],
                       marker="_", s=100, c="#a3aab0", alpha=0.8,
                       label="unprojected free reference")
        ax.scatter([p.volume_sites for p in panel.predictions],
                   [p.energy_cm_at for p in panel.predictions],
                   marker="D", s=42, facecolors="none", edgecolors="#d27d22",
                   linewidths=1.4, label="model roots", zorder=4)
        used = [p for p in panel.observations if p.used]
        excluded = [p for p in panel.observations if not p.used]
        for records, color, label in ((used, "#243b53", "mock observed"),
                                      (excluded, "#949da5", "not fitted")):
            if records:
                ax.errorbar([p.volume_sites for p in records],
                            [p.energy_cm_at for p in records],
                            yerr=[p.uncertainty_at or 0 for p in records],
                            fmt="o", ms=4.5, capsize=2, color=color, label=label, zorder=5)
        volumes = sorted({p.volume_sites for p in panel.predictions})
        ax.set_xticks(volumes)
        ax.set_xlim(min(volumes) - 2, max(volumes) + 2)
        ax.set_title(panel.label)
        ax.set_xlabel(r"$L/a_s$")
        ax.grid(axis="y", alpha=0.18)
        ax.legend(fontsize=7, loc="best")
    axes[0].set_ylabel(r"$a_t E_{\rm cm}$")
    figure.suptitle("Finite-volume spectrum and calculated roots", fontsize=13)
    _provenance(figure, provenance)
    return figure


def amplitude_matrix_figure(
    energies_cm_at, strengths, channel_labels: Sequence[str], thresholds_at,
    *, provenance: str,
):
    """Plot declared ``rho_i rho_j |t_ij|²`` values on the real axis.

    Closed-channel or invalid entries must be NaN. The caller is responsible
    for constructing ``t`` and ``rho`` in the same amplitude convention.
    """
    energies = _vector(energies_cm_at, "energies_cm_at", increasing=True)
    labels = tuple(channel_labels)
    if not labels or any(not str(label).strip() for label in labels):
        raise ValueError("nonempty channel labels required")
    thresholds = _vector(thresholds_at, "thresholds_at")
    n = len(labels)
    values = np.asarray(strengths, dtype=float)
    if thresholds.shape != (n,) or values.shape != (len(energies), n, n):
        raise ValueError("amplitude strengths must have shape (energies, channels, channels)")
    if np.any(np.isinf(values)) or np.any(values[np.isfinite(values)] < 0):
        raise ValueError("amplitude strengths must be nonnegative or NaN")
    pairs = [(i, j) for i in range(n) for j in range(i, n)]
    columns = min(3, len(pairs))
    rows = (len(pairs) + columns - 1) // columns
    figure = _figure((4.4 * columns, 3.45 * rows + 1.0))
    axes = figure.subplots(rows, columns, sharex=True, sharey=True, squeeze=False)
    finite = values[np.isfinite(values)]
    upper = max(1.05, float(np.max(finite) * 1.12) if finite.size else 1.05)
    for index, (i, j) in enumerate(pairs):
        ax = axes.flat[index]
        opening = max(thresholds[i], thresholds[j])
        if energies[0] < opening:
            ax.axvspan(energies[0], opening, color="#e9ecef", alpha=0.85)
        ax.plot(energies, values[:, i, j], color="#285f8f", lw=1.8)
        for channel, threshold in enumerate(thresholds):
            ax.axvline(threshold, color=("#5d85aa", "#789653", "#9876a2")[channel % 3],
                       ls=":", lw=1,
                       label=f"{labels[channel]} threshold" if index == 0 else None)
        ax.set_title(f"{labels[i]} → {labels[j]}", fontsize=10)
        ax.set_xlim(energies[0], energies[-1])
        ax.set_ylim(0, upper)
        ax.grid(alpha=0.15)
        if index // columns == rows - 1:
            ax.set_xlabel(r"$a_t E_{\rm cm}$")
        if index % columns == 0:
            ax.set_ylabel(r"$\rho_i\rho_j |t_{ij}|^2$")
        if index == 0:
            ax.legend(fontsize=7, loc="upper left")
    for ax in list(axes.flat)[len(pairs):]:
        ax.axis("off")
    figure.suptitle("Coupled-channel amplitude strength (open channels)", fontsize=13)
    _provenance(figure, provenance)
    return figure


def root_diagnostic_figure(
    energies_lab_at, eigenvalues, roots_at, residuals,
    *, free_poles_at=(), model_breakpoints_at=(), provenance: str,
):
    """Scaled Hermitian eigenbranches, found roots, and their backward residuals."""
    energies = _vector(energies_lab_at, "energies_lab_at", increasing=True)
    branches = np.asarray(eigenvalues, dtype=float)
    if branches.ndim != 2 or branches.shape[0] != len(energies) or branches.shape[1] < 1 or np.any(np.isinf(branches)):
        raise ValueError("eigenvalues must have one finite-or-NaN row per energy")
    roots = np.asarray(roots_at, dtype=float)
    errors = np.asarray(residuals, dtype=float)
    if roots.ndim != 1 or errors.shape != roots.shape or np.any(~np.isfinite(roots)) or np.any(~np.isfinite(errors)) or np.any(errors < 0):
        raise ValueError("roots and residuals must be aligned finite vectors")
    scale = np.maximum(1.0, np.max(np.where(np.isfinite(branches), np.abs(branches), 0.0), axis=1))
    scaled = branches / scale[:, None]
    figure = _figure((8.4, 6.2))
    top, bottom = figure.subplots(2, 1, sharex=True, height_ratios=(3, 1))
    for index in range(scaled.shape[1]):
        top.plot(energies, scaled[:, index], lw=1.4, label=f"eigenbranch {index + 1}")
    top.axhline(0, color="#303942", lw=0.8)
    for index, pole in enumerate(free_poles_at):
        top.axvline(_finite_scalar(pole, "free pole"), color="#9aa2a9", ls="--", lw=0.8,
                    label="free reference pole" if index == 0 else None)
    for index, point in enumerate(model_breakpoints_at):
        top.axvline(_finite_scalar(point, "model breakpoint"), color="#99739b", ls=":", lw=1,
                    label="declared model breakpoint" if index == 0 else None)
    if len(roots):
        top.scatter(roots, np.zeros_like(roots), marker="o", s=35, color="#dc8124", zorder=5,
                    label="reported root")
        bottom.scatter(roots, np.maximum(errors, np.finfo(float).tiny), s=30, color="#dc8124")
    top.set_ylabel("scaled row-matrix eigenvalue")
    top.set_xlim(energies[0], energies[-1])
    top.grid(alpha=0.18)
    top.legend(fontsize=8, loc="best")
    bottom.set_yscale("log")
    bottom.set_ylabel("root residual")
    bottom.set_xlabel(r"$a_t E_{\rm lab}$")
    bottom.grid(alpha=0.18)
    figure.suptitle("Quantization root diagnostic", fontsize=13)
    _provenance(figure, provenance)
    return figure


def pole_map_figure(
    markers: Sequence[PoleMarker], channel_labels: Sequence[str],
    thresholds_cm_at, *, reference_index: int = 0, provenance: str,
):
    """Complex-energy pole map plus absolute residue matrix for one reference."""
    markers = tuple(markers)
    labels = tuple(channel_labels)
    thresholds = _vector(thresholds_cm_at, "thresholds_cm_at")
    if not markers or not all(isinstance(marker, PoleMarker) for marker in markers):
        raise ValueError("at least one PoleMarker is required")
    if len(labels) != len(thresholds) or any(len(marker.sheet) != len(labels) for marker in markers):
        raise ValueError("pole channels, sheets and thresholds must agree")
    if not 0 <= reference_index < len(markers):
        raise ValueError("reference_index outside pole list")
    figure = _figure((10.0, 4.8))
    left, right = figure.subplots(1, 2, width_ratios=(1.25, 1))
    for index, marker in enumerate(markers):
        energy = np.sqrt(marker.s_at2)
        sheet = "(" + ",".join("I" if v == 1 else "II" for v in marker.sheet) + ")"
        left.scatter(energy.real, energy.imag, s=100 if index == reference_index else 62,
                     marker="*" if index == reference_index else "o", zorder=5,
                     label=f"{marker.label} {sheet}")
    for index, threshold in enumerate(thresholds):
        left.axvline(threshold, color="#9aa2a9", ls="--", lw=0.8,
                     label=f"{labels[index]} threshold")
    left.axhline(0, color="#303942", lw=0.8)
    left.set_xlabel(r"Re $a_t\sqrt{s}$")
    left.set_ylabel(r"Im $a_t\sqrt{s}$")
    left.grid(alpha=0.18)
    left.legend(fontsize=8, loc="best")
    reference = np.abs(np.asarray(markers[reference_index].residue))
    image = right.imshow(reference, cmap="Blues", origin="lower")
    right.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    right.set_yticks(range(len(labels)), labels)
    right.set_title(f"|residue|: {markers[reference_index].label}")
    figure.colorbar(image, ax=right, shrink=0.78)
    figure.suptitle("Locally refined simple poles; sheet labels are explicit", fontsize=13)
    _provenance(figure, provenance)
    return figure


def fit_diagnostic_figure(
    observed_cm_at, predicted_cm_at, covariance_at2,
    *, provenance: str, cutoff_guides=(0.01, 0.02, 0.04),
):
    """Cholesky-whitened residuals and correlation eigenvalues.

    Dashed eigenvalue guides are visual references; no covariance modes are
    removed and no fit is performed by this function.
    """
    observed = _vector(observed_cm_at, "observed_cm_at")
    predicted = _vector(predicted_cm_at, "predicted_cm_at")
    covariance = np.asarray(covariance_at2, dtype=float)
    n = len(observed)
    if predicted.shape != observed.shape or covariance.shape != (n, n) or not np.all(np.isfinite(covariance)):
        raise ValueError("observations, predictions and covariance shapes must agree")
    if not np.allclose(covariance, covariance.T, atol=1e-14, rtol=1e-12):
        raise ValueError("covariance must be symmetric")
    try:
        cholesky = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance must be positive definite") from exc
    whitened = np.linalg.solve(cholesky, observed - predicted)
    sigma = np.sqrt(np.diag(covariance))
    correlation = covariance / sigma[:, None] / sigma[None, :]
    eigenvalues = np.linalg.eigvalsh(correlation)[::-1]
    figure = _figure((10.3, 4.5))
    left, right = figure.subplots(1, 2)
    left.bar(np.arange(n), whitened, color="#486c8e", width=0.75)
    left.axhline(0, color="#303942", lw=0.8)
    left.axhline(1, color="#b5745e", ls="--", lw=0.8)
    left.axhline(-1, color="#b5745e", ls="--", lw=0.8)
    left.set_xlabel("observation index (declared order)")
    left.set_ylabel("Cholesky-whitened residual")
    left.grid(axis="y", alpha=0.18)
    right.semilogy(np.arange(1, n + 1), eigenvalues / eigenvalues[0], "o-", color="#486c8e")
    for guide in cutoff_guides:
        value = _finite_scalar(guide, "cutoff guide", positive=True)
        right.axhline(value, color="#a5a8ab", ls=":", lw=0.8)
    right.set_xlabel("correlation eigenvalue rank")
    right.set_ylabel(r"$\lambda_i/\lambda_1$")
    right.set_title("Guides only: no eigenmode cutoff applied", fontsize=9)
    right.grid(alpha=0.18)
    figure.suptitle("Residual and covariance diagnostics", fontsize=13)
    _provenance(figure, provenance)
    return figure


def save_figure(figure, stem: str | Path, *, formats=("png", "svg"), dpi: int = 160) -> tuple[Path, ...]:
    """Save one figure in standard scientific formats and return absolute paths."""
    target = Path(stem).resolve()
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi < 72:
        raise ValueError("dpi must be an integer >= 72")
    extensions = tuple(formats)
    if not extensions or any(ext not in ("png", "svg", "pdf") for ext in extensions):
        raise ValueError("formats must be png, svg or pdf")
    target.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for extension in extensions:
        output = target.with_suffix("." + extension)
        figure.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
        results.append(output)
    return tuple(results)


__all__ = [
    "SpectrumPoint", "SpectrumPanel", "PoleMarker", "spectrum_figure",
    "amplitude_matrix_figure", "root_diagnostic_figure", "pole_map_figure",
    "fit_diagnostic_figure", "save_figure",
]
