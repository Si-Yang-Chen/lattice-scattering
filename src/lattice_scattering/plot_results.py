"""Render declared scattering-calculation arrays from JSON without rerunning them."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .plotting import (
    PoleMarker, SpectrumPanel, SpectrumPoint, amplitude_matrix_figure,
    fit_diagnostic_figure, pole_map_figure, root_diagnostic_figure,
    save_figure, spectrum_figure,
)


def render_plot_results(source: dict | str | Path, output_dir: str | Path) -> dict:
    """Render the five views from saved calculation arrays, without recomputing them.

    The input schema is intentionally explicit about units and provenance.
    It can be assembled from other calculations after checking that their
    amplitude, root, and covariance conventions match these plotting fields.
    """
    if isinstance(source, (str, Path)):
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
    elif isinstance(source, dict):
        payload = source
    else:
        raise ValueError("plot results must be a JSON path or object")
    if payload.get("schema") != "lattice-scattering-plot-results/v1":
        raise ValueError("expected lattice-scattering-plot-results/v1")
    if payload.get("units") != "temporal_lattice":
        raise ValueError("plot results must declare temporal_lattice units")
    provenance = payload.get("provenance")
    pole_provenance = payload.get("pole_provenance")
    if not isinstance(provenance, str) or not provenance.strip() or not isinstance(pole_provenance, str) or not pole_provenance.strip():
        raise ValueError("plot results require visible provenance labels")
    panels = tuple(
        SpectrumPanel(
            entry["label"],
            tuple(SpectrumPoint(**point) for point in entry["observations"]),
            tuple(SpectrumPoint(**point) for point in entry["predictions"]),
            tuple(SpectrumPoint(**point) for point in entry["free_references"]),
            entry["thresholds_at"],
        )
        for entry in payload["spectrum_panels"]
    )
    masses = np.asarray(payload["channel_masses_at"], dtype=float)
    labels = tuple(payload["channel_labels"])
    if masses.shape != (len(labels), 2) or not np.all(np.isfinite(masses)) or np.any(masses <= 0):
        raise ValueError("plot channel masses must be positive (N, 2) values")
    thresholds = masses.sum(axis=1)
    amplitude_data = payload["amplitude"]
    root_data = payload["root_diagnostic"]
    pole_markers = tuple(
        PoleMarker(
            entry["label"], complex(*entry["s_at2"]), tuple(entry["sheet"]),
            np.asarray(entry["residue_real"], dtype=float) +
            1j * np.asarray(entry["residue_imag"], dtype=float),
        )
        for entry in payload["poles"]
    )
    destination = Path(output_dir).resolve()
    figures = {
        "spectrum": spectrum_figure(panels, provenance=provenance),
        "amplitudes": amplitude_matrix_figure(
            amplitude_data["energies_cm_at"], amplitude_data["rho_rho_abs_t_squared"],
            labels, thresholds, provenance=provenance,
        ),
        "root_diagnostic": root_diagnostic_figure(
            root_data["energies_lab_at"], root_data["eigenvalues"],
            root_data["roots_lab_at"], root_data["residuals"],
            free_poles_at=root_data["free_poles_lab_at"],
            model_breakpoints_at=root_data["model_breakpoints_lab_at"],
            provenance=provenance,
        ),
        "poles": pole_map_figure(
            pole_markers, labels, thresholds,
            reference_index=payload["pole_reference_index"],
            provenance=pole_provenance,
        ),
        "fit_diagnostics": fit_diagnostic_figure(
            payload["observations_cm_at"], payload["predictions_cm_at"],
            payload["covariance_cm_at2"], provenance=provenance,
        ),
    }
    files = {}
    for name, figure in figures.items():
        files[name] = [str(path) for path in save_figure(figure, destination / name)]
        figure.clear()
    return {"schema": payload["schema"], "synthetic": payload.get("synthetic"),
            "output_dir": str(destination), "figures": files}


__all__ = ["render_plot_results"]
