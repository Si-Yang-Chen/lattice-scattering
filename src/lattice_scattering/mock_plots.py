"""Reproducible synthetic calculation and figure gallery.

The numbers generated here are toy-model results. They are not transcribed
energies, covariances, parameter estimates, or figures from a research paper.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .amplitudes import PolePolynomialK, amplitude, effective_inverse_k, physical_rho, solve_poles
from .finite_volume import coupled_s_roots, matrix_root_residual, row_matrix
from .kinematics import LatticeFrame
from .kinematics.free import free_two_body_states
from .plot_results import render_plot_results
from .plotting import PoleMarker, SpectrumPanel, SpectrumPoint
from .symmetry import double_cover, little_group


MASSES_AT = ((0.30, 0.30), (0.33, 0.33))
CHANNELS = ("A+A", "B+B")
PROVENANCE = "SYNTHETIC MOCK · not paper data"


def _model(coupling_scale: float = 1.0) -> PolePolynomialK:
    return PolePolynomialK(
        mass=0.78,
        couplings=np.array([0.14, 0.12]) * coupling_scale,
        coefficients=(np.array([[0.20, 0.02], [0.02, 0.15]]),),
    )


def _cm_energy(energy_lab_at: float, frame: LatticeFrame) -> float:
    return float(np.sqrt(energy_lab_at**2 - sum(momentum**2 for momentum in frame.momentum_at)))


def _free_references(frame: LatticeFrame, maximum: float, window) -> tuple[SpectrumPoint, ...]:
    result = []
    for channel, masses in zip(CHANNELS, MASSES_AT):
        states = free_two_body_states(
            mass1_at=masses[0], mass2_at=masses[1], frame=frame,
            energy_max_at=maximum, max_vectors=100_000,
        )["states"]
        seen = set()
        for state in states:
            energy = float(state["energy_lab_at"])
            if not window[0] <= energy <= window[1]:
                continue
            key = round(energy, 9)
            if key in seen:
                continue
            seen.add(key)
            result.append(SpectrumPoint(frame.spatial_sites, _cm_energy(energy, frame), channel=channel))
    return tuple(result)


def _spectrum_data(model: PolePolynomialK):
    rng = np.random.default_rng(230914071)
    thresholds = {name: sum(masses) for name, masses in zip(CHANNELS, MASSES_AT)}
    panels = []
    predicted_all = []
    uncertainties = []
    condition_results = []
    setups = (
        ("rest [000] A1g", (0, 0, 0), ((16, (.67, .98)), (20, (.67, .98)), (24, (.67, .98)))),
        ("moving [001] A1", (0, 0, 1), ((16, (.75, .94)), (20, (.75, 1.02)), (24, (.75, 1.02)))),
    )
    for label, momentum, volumes in setups:
        predictions = []
        free = []
        for sites, window in volumes:
            frame = LatticeFrame(sites, 1.0, momentum)
            roots = coupled_s_roots(frame, MASSES_AT, model, "chew-mandelstam", window, samples=36)
            if not roots:
                raise RuntimeError(f"mock condition {label}, L={sites} returned no roots")
            for root in roots:
                predicted = _cm_energy(float(root), frame)
                sigma = 0.0015 + 0.00015 * (len(predicted_all) % 4)
                predictions.append(SpectrumPoint(sites, predicted))
                predicted_all.append(predicted)
                uncertainties.append(sigma)
            free.extend(_free_references(frame, window[1], window))
            condition_results.append({
                "label": label, "spatial_sites": sites, "momentum": list(momentum),
                "energy_window_lab_at": list(window), "roots_lab_at": list(map(float, roots)),
                "roots_cm_at": [_cm_energy(float(root), frame) for root in roots],
            })
        panels.append(SpectrumPanel(label, (), tuple(predictions), tuple(free), thresholds))
    sigma = np.asarray(uncertainties)
    indices = np.arange(len(sigma))
    correlation = 0.85 ** np.abs(indices[:, None] - indices[None, :])
    covariance = sigma[:, None] * correlation * sigma[None, :]
    predicted = np.asarray(predicted_all)
    observed = predicted + 0.35 * rng.multivariate_normal(np.zeros(len(sigma)), covariance)
    realized_panels = []
    offset = 0
    for panel in panels:
        count = len(panel.predictions)
        points = tuple(
            SpectrumPoint(point.volume_sites, float(observed[offset + index]),
                          float(sigma[offset + index]))
            for index, point in enumerate(panel.predictions)
        )
        realized_panels.append(SpectrumPanel(
            panel.label, points, panel.predictions, panel.free_references, panel.thresholds_at,
        ))
        offset += count
    return tuple(realized_panels), observed, predicted, covariance, condition_results


def _amplitude_data(model: PolePolynomialK):
    energies = np.linspace(0.58, 0.94, 180)
    strengths = np.full((len(energies), len(MASSES_AT), len(MASSES_AT)), np.nan)
    for index, energy in enumerate(energies):
        t_matrix = amplitude(model, MASSES_AT, "chew-mandelstam", float(energy**2))
        rho = [physical_rho(float(energy**2), *masses) for masses in MASSES_AT]
        for i, masses_i in enumerate(MASSES_AT):
            for j, masses_j in enumerate(MASSES_AT):
                if energy > sum(masses_i) and energy > sum(masses_j):
                    strengths[index, i, j] = float((rho[i] * rho[j]).real * abs(t_matrix[i, j]) ** 2)
    if not np.all(np.isfinite(strengths[np.isfinite(strengths)])):
        raise ArithmeticError("non-finite mock amplitude strength")
    return energies, strengths


def _root_data(model: PolePolynomialK):
    frame = LatticeFrame(20, 1.0, (0, 0, 0))
    window = (0.67, 0.87)
    roots = coupled_s_roots(frame, MASSES_AT, model, "chew-mandelstam", window, samples=48)
    free = []
    for masses in MASSES_AT:
        for state in free_two_body_states(
            mass1_at=masses[0], mass2_at=masses[1], frame=frame,
            energy_max_at=window[1], max_vectors=100_000,
        )["states"]:
            energy = float(state["energy_lab_at"])
            if window[0] < energy < window[1] and all(abs(energy - old) > 1e-9 for old in free):
                free.append(energy)
    breaks = [float(np.sqrt(s)) for s in model.s_breakpoints() if window[0]**2 < s < window[1]**2]
    group = double_cover(little_group(frame.d))

    def reduced_inverse(s_at2):
        inverse = effective_inverse_k(model, MASSES_AT, "chew-mandelstam", s_at2)
        return np.sqrt(s_at2) * frame.length_at / (4 * np.pi) * inverse

    def matrix(energy):
        return row_matrix(
            energy, frame, MASSES_AT, ((0, 0),), {0: reduced_inverse},
            group=group, irrep="A1g", weighting="threshold", pole_policy="flag",
        )

    energies = np.linspace(*window, 170)
    eigenvalues = np.full((len(energies), 2), np.nan)
    display_mask = 0.0006
    for index, energy in enumerate(energies):
        if any(abs(energy - point) < display_mask for point in (*free, *breaks)):
            continue
        eigenvalues[index] = np.linalg.eigvalsh(matrix(float(energy)))
    residuals = [matrix_root_residual(matrix(float(root))) for root in roots]
    return energies, eigenvalues, np.asarray(roots), np.asarray(residuals), free, breaks


def _pole_data():
    markers = []
    for scale in (0.85, 1.0, 1.15):
        poles = solve_poles(
            _model(scale), MASSES_AT, "chew-mandelstam",
            ((0.42, -0.12), (0.75, -0.00001)), sheets=(2, 2),
            initial=((0.62, -0.02),), samples=3,
        )
        if len(poles) != 1:
            raise RuntimeError(f"mock pole scan at coupling scale {scale} found {len(poles)} poles")
        pole = poles[0]
        markers.append(PoleMarker(f"coupling × {scale:.2f}", pole.s, pole.sheet, pole.residue))
    return tuple(markers)


def _json_numeric(value):
    """Represent display-mask NaNs as JSON null without losing numeric data."""
    if isinstance(value, np.ndarray):
        return _json_numeric(value.tolist())
    if isinstance(value, list):
        return [_json_numeric(entry) for entry in value]
    number = float(value)
    return number if np.isfinite(number) else None


def generate_mock_plots(output_dir: str | Path) -> dict:
    """Compute one two-channel toy model and write five figures plus JSON data."""
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    model = _model()
    panels, observed, predicted, covariance, conditions = _spectrum_data(model)
    energies, strengths = _amplitude_data(model)
    root_energies, eigenvalues, roots, residuals, free_poles, breaks = _root_data(model)
    poles = _pole_data()
    payload = {
        "schema": "lattice-scattering-plot-results/v1",
        "synthetic": True,
        "paper_data_compared": False,
        "paper_reproduction": False,
        "provenance": PROVENANCE,
        "pole_provenance": "TOY SWEEP · not paper data",
        "pole_reference_index": 1,
        "description": "two-channel pole-plus-constant-K mock; plots use calculated results",
        "units": "temporal_lattice",
        "channel_labels": list(CHANNELS),
        "channel_masses_at": [list(pair) for pair in MASSES_AT],
        "model": {"name": "PolePolynomialK", "mass": model.mass,
                  "couplings": model.couplings.tolist(),
                  "background": model.coefficients[0].tolist(),
                  "phase_space": "chew-mandelstam"},
        "conditions": conditions,
        "observations_cm_at": observed.tolist(),
        "predictions_cm_at": predicted.tolist(),
        "covariance_cm_at2": covariance.tolist(),
        "spectrum_panels": [
            {"label": panel.label,
             "observations": [asdict(point) for point in panel.observations],
             "predictions": [asdict(point) for point in panel.predictions],
             "free_references": [asdict(point) for point in panel.free_references],
             "thresholds_at": dict(panel.thresholds_at)}
            for panel in panels
        ],
        "amplitude": {"energies_cm_at": energies.tolist(),
                      "rho_rho_abs_t_squared": _json_numeric(strengths)},
        "root_diagnostic": {"energies_lab_at": root_energies.tolist(),
                            "eigenvalues": _json_numeric(eigenvalues),
                            "roots_lab_at": roots.tolist(), "residuals": residuals.tolist(),
                            "free_poles_lab_at": free_poles,
                            "model_breakpoints_lab_at": breaks},
        "poles": [{"label": pole.label, "s_at2": [pole.s_at2.real, pole.s_at2.imag],
                   "sheet": list(pole.sheet),
                   "residue_real": pole.residue.real.tolist(),
                   "residue_imag": pole.residue.imag.tolist()}
                  for pole in poles],
        "figures": {name: [f"{name}.png", f"{name}.svg"] for name in
                    ("spectrum", "amplitudes", "root_diagnostic", "poles", "fit_diagnostics")},
    }
    result_path = destination / "mock-results.json"
    result_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    rendered = render_plot_results(payload, destination)
    return {**rendered, "result": str(result_path)}


__all__ = ["generate_mock_plots"]
