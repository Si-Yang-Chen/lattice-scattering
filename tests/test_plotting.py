"""Plotting contracts and a full synthetic calculation-to-figure smoke test."""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from lattice_scattering.mock_plots import generate_mock_plots
from lattice_scattering.plot_results import render_plot_results
from lattice_scattering.plotting import (
    PoleMarker,
    SpectrumPanel,
    SpectrumPoint,
    amplitude_matrix_figure,
    fit_diagnostic_figure,
    pole_map_figure,
    root_diagnostic_figure,
    save_figure,
    spectrum_figure,
)


def test_spectrum_figure_needs_declared_points_and_provenance(tmp_path):
    panel = SpectrumPanel(
        "rest A1g",
        (SpectrumPoint(20, 0.72, 0.002),),
        (SpectrumPoint(20, 0.721),),
        (SpectrumPoint(20, 0.71, channel="A+A"),),
        {"A+A": 0.60},
    )
    figure = spectrum_figure([panel], provenance="SYNTHETIC MOCK · not paper data")
    paths = save_figure(figure, tmp_path / "spectrum", formats=("png", "svg"))
    assert paths[0].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert paths[1].is_file()
    assert any("SYNTHETIC MOCK" in label.get_text() for label in figure.texts)
    with pytest.raises(ValueError, match="provenance"):
        spectrum_figure([panel], provenance="")
    with pytest.raises(ValueError, match="positive integer"):
        SpectrumPoint(0, 0.72)


def test_amplitude_figure_masks_closed_channel_and_checks_shape():
    energies = np.array([0.62, 0.68, 0.75])
    strength = np.full((3, 2, 2), np.nan)
    strength[:, 0, 0] = [0.1, 0.2, 0.3]
    strength[1:, 0, 1] = strength[1:, 1, 0] = [0.04, 0.09]
    strength[1:, 1, 1] = [0.12, 0.25]
    figure = amplitude_matrix_figure(
        energies, strength, ("A+A", "B+B"), (0.6, 0.66), provenance="synthetic",
    )
    assert len(figure.axes) == 3
    with pytest.raises(ValueError, match="shape"):
        amplitude_matrix_figure(energies, strength[:, 0, :], ("A+A", "B+B"),
                                (0.6, 0.66), provenance="synthetic")


def test_root_pole_and_covariance_figures_preserve_diagnostic_meanings():
    root_figure = root_diagnostic_figure(
        (0.7, 0.8, 0.9), ((1.0, 2.0), (0.0, 1.0), (-1.0, 0.5)),
        (0.8,), (1e-12,), free_poles_at=(0.85,), provenance="synthetic",
    )
    assert len(root_figure.axes) == 2
    marker = PoleMarker("toy", complex(0.6, -0.02), (2, 2), np.eye(2, dtype=complex))
    pole_figure = pole_map_figure((marker,), ("A+A", "B+B"), (0.6, 0.66), provenance="toy")
    assert len(pole_figure.axes) >= 2
    observed = np.array([0.7, 0.8, 0.9])
    predicted = np.array([0.701, 0.799, 0.9])
    covariance = np.diag([0.002**2, 0.003**2, 0.002**2])
    fit_figure = fit_diagnostic_figure(observed, predicted, covariance, provenance="synthetic")
    assert "no eigenmode cutoff" in fit_figure.axes[1].get_title().lower()
    with pytest.raises(ValueError, match="positive definite"):
        fit_diagnostic_figure(observed, predicted, np.zeros((3, 3)), provenance="synthetic")


def test_saved_result_renderer_requires_schema_units_and_provenance(tmp_path):
    with pytest.raises(ValueError, match="schema|expected"):
        render_plot_results({"schema": "unrelated"}, tmp_path)
    with pytest.raises(ValueError, match="temporal_lattice"):
        render_plot_results({"schema": "lattice-scattering-plot-results/v1", "units": "GeV"}, tmp_path)
    with pytest.raises(ValueError, match="provenance"):
        render_plot_results({"schema": "lattice-scattering-plot-results/v1", "units": "temporal_lattice"}, tmp_path)


@pytest.mark.slow
def test_mock_gallery_uses_calculated_roots_and_writes_valid_images(tmp_path):
    output = generate_mock_plots(tmp_path)
    assert output["synthetic"] is True
    assert len(output["figures"]) == 5
    for paths in output["figures"].values():
        assert len(paths) == 2
        assert paths[0].endswith(".png") and paths[1].endswith(".svg")
        assert Path(paths[0]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    data = json.loads((tmp_path / "mock-results.json").read_text(encoding="utf-8"))
    assert data["synthetic"] and not data["paper_data_compared"]
    assert len(data["conditions"]) == 6
    assert len(data["observations_cm_at"]) == len(data["predictions_cm_at"]) > 15
    assert len(data["poles"]) == 3
    assert max(data["root_diagnostic"]["residuals"]) < 1e-9
    assert data["amplitude"]["rho_rho_abs_t_squared"][0][1][1] is None
    assert data["root_diagnostic"]["eigenvalues"].count([None, None]) > 0
    rendered = render_plot_results(tmp_path / "mock-results.json", tmp_path / "rerendered")
    assert len(rendered["figures"]) == 5
    assert Path(rendered["figures"]["poles"][0]).read_bytes().startswith(b"\x89PNG")
