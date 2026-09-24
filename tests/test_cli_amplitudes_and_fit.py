"""Configuration-level CLI coverage for registry root scans and correlated fits."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lattice_scattering.__main__ import main
from lattice_scattering.amplitudes import ChiralEREPcotdelta
from lattice_scattering.finite_volume import coupled_s_roots
from lattice_scattering.kinematics import LatticeFrame


ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _energy_dependent_roots_config() -> dict:
    return {
        "schema": "lattice-scattering-roots/v1",
        "frame": {"spatial_sites": 24, "anisotropy": 1.0, "d": [0, 0, 0]},
        "channel_masses_at": [[0.3, 0.3]],
        "sectors": [[0, 0]],
        "irrep": "A1g",
        "energy_window_at": [0.61, 0.95],
        "samples": 80,
        "amplitude": {
            "model": "chiral-ere-pcotdelta",
            "phase_space": "simple",
            "parameters": {
                "scattering_length": 1.0,
                "effective_range": 0.1,
                "pion_mass": 0.3,
                "heavy_mass": 0.3,
                "mass1": 0.3,
                "mass2": 0.3,
            },
        },
    }


@pytest.mark.parametrize("weighting", ["scale", "threshold"])
def test_roots_command_scans_registered_energy_dependent_amplitude(tmp_path, capsys, weighting):
    config = _energy_dependent_roots_config()
    config["weighting"] = weighting
    result = main(["roots", str(_write(tmp_path / f"roots-{weighting}.json", config))])

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    expected = coupled_s_roots(
        LatticeFrame(24, 1.0, (0, 0, 0)),
        [[0.3, 0.3]],
        ChiralEREPcotdelta(**config["amplitude"]["parameters"]),
        "simple",
        (0.61, 0.95),
        samples=80,
    )
    assert payload["schema"] == "lattice-scattering-roots-result/v1"
    np.testing.assert_allclose(payload["roots_at"], expected, rtol=0.0, atol=1e-10)


def test_roots_command_accepts_chew_mandelstam_phase_space(tmp_path, capsys):
    config = _energy_dependent_roots_config()
    config["weighting"] = "threshold"
    config["amplitude"]["phase_space"] = "chew-mandelstam"
    config["amplitude"]["subtractions"] = [0.0]

    assert main(["roots", str(_write(tmp_path / "roots-cm.json", config))]) == 0
    payload = json.loads(capsys.readouterr().out)
    expected = coupled_s_roots(
        LatticeFrame(24, 1.0, (0, 0, 0)),
        [[0.3, 0.3]],
        ChiralEREPcotdelta(**config["amplitude"]["parameters"]),
        "chew-mandelstam",
        (0.61, 0.95),
        subtractions=(0.0,),
        samples=80,
    )
    np.testing.assert_allclose(payload["roots_at"], expected, rtol=0.0, atol=1e-10)


def test_roots_schema_rejects_unknown_fields_and_unsupported_wave(tmp_path, capsys):
    config = _energy_dependent_roots_config()
    config["amplitude"]["normalization_guess"] = "do-not-infer"
    assert main(["roots", str(_write(tmp_path / "bad-amplitude.json", config))]) == 2
    assert "unsupported key" in capsys.readouterr().err

    config = _energy_dependent_roots_config()
    config["sectors"] = [[1, 1]]
    assert main(["roots", str(_write(tmp_path / "bad-wave.json", config))]) == 2
    assert "only sectors [[0, 0]]" in capsys.readouterr().err

    config = _energy_dependent_roots_config()
    config["amplitude"]["model"] = "not-registered"
    assert main(["roots", str(_write(tmp_path / "unknown-model.json", config))]) == 2
    assert "unknown amplitude model" in capsys.readouterr().err


def test_models_command_lists_registry_descriptions(capsys):
    assert main(["models"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "lattice-scattering-models/v1"
    assert "chiral-ere-pcotdelta" in payload["models"]
    assert "scattering_length" in payload["models"]["chiral-ere-pcotdelta"]["required"]


def test_correlated_fit_command_recovers_registered_model_parameter(capsys):
    assert main(["fit", str(ROOT / "examples" / "fit.json")]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "lattice-scattering-fit-result/v1"
    assert payload["status"] == "converged"
    assert abs(payload["parameters"][0] - 0.3) < 1e-6
    assert payload["diagnostics"]["matching_strategy"] == "grouped"
    matches = payload["diagnostics"]["matching"]
    assert all(not item["matching"]["missing_levels"] for item in matches)


def test_correlated_fit_rejects_invalid_covariance_and_parameter_path(tmp_path, capsys):
    data = json.loads((ROOT / "examples" / "fit.json").read_text(encoding="utf-8"))
    data["covariance"] = [[1.0, 0.0], [0.0, -1.0]]
    assert main(["fit", str(_write(tmp_path / "bad-covariance.json", data))]) == 2
    assert "positive definite" in capsys.readouterr().err

    data = json.loads((ROOT / "examples" / "fit.json").read_text(encoding="utf-8"))
    data["fit_parameters"][0]["path"] = "/couplings/2"
    assert main(["fit", str(_write(tmp_path / "bad-path.json", data))]) == 2
    assert "invalid array index" in capsys.readouterr().err
