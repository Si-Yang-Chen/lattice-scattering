from pathlib import Path
import json

from lattice_scattering.__main__ import main

ROOT = Path(__file__).resolve().parents[1]


def test_validate_spectrum_command(capsys):
    result = main(["validate-spectrum", str(ROOT / "examples" / "spectrum.json")])
    assert result == 0
    output = json.loads(capsys.readouterr().out)
    assert output == {"schema": "lattice-scattering-validation/v1", "valid": True,
                      "levels": 1, "channels": 1}


def test_roots_command_uses_standalone_jls(capsys):
    result = main(["roots", str(ROOT / "examples" / "roots.json")])
    assert result == 0
    output = json.loads(capsys.readouterr().out)
    assert output["schema"] == "lattice-scattering-roots-result/v1"
    assert len(output["roots_at"]) == 2
    assert abs(output["roots_at"][0] - 0.7830407258164542) < 1e-9
    assert abs(output["roots_at"][1] - 0.9323081724876167) < 1e-9
