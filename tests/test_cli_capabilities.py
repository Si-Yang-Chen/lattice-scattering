"""CLI plumbing tests for bounded spectrum units and extended JLS options."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import lattice_scattering.__main__ as cli


ROOT = Path(__file__).resolve().parents[1]


def _roots_config() -> dict:
    return json.loads((ROOT / "examples" / "roots.json").read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_roots_forwards_extended_jls_options_and_preserves_result_schema(
    tmp_path, monkeypatch, capsys
):
    data = _roots_config()
    data.pop("sectors")
    data["channel_masses_at"] = [[0.3, 0.3], [0.4, 0.4]]
    data["channel_sectors"] = [[[0, 0]], [[0, 0]]]
    data["channel_intrinsic_parities"] = [1, -1]
    data["selected_j_sectors"] = {"0": [[0, 0, 0], [1, 0, 0]]}
    captured = {}

    def fake_quantization_roots(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return (0.71, 0.82)

    monkeypatch.setattr(cli, "quantization_roots", fake_quantization_roots)
    result = cli.main(["roots", str(_write_json(tmp_path / "roots.json", data))])

    assert result == 0
    assert json.loads(capsys.readouterr().out) == {
        "schema": "lattice-scattering-roots-result/v1",
        "roots_at": [0.71, 0.82],
    }
    assert captured["args"][3] is None
    assert captured["kwargs"]["channel_sectors"] == (((0, 0),), ((0, 0),))
    assert captured["kwargs"]["channel_intrinsic_parities"] == (1, -1)
    assert captured["kwargs"]["selected_j_sectors"] == {0: ((0, 0, 0), (1, 0, 0))}


def test_roots_legacy_config_omits_new_kwargs(monkeypatch, tmp_path):
    captured = {}

    def fake_quantization_roots(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return ()

    monkeypatch.setattr(cli, "quantization_roots", fake_quantization_roots)
    result = cli.main(["roots", str(_write_json(tmp_path / "roots.json", _roots_config()))])

    assert result == 0
    assert captured["args"][3] == ((0, 0),)
    assert "channel_sectors" not in captured["kwargs"]
    assert "channel_intrinsic_parities" not in captured["kwargs"]
    assert "selected_j_sectors" not in captured["kwargs"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("channel_sectors", [[0, 0]], "channel_sectors"),
        ("channel_intrinsic_parities", [0], "channel_intrinsic_parities"),
        ("selected_j_sectors", {"0": [[0, 0, 0]], "00": [[0, 0, 0]]}, "duplicate keys"),
        ("selected_j_sectors", {"0": []}, "cannot be empty"),
        ("selected_j_sectors", {"0": [[0, 0]]}, "must be a triple"),
    ],
)
def test_roots_rejects_malformed_extended_options(tmp_path, capsys, field, value, message):
    data = _roots_config()
    data[field] = value

    result = cli.main(["roots", str(_write_json(tmp_path / "roots.json", data))])

    assert result == 2
    assert message in capsys.readouterr().err


def test_validate_spectrum_rejects_unsupported_units(tmp_path, capsys):
    data = json.loads((ROOT / "examples" / "spectrum.json").read_text(encoding="utf-8"))
    data["units"] = "MeV"

    result = cli.main(["validate-spectrum", str(_write_json(tmp_path / "spectrum.json", data))])

    assert result == 2
    assert "unsupported units" in capsys.readouterr().err
