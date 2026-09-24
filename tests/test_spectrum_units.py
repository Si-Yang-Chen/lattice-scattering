"""Spectrum unit declarations are explicit and bounded to temporal-lattice units."""
from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from lattice_scattering.core import TEMPORAL_LATTICE_UNITS
from lattice_scattering.io import SchemaError, dump_spectrum, from_dict, load_spectrum, to_dict


ROOT = Path(__file__).resolve().parents[1]


def _example_data() -> dict:
    return json.loads((ROOT / "examples" / "spectrum.json").read_text(encoding="utf-8"))


def test_current_example_units_roundtrip(tmp_path):
    spectrum = load_spectrum(ROOT / "examples" / "spectrum.json")
    assert spectrum.units == TEMPORAL_LATTICE_UNITS

    rebuilt = from_dict(to_dict(spectrum))
    assert rebuilt == spectrum

    output = tmp_path / "spectrum.json"
    dump_spectrum(spectrum, output)
    assert load_spectrum(output) == spectrum


def test_missing_units_keeps_temporal_lattice_default():
    data = _example_data()
    data.pop("units")

    spectrum = from_dict(data)

    assert spectrum.units == TEMPORAL_LATTICE_UNITS
    assert to_dict(spectrum)["units"] == TEMPORAL_LATTICE_UNITS


@pytest.mark.parametrize(
    "units",
    [
        "MeV",
        "temporal_lattice_energy",
        "TEMPORAL_LATTICE",
        "",
        None,
        1,
        [],
        {},
    ],
)
def test_from_dict_rejects_unsupported_or_malformed_units(units):
    data = copy.deepcopy(_example_data())
    data["units"] = units

    with pytest.raises(SchemaError, match=r"units"):
        from_dict(data)


@pytest.mark.parametrize("units", ["MeV", "", None, 1])
def test_model_rejects_unsupported_or_malformed_units(units):
    spectrum = load_spectrum(ROOT / "examples" / "spectrum.json")

    with pytest.raises(ValueError, match=r"units"):
        replace(spectrum, units=units)
