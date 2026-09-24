"""JSON serialization and validation for the coupled-spectrum/v2 format."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..core.model import (
    Channel,
    Ensemble,
    ExchangeRule,
    Frame,
    Hadron,
    Level,
    Spectrum,
    TEMPORAL_LATTICE_UNITS,
)

__all__ = [
    "SCHEMA",
    "SchemaError",
    "validate",
    "to_dict",
    "from_dict",
    "load_spectrum",
    "dump_spectrum",
]

SCHEMA = "coupled-spectrum/v2"

_TOP_LEVEL_KEYS = (
    "schema",
    "units",
    "ensembles",
    "hadrons",
    "channels",
    "frames",
    "irrep_map",
    "levels",
    "covariance",
)


class SchemaError(ValueError):
    """Raised when a JSON document does not match the v2 coupled-spectrum schema."""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _require_mapping(value: Any, what: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{what} must be an object, got {type(value).__name__}")
    return value


def _require_sequence(value: Any, what: str) -> list:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, (list, tuple)):
        raise SchemaError(f"{what} must be a list, got {type(value).__name__}")
    return list(value)


def _require_keyed(data: Mapping[str, Any], key: str, what: str) -> Any:
    if key not in data:
        raise SchemaError(f"{what} is missing required key {key!r}")
    return data[key]


def _get_str(data: Mapping[str, Any], key: str, what: str) -> str:
    value = _require_keyed(data, key, what)
    if not isinstance(value, str):
        raise SchemaError(f"{what}.{key} must be a string, got {type(value).__name__}")
    return value


def _get_optional(value: Any, default: Any) -> Any:
    return default if value is None else value


def _spectrum_units(root: Mapping[str, Any]) -> str:
    """Read the optional spectrum unit declaration without guessing conversions."""
    if "units" not in root:
        return TEMPORAL_LATTICE_UNITS
    value = root["units"]
    if not isinstance(value, str):
        raise SchemaError(
            "document.units must be the supported unit label "
            f"{TEMPORAL_LATTICE_UNITS!r}, got {type(value).__name__}"
        )
    if value != TEMPORAL_LATTICE_UNITS:
        raise SchemaError(
            f"document.units has unsupported units {value!r}; only "
            f"{TEMPORAL_LATTICE_UNITS!r} is supported"
        )
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _bad_model(exc: Exception, what: str) -> SchemaError:
    return SchemaError(f"invalid {what}: {exc}")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------
def validate(spectrum: Spectrum) -> None:
    """Re-check a :class:`Spectrum` against the schema invariants (no copy)."""
    if not isinstance(spectrum, Spectrum):
        raise SchemaError(f"expected a Spectrum, got {type(spectrum).__name__}")
    unknown = set(spectrum.irrep_map) - {f.id for f in spectrum.frames}
    if unknown:
        raise SchemaError(f"irrep_map references unknown frames: {sorted(unknown)}")
    try:
        rebuilt = Spectrum(
            ensembles=spectrum.ensembles,
            hadrons=spectrum.hadrons,
            channels=spectrum.channels,
            frames=spectrum.frames,
            levels=spectrum.levels,
            covariance=spectrum.covariance,
            irrep_map=spectrum.irrep_map,
            units=spectrum.units,
            require_positive_definite=spectrum.require_positive_definite,
        )
    except (ValueError, TypeError) as exc:
        raise _bad_model(exc, "spectrum") from exc
    del rebuilt


# ---------------------------------------------------------------------------
# to_dict / from_dict
# ---------------------------------------------------------------------------
def to_dict(spectrum: Spectrum) -> dict:
    """Serialize ``spectrum`` to a JSON-compatible dict."""
    if not isinstance(spectrum, Spectrum):
        raise SchemaError(f"expected a Spectrum, got {type(spectrum).__name__}")
    hadron_index = {h.label: h for h in spectrum.hadrons}
    out: dict[str, Any] = {
        "schema": SCHEMA,
        "units": spectrum.units,
        "ensembles": [
            {
                "id": e.id,
                "xi": float(e.xi),
                "xi_unc": float(e.xi_unc),
                "mass_covariance": _jsonable(e.mass_covariance),
            }
            for e in spectrum.ensembles
        ],
        "hadrons": [
            {
                "label": h.label,
                "twice_spin": int(h.twice_spin),
                "parity": int(h.parity),
                "mass_at": float(h.mass_at),
                "mass_unc_at": float(h.mass_unc_at),
                "ensemble": h.ensemble_id,
            }
            for h in spectrum.hadrons
        ],
        "channels": [
            {
                "label": c.label,
                "a": c.hadron_a.label,
                "b": c.hadron_b.label,
                "exchange": str(c.exchange),
            }
            for c in spectrum.channels
        ],
        "frames": [
            {
                "id": f.id,
                "spatial_sites": int(f.spatial_sites),
                "anisotropy": float(f.anisotropy),
                "d": [int(v) for v in f.d],
            }
            for f in spectrum.frames
        ],
        "irrep_map": _jsonable(spectrum.irrep_map),
        "levels": [
            {
                "id": lv.id,
                "frame": lv.frame_id,
                "irrep": lv.irrep,
                "row": int(lv.row),
                "energy_at": float(lv.energy_at),
                "energy_unc_at": float(lv.energy_unc_at),
            }
            for lv in spectrum.levels
        ],
        "covariance": _jsonable(np.asarray(spectrum.covariance, dtype=float)),
    }
    for channel in spectrum.channels:
        if channel.hadron_a.label not in hadron_index or channel.hadron_b.label not in hadron_index:
            raise SchemaError(
                f"channel {channel.label!r} references hadrons outside the hadron list"
            )
    return out


def _hadron_from_dict(item: Any, index: int) -> Hadron:
    data = _require_mapping(item, f"hadrons[{index}]")
    label = _get_str(data, "label", f"hadrons[{index}]")
    ensemble = data.get("ensemble", data.get("ensemble_id"))
    if ensemble is not None and not isinstance(ensemble, str):
        raise SchemaError(f"hadrons[{index}].ensemble must be a string or null")
    try:
        return Hadron(
            label=label,
            twice_spin=_require_keyed(data, "twice_spin", f"hadrons[{index}]"),
            parity=_require_keyed(data, "parity", f"hadrons[{index}]"),
            mass_at=_require_keyed(data, "mass_at", f"hadrons[{index}]"),
            mass_unc_at=_get_optional(data.get("mass_unc_at"), 0.0),
            ensemble_id=ensemble,
        )
    except (ValueError, TypeError) as exc:
        raise _bad_model(exc, f"hadron {label!r}") from exc


def _ensemble_from_dict(item: Any, index: int) -> Ensemble:
    data = _require_mapping(item, f"ensembles[{index}]")
    covariance = data.get("mass_covariance")
    if covariance is not None:
        covariance = tuple(tuple(row) for row in _require_sequence(covariance, f"ensembles[{index}].mass_covariance"))
    try:
        return Ensemble(
            id=_get_str(data, "id", f"ensembles[{index}]"),
            xi=_require_keyed(data, "xi", f"ensembles[{index}]"),
            xi_unc=_get_optional(data.get("xi_unc"), 0.0),
            mass_covariance=covariance,
        )
    except (ValueError, TypeError) as exc:
        raise _bad_model(exc, f"ensemble at index {index}") from exc


def _frame_from_dict(item: Any, index: int) -> Frame:
    data = _require_mapping(item, f"frames[{index}]")
    d = _require_sequence(_get_optional(data.get("d"), [0, 0, 0]), f"frames[{index}].d")
    try:
        return Frame(
            id=_get_str(data, "id", f"frames[{index}]"),
            spatial_sites=_require_keyed(data, "spatial_sites", f"frames[{index}]"),
            anisotropy=_require_keyed(data, "anisotropy", f"frames[{index}]"),
            d=tuple(d),
        )
    except (ValueError, TypeError) as exc:
        raise _bad_model(exc, f"frame at index {index}") from exc


def _level_from_dict(item: Any, index: int) -> Level:
    data = _require_mapping(item, f"levels[{index}]")
    frame_id = data.get("frame", data.get("frame_id"))
    if frame_id is None:
        raise SchemaError(f"levels[{index}] is missing required key 'frame'")
    try:
        return Level(
            id=_get_str(data, "id", f"levels[{index}]"),
            frame_id=frame_id,
            irrep=_get_str(data, "irrep", f"levels[{index}]"),
            row=_require_keyed(data, "row", f"levels[{index}]"),
            energy_at=_require_keyed(data, "energy_at", f"levels[{index}]"),
            energy_unc_at=_get_optional(data.get("energy_unc_at"), 0.0),
        )
    except (ValueError, TypeError) as exc:
        raise _bad_model(exc, f"level at index {index}") from exc


def from_dict(data: Any) -> Spectrum:
    """Deserialize a v2 spectrum dict; raise :class:`SchemaError` on mismatch."""
    root = _require_mapping(data, "document")
    schema = root.get("schema")
    if schema != SCHEMA:
        raise SchemaError(f"unknown schema {schema!r}, expected {SCHEMA!r}")
    units = _spectrum_units(root)

    hadrons = tuple(
        _hadron_from_dict(item, i)
        for i, item in enumerate(_require_sequence(_require_keyed(root, "hadrons", "document"), "hadrons"))
    )
    hadron_by_label = {h.label: h for h in hadrons}
    if len(hadron_by_label) != len(hadrons):
        raise SchemaError("duplicate hadron labels in document")

    channels: list[Channel] = []
    for i, item in enumerate(_require_sequence(_require_keyed(root, "channels", "document"), "channels")):
        spec = _require_mapping(item, f"channels[{i}]")
        label = _get_str(spec, "label", f"channels[{i}]")
        a_label = _get_str(spec, "a", f"channels[{i}]")
        b_label = _get_str(spec, "b", f"channels[{i}]")
        for side in (a_label, b_label):
            if side not in hadron_by_label:
                raise SchemaError(f"channel {label!r} references unknown hadron {side!r}")
        exchange = _get_optional(spec.get("exchange"), ExchangeRule.DISTINGUISHABLE.value)
        try:
            exchange_rule = ExchangeRule(exchange)
        except (ValueError, TypeError) as exc:
            raise SchemaError(f"channels[{i}].exchange is invalid: {exchange!r}") from exc
        try:
            channels.append(
                Channel(
                    label=label,
                    hadron_a=hadron_by_label[a_label],
                    hadron_b=hadron_by_label[b_label],
                    exchange=exchange_rule,
                )
            )
        except (ValueError, TypeError) as exc:
            raise _bad_model(exc, f"channel {label!r}") from exc

    ensembles = tuple(
        _ensemble_from_dict(item, i)
        for i, item in enumerate(_require_sequence(_require_keyed(root, "ensembles", "document"), "ensembles"))
    )
    frames = tuple(
        _frame_from_dict(item, i)
        for i, item in enumerate(_require_sequence(_require_keyed(root, "frames", "document"), "frames"))
    )
    levels = tuple(
        _level_from_dict(item, i)
        for i, item in enumerate(_require_sequence(_require_keyed(root, "levels", "document"), "levels"))
    )

    irrep_map_raw = _require_mapping(_get_optional(root.get("irrep_map"), {}), "irrep_map")
    irrep_map: dict[str, dict[str, str]] = {}
    for frame_id, inner in irrep_map_raw.items():
        inner_map = _require_mapping(inner, f"irrep_map[{frame_id!r}]")
        irrep_map[str(frame_id)] = {str(k): str(v) for k, v in inner_map.items()}

    covariance_raw = _require_keyed(root, "covariance", "document")
    covariance_rows = _require_sequence(covariance_raw, "covariance")
    try:
        covariance = np.asarray(covariance_rows, dtype=float)
    except (ValueError, TypeError) as exc:
        raise SchemaError(f"covariance must be a numeric matrix: {exc}") from exc

    try:
        return Spectrum(
            ensembles=ensembles,
            hadrons=hadrons,
            channels=tuple(channels),
            frames=frames,
            levels=levels,
            covariance=covariance,
            irrep_map=irrep_map,
            units=units,
        )
    except (ValueError, TypeError) as exc:
        raise _bad_model(exc, "spectrum document") from exc


def load_spectrum(path: str | Path) -> Spectrum:
    """Read and validate a v2 spectrum JSON file."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SchemaError(f"{path}: invalid JSON: {exc}") from exc
    return from_dict(raw)


def dump_spectrum(spectrum: Spectrum, path: str | Path) -> None:
    """Validate and write ``spectrum`` as v2 JSON."""
    validate(spectrum)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_dict(spectrum), indent=2, sort_keys=False) + "\n", encoding="utf-8")
