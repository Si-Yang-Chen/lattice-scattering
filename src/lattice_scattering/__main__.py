"""Command-line entry point for the standalone lattice-scattering package."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .cli_workflows import (
    CLIConvergenceError,
    load_config,
    model_listing,
    parse_scan_options,
    run_correlated_fit,
    run_root_scan,
)
from .finite_volume.jls_matrix import quantization_roots
from .io import load_spectrum
from .kinematics import LatticeFrame
from .symmetry import double_cover, little_group


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}")
    return int(value)


def _sector_pairs(value: object, name: str) -> tuple[tuple[int, int], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a sequence of [ell, twice_S] pairs")
    pairs = []
    for index, pair in enumerate(value):
        if isinstance(pair, (str, bytes)) or not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"{name}[{index}] must be a pair [ell, twice_S]")
        pairs.append(
            (
                _integer(pair[0], f"{name}[{index}][0]"),
                _integer(pair[1], f"{name}[{index}][1]"),
            )
        )
    return tuple(pairs)


def _channel_sectors(value: object) -> tuple[tuple[tuple[int, int], ...], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError("channel_sectors must be a sequence of per-channel sector lists")
    return tuple(
        _sector_pairs(channel, f"channel_sectors[{index}]")
        for index, channel in enumerate(value)
    )


def _channel_parities(value: object) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError("channel_intrinsic_parities must be a sequence of +1/-1 values")
    parities = []
    for index, parity in enumerate(value):
        if isinstance(parity, bool) or not isinstance(parity, int) or parity not in (-1, 1):
            raise ValueError(
                f"channel_intrinsic_parities[{index}] must be +1 or -1, got {parity!r}"
            )
        parities.append(int(parity))
    return tuple(parities)


def _selected_j_sectors(value: object) -> dict[int, tuple[tuple[int, int, int], ...]]:
    if not isinstance(value, dict):
        raise ValueError("selected_j_sectors must be an object keyed by twice_J")
    selected: dict[int, tuple[tuple[int, int, int], ...]] = {}
    raw_keys: dict[int, object] = {}
    for raw_key, entries in value.items():
        if not isinstance(raw_key, str):
            raise ValueError(f"selected_j_sectors key must be a string, got {raw_key!r}")
        try:
            twice_j = int(raw_key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"selected_j_sectors key must be an integer, got {raw_key!r}") from exc
        if twice_j < 0:
            raise ValueError(f"selected_j_sectors key must be >= 0, got {raw_key!r}")
        if twice_j in selected:
            raise ValueError(
                "selected_j_sectors has duplicate keys after integer normalization: "
                f"{raw_keys[twice_j]!r} and {raw_key!r}"
            )
        if isinstance(entries, (str, bytes)) or not isinstance(entries, (list, tuple)):
            raise ValueError(f"selected_j_sectors[{raw_key!r}] must be a sequence of triples")
        if not entries:
            raise ValueError(
                f"selected_j_sectors[{raw_key!r}] cannot be empty; omit the key to omit that J"
            )
        incidences = []
        for index, incidence in enumerate(entries):
            if (
                isinstance(incidence, (str, bytes))
                or not isinstance(incidence, (list, tuple))
                or len(incidence) != 3
            ):
                raise ValueError(
                    f"selected_j_sectors[{raw_key!r}][{index}] must be a triple "
                    "[channel, ell, twice_S]"
                )
            incidences.append(
                (
                    _integer(incidence[0], f"selected_j_sectors[{raw_key!r}][{index}][0]"),
                    _integer(incidence[1], f"selected_j_sectors[{raw_key!r}][{index}][1]"),
                    _integer(incidence[2], f"selected_j_sectors[{raw_key!r}][{index}][2]"),
                )
            )
        raw_keys[twice_j] = raw_key
        selected[twice_j] = tuple(incidences)
    return selected


def _roots(config_path: Path) -> dict:
    data = load_config(config_path)
    if data.get("schema") != "lattice-scattering-roots/v1":
        raise ValueError("expected schema 'lattice-scattering-roots/v1'")
    if "amplitude" in data:
        return run_root_scan(data, scanner=quantization_roots)
    if "j_blocks" not in data:
        raise ValueError("root configuration requires either 'j_blocks' or 'amplitude'")
    frame = LatticeFrame(**data["frame"])
    sectors = (
        _sector_pairs(data["sectors"], "sectors")
        if "sectors" in data and data["sectors"] is not None
        else None
    )
    blocks = {int(key): np.asarray(value, dtype=float) for key, value in data["j_blocks"].items()}
    callbacks = {key: (lambda _s, value=value: value) for key, value in blocks.items()}
    group = double_cover(little_group(frame.d))
    optional = parse_scan_options(data)
    if "channel_sectors" in data and data["channel_sectors"] is not None:
        optional["channel_sectors"] = _channel_sectors(data["channel_sectors"])
    if "channel_intrinsic_parities" in data and data["channel_intrinsic_parities"] is not None:
        optional["channel_intrinsic_parities"] = _channel_parities(data["channel_intrinsic_parities"])
    if "selected_j_sectors" in data and data["selected_j_sectors"] is not None:
        optional["selected_j_sectors"] = _selected_j_sectors(data["selected_j_sectors"])
    roots = quantization_roots(
        data["energy_window_at"], frame, data["channel_masses_at"], sectors,
        callbacks, group=group, irrep=data["irrep"], row=data.get("row", 0),
        intrinsic_parity=data.get("intrinsic_parity", 1),
        weighting=data.get("weighting", "scale"), samples=data.get("samples", 60),
        breakpoints_at2=data.get("breakpoints_at2"),
        **optional,
    )
    return {"schema": "lattice-scattering-roots-result/v1", "roots_at": list(roots)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Finite-volume lattice-scattering tools")
    commands = parser.add_subparsers(dest="command", required=True)
    roots_parser = commands.add_parser("roots", help="scan JLS roots from constant blocks or a registered S-wave amplitude")
    roots_parser.add_argument("config", type=Path)
    fit_parser = commands.add_parser("fit", help="run a correlated fit from an S-wave registry-model JSON config")
    fit_parser.add_argument("config", type=Path)
    commands.add_parser("models", help="list registered amplitude models")
    spectrum_parser = commands.add_parser("validate-spectrum", help="validate a coupled-spectrum/v2 JSON file")
    spectrum_parser.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "roots":
            result = _roots(args.config)
        elif args.command == "fit":
            result = run_correlated_fit(load_config(args.config))
        elif args.command == "models":
            result = model_listing()
        else:
            spectrum = load_spectrum(args.path)
            result = {"schema": "lattice-scattering-validation/v1", "valid": True,
                      "levels": len(spectrum.levels), "channels": len(spectrum.channels)}
    except CLIConvergenceError as exc:
        print(json.dumps(exc.result, indent=2))
        return 1
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
        print(f"lattice-scattering: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
