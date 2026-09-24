"""Validated configuration workflows used by the public command-line interface.

The registry models currently expose channel-basis S-wave K matrices.  This
module adapts that explicitly supported domain to the JLS reduced-inverse
convention; it does not guess a higher-wave or spinful conversion.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
from pathlib import Path
import re

import numpy as np

from .amplitudes import build as build_amplitude
from .amplitudes import effective_inverse_k, registered as registered_amplitudes
from .fitting import JointFitProblem, Observation, fit_joint
from .fitting.joint import FitConvergenceError
from .finite_volume.jls_matrix import quantization_roots
from .kinematics import LatticeFrame
from .symmetry import double_cover, little_group

__all__ = ["load_config", "run_root_scan", "run_correlated_fit", "model_listing"]


def load_config(path: Path) -> dict:
    """Read one JSON object and give malformed documents a contextual error."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error
    return _object(value, "configuration")


def model_listing() -> dict:
    """Return registry names and their constructor requirements for discovery."""
    return {"schema": "lattice-scattering-models/v1", "models": registered_amplitudes()}


def run_root_scan(data: Mapping, *, scanner=quantization_roots) -> dict:
    """Run an S-wave energy-dependent registered amplitude from a roots config."""
    _check_root_amplitude_config(data)
    context = _root_context(data)
    return _scan_context(context, scanner=scanner)


def run_correlated_fit(data: Mapping) -> dict:
    """Fit registered amplitude parameters to grouped, correlated energy levels."""
    _keys(
        data,
        required={"schema", "amplitude", "fit_parameters", "conditions", "covariance"},
        optional={"matching_tolerance", "solver"},
        what="fit configuration",
    )
    if data["schema"] != "lattice-scattering-fit/v1":
        raise ValueError("expected schema 'lattice-scattering-fit/v1'")

    amplitude_config = _amplitude_config(data["amplitude"], n_channels=None, validate_model=False)
    base_parameters = amplitude_config["parameters"]
    fit_specs = _fit_parameter_specs(data["fit_parameters"], base_parameters)
    if not fit_specs:
        raise ValueError("fit_parameters must contain at least one free scalar parameter")
    initial_parameters = deepcopy(base_parameters)
    for spec in fit_specs:
        _set_json_pointer(initial_parameters, spec["segments"], spec["initial"], spec["path"])
    _validate_amplitude_model(
        amplitude_config["model"],
        build_amplitude(amplitude_config["model"], initial_parameters),
    )

    conditions = _fit_conditions(data["conditions"])
    channel_counts = {len(condition["masses"]) for condition in conditions}
    if len(channel_counts) != 1:
        raise ValueError("all fit conditions must use the same number of channels")
    n_channels = next(iter(channel_counts))
    if amplitude_config["subtractions"] and len(amplitude_config["subtractions"]) != n_channels:
        raise ValueError(
            f"amplitude.subtractions must contain one entry per channel ({n_channels})"
        )

    level_specs = []
    observations = []
    for condition in conditions:
        for level in condition["levels"]:
            level_specs.append(level)
            observations.append(
                Observation(
                    level_id=level["id"],
                    frame=condition["frame"],
                    frame_id=condition["id"],
                    irrep=condition["irrep"],
                    row=condition["row"],
                    energy=level["energy_at"],
                    group=condition["id"],
                )
            )

    energies = np.asarray([level["energy_at"] for level in level_specs], dtype=float)
    covariance = _covariance(data["covariance"], len(energies))
    initial = np.asarray([item["initial"] for item in fit_specs], dtype=float)
    lower = np.asarray([item["lower"] for item in fit_specs], dtype=float)
    upper = np.asarray([item["upper"] for item in fit_specs], dtype=float)
    if len(energies) <= len(initial):
        raise ValueError("the fit needs more measured levels than free parameters")
    tolerance = data.get("matching_tolerance")
    if tolerance is not None:
        tolerance = _number(tolerance, "matching_tolerance", positive=True)
    solver_options = _solver_options(data.get("solver", {}))

    by_id = {condition["id"]: condition for condition in conditions}

    def predict_roots(theta, observation):
        parameters = deepcopy(base_parameters)
        for value, spec in zip(theta, fit_specs):
            _set_json_pointer(parameters, spec["segments"], float(value), spec["path"])
        model = build_amplitude(amplitude_config["model"], parameters)
        condition = by_id[observation.group]
        return _scan_context(
            _context_with_model(condition, amplitude_config, model),
        )["roots_at"]

    problem = JointFitProblem(
        observations=tuple(observations),
        predict_roots=predict_roots,
        matching_strategy="grouped",
        tolerance=tolerance,
    )
    try:
        result = fit_joint(
            problem,
            initial,
            covariance=covariance,
            energies=energies,
            bounds=(lower, upper),
            **solver_options,
        )
    except FitConvergenceError as error:
        result = dict(error.result)
        result["schema"] = "lattice-scattering-fit-result/v1"
        result["diagnostics"]["matching"] = list(problem.match_history)
        result["diagnostics"]["n_observations"] = len(observations)
        result["diagnostics"]["matching_strategy"] = "grouped"
        result["diagnostics"]["matching_root_uniqueness"] = "enforced within explicit conditions"
        result["diagnostics"]["fit_parameter_paths"] = [item["path"] for item in fit_specs]
        result["diagnostics"]["condition_ids"] = [item["id"] for item in conditions]
        result["diagnostics"]["root_scan_model_domain"] = "registered channel-basis S-wave"
        raise CLIConvergenceError(result) from error

    result["schema"] = "lattice-scattering-fit-result/v1"
    result["diagnostics"]["fit_parameter_paths"] = [item["path"] for item in fit_specs]
    result["diagnostics"]["condition_ids"] = [item["id"] for item in conditions]
    result["diagnostics"]["root_scan_model_domain"] = "registered channel-basis S-wave"
    return result


class CLIConvergenceError(RuntimeError):
    """A fit failed to converge but has a serializable last-iterate result."""

    def __init__(self, result: dict):
        self.result = result
        super().__init__(result.get("diagnostics", {}).get("optimizer_message", "fit did not converge"))


SCAN_OPTION_FIELDS = {"root_method", "subdivisions", "residual_tol", "xtol", "lhc_domain"}


def parse_scan_options(data):
    """Shared explicit scan controls for constant/model roots and fit conditions."""
    options = {}
    if "root_method" in data:
        options["root_method"] = _choice(data["root_method"], "root_method", {"eigenvalues", "determinant"})
    if "subdivisions" in data:
        depth = _integer(data["subdivisions"], "subdivisions")
        if depth > 8:
            raise ValueError("subdivisions must lie in 0..8")
        options["subdivisions"] = depth
    for name in ("residual_tol", "xtol"):
        if name in data:
            options[name] = _number(data[name], name, positive=True)
    if options.get("residual_tol", 1e-10) >= 1:
        raise ValueError("residual_tol must lie in (0, 1)")
    if "lhc_domain" in data:
        from .amplitudes.left_hand_cut import EqualMassExchangeDomain
        spec = data["lhc_domain"]
        _keys(spec, required={"mass"}, optional={"exchange_mass", "exchange_allowed", "coupling", "mechanism", "units"}, what="lhc_domain")
        options["lhc_domain"] = EqualMassExchangeDomain(**spec)
    return options


def _check_root_amplitude_config(data: Mapping) -> None:
    _keys(
        data,
        required={"schema", "frame", "channel_masses_at", "energy_window_at", "irrep", "amplitude"},
        optional={
            "sectors", "channel_sectors", "channel_intrinsic_parities", "selected_j_sectors",
            "row", "intrinsic_parity", "weighting", "samples", "breakpoints_at2", "j_blocks",
        } | SCAN_OPTION_FIELDS,
        what="root configuration",
    )
    if data["schema"] != "lattice-scattering-roots/v1":
        raise ValueError("expected schema 'lattice-scattering-roots/v1'")
    if "j_blocks" in data:
        raise ValueError("amplitude and j_blocks are mutually exclusive")
    if "selected_j_sectors" in data:
        raise ValueError("selected_j_sectors is not supported for registry amplitudes")


def _root_context(data: Mapping) -> dict:
    frame = _frame(data["frame"], "frame")
    masses = _masses(data["channel_masses_at"], "channel_masses_at")
    window = _window(data["energy_window_at"], "energy_window_at")
    irrep = _label(data["irrep"], "irrep")
    amplitude = _amplitude_config(data["amplitude"], n_channels=len(masses))
    sectors = data.get("sectors")
    channel_sectors = data.get("channel_sectors")
    if channel_sectors is not None and sectors is not None:
        raise ValueError("pass either sectors or channel_sectors, not both")
    if channel_sectors is not None:
        parsed_channel_sectors = _channel_sectors(channel_sectors, len(masses))
        if any(channel != ((0, 0),) for channel in parsed_channel_sectors):
            raise ValueError(
                "registry amplitudes currently support only [[0, 0]] per channel (spinless S wave)"
            )
        common_sectors = None
    else:
        parsed_channel_sectors = None
        common_sectors = _sector_pairs(sectors, "sectors") if sectors is not None else ((0, 0),)
    if common_sectors is not None and common_sectors != ((0, 0),):
        raise ValueError(
            "registry amplitudes currently support only sectors [[0, 0]] (spinless S wave)"
        )
    intrinsic_parity = _parity(data.get("intrinsic_parity", 1), "intrinsic_parity")
    channel_parities = (
        _parities(data["channel_intrinsic_parities"], len(masses))
        if "channel_intrinsic_parities" in data else None
    )
    if channel_parities is not None and intrinsic_parity != 1:
        raise ValueError("omit intrinsic_parity when channel_intrinsic_parities is supplied")
    return _make_context(
        frame=frame,
        masses=masses,
        window=window,
        irrep=irrep,
        row=_integer(data.get("row", 0), "row"),
        intrinsic_parity=intrinsic_parity,
        channel_parities=channel_parities,
        sectors=common_sectors,
        channel_sectors=parsed_channel_sectors,
        weighting=_choice(data.get("weighting", "scale"), "weighting", {"scale", "threshold"}),
        samples=_integer(data.get("samples", 60), "samples", minimum=2),
        breakpoints=_breakpoints(data.get("breakpoints_at2", []), "breakpoints_at2"),
        amplitude=amplitude,
        scan_options=parse_scan_options(data),
    )


def _fit_conditions(value) -> list[dict]:
    conditions = _sequence(value, "conditions")
    if not conditions:
        raise ValueError("conditions must not be empty")
    out = []
    seen_ids = set()
    seen_level_ids = set()
    for index, raw in enumerate(conditions):
        where = f"conditions[{index}]"
        _keys(
            raw,
            required={"id", "frame", "channel_masses_at", "energy_window_at", "irrep", "levels"},
            optional={"row", "intrinsic_parity", "channel_intrinsic_parities", "weighting", "samples", "breakpoints_at2"} | SCAN_OPTION_FIELDS,
            what=where,
        )
        condition_id = _label(raw["id"], f"{where}.id")
        if condition_id in seen_ids:
            raise ValueError(f"duplicate condition id {condition_id!r}")
        seen_ids.add(condition_id)
        masses = _masses(raw["channel_masses_at"], f"{where}.channel_masses_at")
        frame = _frame(raw["frame"], f"{where}.frame")
        levels = _sequence(raw["levels"], f"{where}.levels")
        if not levels:
            raise ValueError(f"{where}.levels must not be empty")
        parsed_levels = []
        for j, level in enumerate(levels):
            level_where = f"{where}.levels[{j}]"
            _keys(level, required={"id", "energy_at"}, optional=set(), what=level_where)
            level_id = _label(level["id"], f"{level_where}.id")
            if level_id in seen_level_ids:
                raise ValueError(f"duplicate level id {level_id!r}")
            seen_level_ids.add(level_id)
            energy = _number(level["energy_at"], f"{level_where}.energy_at", positive=True)
            parsed_levels.append({"id": level_id, "energy_at": energy})
        intrinsic_parity = _parity(raw.get("intrinsic_parity", 1), f"{where}.intrinsic_parity")
        channel_parities = (
            _parities(raw["channel_intrinsic_parities"], len(masses), f"{where}.channel_intrinsic_parities")
            if "channel_intrinsic_parities" in raw else None
        )
        if channel_parities is not None and intrinsic_parity != 1:
            raise ValueError(f"omit {where}.intrinsic_parity when channel_intrinsic_parities is supplied")
        out.append(
            _make_context(
                id=condition_id,
                frame=frame,
                masses=masses,
                window=_window(raw["energy_window_at"], f"{where}.energy_window_at"),
                irrep=_label(raw["irrep"], f"{where}.irrep"),
                row=_integer(raw.get("row", 0), f"{where}.row"),
                intrinsic_parity=intrinsic_parity,
                channel_parities=channel_parities,
                weighting=_choice(raw.get("weighting", "threshold"), f"{where}.weighting", {"scale", "threshold"}),
                samples=_integer(raw.get("samples", 60), f"{where}.samples", minimum=2),
                breakpoints=_breakpoints(raw.get("breakpoints_at2", []), f"{where}.breakpoints_at2"),
                levels=parsed_levels,
                scan_options=parse_scan_options(raw),
            )
        )
    return out


def _amplitude_config(value, *, n_channels: int | None, validate_model: bool = True) -> dict:
    _keys(
        value,
        required={"model", "parameters"},
        optional={"phase_space", "subtractions"},
        what="amplitude",
    )
    model_name = _label(value["model"], "amplitude.model")
    parameters = _object(value["parameters"], "amplitude.parameters")
    phase_space = _choice(value.get("phase_space", "simple"), "amplitude.phase_space", {"simple", "chew-mandelstam"})
    subtractions = _number_sequence(value.get("subtractions", []), "amplitude.subtractions")
    if phase_space == "simple" and subtractions:
        raise ValueError("amplitude.subtractions is only used with phase_space='chew-mandelstam'")
    if n_channels is not None and subtractions and len(subtractions) != n_channels:
        raise ValueError(f"amplitude.subtractions must have one entry per channel ({n_channels})")
    # Resolve names and parameter spelling now, before any root evaluations.
    if validate_model:
        model_object = build_amplitude(model_name, parameters)
        _validate_amplitude_model(model_name, model_object)
    return {"model": model_name, "parameters": deepcopy(dict(parameters)), "phase_space": phase_space, "subtractions": subtractions}


def _validate_amplitude_model(model_name: str, model_object) -> None:
    if not callable(getattr(model_object, "inverse", None)):
        raise ValueError(f"registered model {model_name!r} has no inverse(s) required for a real-axis root scan")
    if hasattr(model_object, "s_breakpoints") and not callable(model_object.s_breakpoints):
        raise ValueError(f"registered model {model_name!r}.s_breakpoints must be callable")


def _make_context(**values) -> dict:
    """Create the normalized S-wave scan record used by roots and fit."""
    return dict(values)


def _context_with_model(condition: Mapping, amplitude: Mapping, model) -> dict:
    return _make_context(
        frame=condition["frame"],
        masses=condition["masses"],
        window=condition["window"],
        irrep=condition["irrep"],
        row=condition["row"],
        intrinsic_parity=condition["intrinsic_parity"],
        channel_parities=condition["channel_parities"],
        weighting=condition["weighting"],
        samples=condition["samples"],
        breakpoints=condition["breakpoints"],
        amplitude={**amplitude, "model_object": model},
        scan_options=condition["scan_options"],
    )


def _scan_context(context: Mapping, *, scanner=quantization_roots) -> dict:
    frame = context["frame"]
    masses = context["masses"]
    amplitude = context["amplitude"]
    model = amplitude.get("model_object") or build_amplitude(amplitude["model"], amplitude["parameters"])
    subtractions = amplitude["subtractions"]
    if subtractions and len(subtractions) != len(masses):
        raise ValueError(f"amplitude.subtractions must have one entry per channel ({len(masses)})")

    def reduced_inverse(s_at2):
        effective = effective_inverse_k(
            model, masses, amplitude["phase_space"], s_at2, subtractions=subtractions
        )
        if context["weighting"] == "threshold":
            factor = np.sqrt(s_at2) * frame.length_at / (4.0 * np.pi)
        else:
            factor = np.sqrt(s_at2) / 2.0
        return factor * effective

    model_breakpoints = ()
    if hasattr(model, "s_breakpoints"):
        try:
            raw_breakpoints = model.s_breakpoints()
            model_breakpoints = tuple(
                _number(item, f"{amplitude['model']}.s_breakpoints()[{index}]")
                for index, item in enumerate(raw_breakpoints)
            )
        except TypeError as error:
            raise ValueError(f"{amplitude['model']}.s_breakpoints() must return a sequence") from error
    breakpoints = sorted(set(float(v) for v in (*context["breakpoints"], *model_breakpoints)))
    optional = dict(context.get("scan_options", {}))
    if context["channel_parities"] is not None:
        optional["channel_intrinsic_parities"] = context["channel_parities"]
    if context.get("channel_sectors") is not None:
        optional["channel_sectors"] = context["channel_sectors"]
    roots = scanner(
        context["window"],
        frame,
        masses,
        context.get("sectors", ((0, 0),)),
        {0: reduced_inverse},
        group=double_cover(little_group(frame.d)),
        irrep=context["irrep"],
        row=context["row"],
        intrinsic_parity=context["intrinsic_parity"],
        weighting=context["weighting"],
        samples=context["samples"],
        breakpoints_at2=breakpoints,
        **optional,
    )
    return {"schema": "lattice-scattering-roots-result/v1", "roots_at": list(roots)}


def _fit_parameter_specs(value, parameters: Mapping) -> list[dict]:
    entries = _sequence(value, "fit_parameters")
    out = []
    used = set()
    for index, raw in enumerate(entries):
        where = f"fit_parameters[{index}]"
        _keys(raw, required={"path", "initial", "bounds"}, optional=set(), what=where)
        path = _label(raw["path"], f"{where}.path")
        segments = _json_pointer(path)
        if segments in used:
            raise ValueError(f"duplicate fit parameter path {path!r}")
        used.add(segments)
        current = _get_json_pointer(parameters, segments, path)
        if isinstance(current, (dict, list)) or isinstance(current, bool) or not isinstance(current, (int, float)):
            raise ValueError(f"{where}.path must point to an existing numeric scalar in amplitude.parameters")
        initial = _number(raw["initial"], f"{where}.initial")
        bounds = _number_sequence(raw["bounds"], f"{where}.bounds")
        if len(bounds) != 2 or not bounds[0] < bounds[1]:
            raise ValueError(f"{where}.bounds must be an increasing [lower, upper] pair")
        if not bounds[0] <= initial <= bounds[1]:
            raise ValueError(f"{where}.initial must lie within its bounds")
        out.append({"path": path, "segments": segments, "initial": initial, "lower": bounds[0], "upper": bounds[1]})
    return out


def _solver_options(value) -> dict:
    _keys(value, required=set(), optional={"jacobian_scheme", "difference_step", "max_evaluations", "xtol", "ftol", "gtol"}, what="solver")
    out = {}
    if "jacobian_scheme" in value:
        out["jacobian_scheme"] = _choice(value["jacobian_scheme"], "solver.jacobian_scheme", {"2-point", "3-point"})
    if "difference_step" in value:
        out["difference_step"] = _number(value["difference_step"], "solver.difference_step", positive=True)
    if "max_evaluations" in value:
        out["max_evaluations"] = _integer(value["max_evaluations"], "solver.max_evaluations", minimum=1)
    for key in ("xtol", "ftol", "gtol"):
        if key in value:
            out[key] = _number(value[key], f"solver.{key}", positive=True)
    return out


def _covariance(value, n: int) -> np.ndarray:
    covariance = np.asarray(_number_tree(value, "covariance"), dtype=float)
    if covariance.shape != (n, n):
        raise ValueError(f"covariance must have shape ({n}, {n}), got {covariance.shape}")
    if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-14):
        raise ValueError("covariance must be symmetric")
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError("covariance must be positive definite") from error
    return covariance


def _frame(value, what: str) -> LatticeFrame:
    _keys(value, required={"spatial_sites", "anisotropy"}, optional={"d"}, what=what)
    if "d" in value:
        d = _integer_sequence(value["d"], f"{what}.d", exact_length=3, minimum=None)
    else:
        d = (0, 0, 0)
    try:
        return LatticeFrame(
            spatial_sites=_integer(value["spatial_sites"], f"{what}.spatial_sites", minimum=1),
            anisotropy=_number(value["anisotropy"], f"{what}.anisotropy", positive=True),
            d=d,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {what}: {error}") from error


def _masses(value, what: str) -> np.ndarray:
    rows = _sequence(value, what)
    if not rows:
        raise ValueError(f"{what} must contain at least one channel")
    parsed = []
    for index, row in enumerate(rows):
        pair = _number_sequence(row, f"{what}[{index}]")
        if len(pair) != 2 or any(mass <= 0 for mass in pair):
            raise ValueError(f"{what}[{index}] must be a pair of positive masses")
        parsed.append(pair)
    return np.asarray(parsed, dtype=float)


def _window(value, what: str) -> tuple[float, float]:
    values = _number_sequence(value, what)
    if len(values) != 2 or values[0] <= 0 or values[1] <= values[0]:
        raise ValueError(f"{what} must be a positive increasing [low, high] pair")
    return values[0], values[1]


def _breakpoints(value, what: str) -> tuple[float, ...]:
    values = _number_sequence(value, what)
    return tuple(sorted(set(values)))


def _sector_pairs(value, what: str) -> tuple[tuple[int, int], ...]:
    pairs = _sequence(value, what)
    return tuple(_integer_sequence(pair, f"{what}[{i}]", exact_length=2) for i, pair in enumerate(pairs))


def _channel_sectors(value, n_channels: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    channels = _sequence(value, "channel_sectors")
    if len(channels) != n_channels:
        raise ValueError(f"channel_sectors must contain one entry per channel ({n_channels})")
    return tuple(_sector_pairs(channel, f"channel_sectors[{i}]") for i, channel in enumerate(channels))


def _parities(value, n_channels: int, what: str = "channel_intrinsic_parities") -> tuple[int, ...]:
    values = _integer_sequence(value, what, exact_length=n_channels)
    if any(item not in (-1, 1) for item in values):
        raise ValueError(f"{what} entries must be +1 or -1")
    return values


def _parity(value, what: str) -> int:
    parsed = _integer(value, what, minimum=None)
    if parsed not in (-1, 1):
        raise ValueError(f"{what} must be +1 or -1")
    return parsed


def _integer_sequence(value, what: str, *, exact_length: int | None = None, minimum: int | None = 0) -> tuple[int, ...]:
    values = _sequence(value, what)
    if exact_length is not None and len(values) != exact_length:
        raise ValueError(f"{what} must contain exactly {exact_length} integers")
    return tuple(_integer(item, f"{what}[{i}]", minimum=minimum) for i, item in enumerate(values))


def _integer(value, what: str, *, minimum: int | None = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{what} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{what} must be >= {minimum}")
    return value


def _number(value, what: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{what} must be a finite real number")
    try:
        value = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{what} must be a finite real number") from error
    if not np.isfinite(value):
        raise ValueError(f"{what} must be a finite real number")
    if positive and value <= 0:
        raise ValueError(f"{what} must be positive")
    return value


def _number_sequence(value, what: str) -> tuple[float, ...]:
    values = _sequence(value, what)
    return tuple(_number(item, f"{what}[{i}]") for i, item in enumerate(values))


def _number_tree(value, what: str):
    if isinstance(value, list):
        return [_number_tree(item, f"{what}[{i}]") for i, item in enumerate(value)]
    return _number(value, what)


def _choice(value, what: str, choices: set[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{what} must be one of {sorted(choices)}")
    return value


def _label(value, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{what} must be a non-empty string")
    return value


def _sequence(value, what: str) -> list:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise ValueError(f"{what} must be a list")
    return list(value)


def _object(value, what: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{what} must be an object")
    return value


def _keys(value, *, required: set[str], optional: set[str], what: str) -> None:
    value = _object(value, what)
    missing = required - set(value)
    if missing:
        raise ValueError(f"{what} is missing required key(s): {sorted(missing)}")
    unknown = set(value) - required - optional
    if unknown:
        raise ValueError(f"{what} has unsupported key(s): {sorted(unknown)}")


def _json_pointer(path: str) -> tuple[str, ...]:
    if not path.startswith("/") or path == "/":
        raise ValueError("fit parameter path must be a non-root JSON Pointer such as '/couplings/0'")
    parts = path[1:].split("/")
    if any(re.search(r"~(?![01])", part) for part in parts):
        raise ValueError(f"invalid JSON Pointer escape in fit parameter path {path!r}")
    return tuple(part.replace("~1", "/").replace("~0", "~") for part in parts)


def _pointer_child(value, segment: str, path: str):
    if isinstance(value, Mapping):
        if segment not in value:
            raise ValueError(f"fit parameter path {path!r} does not exist")
        return value[segment]
    if isinstance(value, list):
        if re.fullmatch(r"(?:0|[1-9][0-9]*)", segment) is None or int(segment) >= len(value):
            raise ValueError(f"fit parameter path {path!r} has an invalid array index")
        return value[int(segment)]
    raise ValueError(f"fit parameter path {path!r} traverses a scalar")


def _get_json_pointer(value, segments: tuple[str, ...], path: str):
    for segment in segments:
        value = _pointer_child(value, segment, path)
    return value


def _set_json_pointer(value, segments: tuple[str, ...], replacement, path: str) -> None:
    for segment in segments[:-1]:
        value = _pointer_child(value, segment, path)
    target = segments[-1]
    if isinstance(value, dict):
        if target not in value:
            raise ValueError(f"fit parameter path {path!r} does not exist")
        value[target] = replacement
    elif isinstance(value, list):
        if re.fullmatch(r"(?:0|[1-9][0-9]*)", target) is None or int(target) >= len(value):
            raise ValueError(f"fit parameter path {path!r} has an invalid array index")
        value[int(target)] = replacement
    else:
        raise ValueError(f"fit parameter path {path!r} traverses a scalar")
