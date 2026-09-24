"""Validated hadron, channel, frame, level, and spectrum data models."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from numbers import Integral

import numpy as np

__all__ = [
    "ExchangeRule",
    "Hadron",
    "Channel",
    "Ensemble",
    "Frame",
    "Level",
    "Spectrum",
    "TEMPORAL_LATTICE_UNITS",
]


TEMPORAL_LATTICE_UNITS = "temporal_lattice"


class ExchangeRule(StrEnum):
    """How the two hadrons of a channel are related under exchange."""

    DISTINGUISHABLE = "distinguishable"
    IDENTICAL = "identical"
    PARTICLE_ANTIPARTICLE = "particle_antiparticle"


def _as_int(value: object, name: str, *, minimum: int | None = None) -> int:
    """Return ``value`` as a plain int, rejecting bools and non-integers."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(
            f"{name} must be an integer, got {type(value).__name__}: {value!r}"
        )
    out = int(value)
    if minimum is not None and out < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {out}")
    return out


def _as_float(
    value: object, name: str, *, minimum: float | None = None, strict: bool = False
) -> float:
    """Return ``value`` as a finite float with an optional lower bound."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise ValueError(
            f"{name} must be a real number, got {type(value).__name__}: {value!r}"
        )
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if minimum is not None:
        if strict and out <= minimum:
            raise ValueError(f"{name} must be > {minimum}, got {out}")
        if not strict and out < minimum:
            raise ValueError(f"{name} must be >= {minimum}, got {out}")
    return out


def _as_label(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty str, got {value!r}")
    return value


def _as_spectrum_units(value: object, name: str) -> str:
    """Validate the one unit convention currently supported by spectra.

    Spectrum values are consumed directly by temporal-lattice kinematics.  A
    different label would require a scale and conversion policy that this
    model does not carry, so accepting it would only make the data look more
    usable without changing its numerical values.
    """
    if not isinstance(value, str):
        raise ValueError(
            f"{name} must be the supported unit label {TEMPORAL_LATTICE_UNITS!r}, "
            f"got {type(value).__name__}: {value!r}"
        )
    if value != TEMPORAL_LATTICE_UNITS:
        raise ValueError(
            f"{name} has unsupported units {value!r}; only "
            f"{TEMPORAL_LATTICE_UNITS!r} is supported"
        )
    return value


@dataclass(frozen=True)
class Hadron:
    """A stable hadron with exact half-integer spin encoded as ``2S``."""

    label: str
    twice_spin: int
    parity: int
    mass_at: float
    mass_unc_at: float = 0.0
    ensemble_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", _as_label(self.label, "Hadron.label"))
        object.__setattr__(
            self, "twice_spin", _as_int(self.twice_spin, "Hadron.twice_spin", minimum=0)
        )
        parity = _as_int(self.parity, "Hadron.parity")
        if parity not in (1, -1):
            raise ValueError(
                f"Hadron.parity must be +1 or -1 for {self.label!r}, got {parity}"
            )
        object.__setattr__(self, "parity", parity)
        object.__setattr__(
            self, "mass_at", _as_float(self.mass_at, "Hadron.mass_at", minimum=0.0, strict=True)
        )
        object.__setattr__(
            self,
            "mass_unc_at",
            _as_float(self.mass_unc_at, "Hadron.mass_unc_at", minimum=0.0),
        )
        if self.ensemble_id is not None:
            object.__setattr__(
                self, "ensemble_id", _as_label(self.ensemble_id, "Hadron.ensemble_id")
            )

    @property
    def twice_spin_is_even(self) -> bool:
        """True for integer spins (bosons), False for half-integer (fermions)."""
        return self.twice_spin % 2 == 0


@dataclass(frozen=True)
class Channel:
    """A two-hadron channel with its exchange rule and derived observables."""

    label: str
    hadron_a: Hadron
    hadron_b: Hadron
    exchange: ExchangeRule | str = ExchangeRule.DISTINGUISHABLE

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", _as_label(self.label, "Channel.label"))
        for name in ("hadron_a", "hadron_b"):
            if not isinstance(getattr(self, name), Hadron):
                raise ValueError(f"Channel.{name} must be a Hadron, got {getattr(self, name)!r}")
        try:
            exchange = ExchangeRule(self.exchange)
        except ValueError as exc:
            raise ValueError(
                f"unknown exchange rule {self.exchange!r} for channel {self.label!r}"
            ) from exc
        object.__setattr__(self, "exchange", exchange)
        if exchange is ExchangeRule.IDENTICAL:
            a, b = self.hadron_a, self.hadron_b
            if a.label != b.label or a.twice_spin != b.twice_spin:
                raise ValueError(
                    "identical channels require equal labels and twice_spin, got "
                    f"{a.label!r}/{a.twice_spin} and {b.label!r}/{b.twice_spin}"
                )

    @property
    def intrinsic_parity(self) -> int:
        """Product of the two intrinsic parities: ``pa * pb``."""
        return self.hadron_a.parity * self.hadron_b.parity

    @property
    def allowed_twice_S(self) -> tuple[int, ...]:
        """Total spin values ``2S`` allowed by the constituent spins."""
        two_sa = self.hadron_a.twice_spin
        two_sb = self.hadron_b.twice_spin
        return tuple(range(abs(two_sa - two_sb), two_sa + two_sb + 1, 2))

    def exchange_allowed(self, ell: int, twice_s: int) -> bool:
        """Whether ``(ell, twice_s)`` survives the particle-exchange constraint.

        For identical constituents the exchange phase of the coupled state is

            ``sign = (-1)**ell * (-1)**(S - twice_spin)``,  ``S = twice_s // 2``

        ``sign == +1`` is required for bosons (``twice_spin`` even) and
        ``sign == -1`` for fermions (``twice_spin`` odd).  The intrinsic spin
        factor ``(-1)**(-twice_spin)`` combines with the statistic requirement,
        so both cases reduce to the same explicit integer criterion

            ``(ell + S) % 2 == 0``  <=>  ``(2 * ell + twice_s) % 4 == 0``.

        ``distinguishable`` and ``particle_antiparticle`` impose no constraint
        (``particle_antiparticle`` is kept separate from ``identical`` because a
        ``D Dbar`` pair must not be treated as identical particles).
        """
        ell = _as_int(ell, "ell", minimum=0)
        twice_s = _as_int(twice_s, "twice_s", minimum=0)
        if self.exchange is not ExchangeRule.IDENTICAL:
            return True
        return (2 * ell + twice_s) % 4 == 0


@dataclass(frozen=True)
class Ensemble:
    """A gauge ensemble: anisotropy, its uncertainty and hadron mass covariance."""

    id: str
    xi: float
    xi_unc: float = 0.0
    mass_covariance: tuple[tuple[float, ...], ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _as_label(self.id, "Ensemble.id"))
        object.__setattr__(self, "xi", _as_float(self.xi, "Ensemble.xi", minimum=0.0, strict=True))
        object.__setattr__(
            self, "xi_unc", _as_float(self.xi_unc, "Ensemble.xi_unc", minimum=0.0)
        )
        if self.mass_covariance is not None:
            rows = tuple(tuple(float(v) for v in row) for row in self.mass_covariance)
            n = len(rows)
            if any(len(row) != n for row in rows):
                raise ValueError(
                    f"Ensemble.mass_covariance must be square for {self.id!r} (got {self.mass_covariance!r})"
                )
            for i in range(n):
                for j in range(n):
                    if not np.isfinite(rows[i][j]):
                        raise ValueError(
                            f"Ensemble.mass_covariance not finite for {self.id!r} at {(i, j)}"
                        )
                    if abs(rows[i][j] - rows[j][i]) > 1e-12 * max(1.0, abs(rows[i][j])):
                        raise ValueError(
                            f"Ensemble.mass_covariance not symmetric for {self.id!r} at {(i, j)}"
                        )
            object.__setattr__(self, "mass_covariance", rows)


@dataclass(frozen=True)
class Frame:
    """A moving frame: box size, anisotropy and integer boost ``d``."""

    id: str
    spatial_sites: int
    anisotropy: float
    d: tuple[int, int, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _as_label(self.id, "Frame.id"))
        object.__setattr__(
            self,
            "spatial_sites",
            _as_int(self.spatial_sites, "Frame.spatial_sites", minimum=1),
        )
        object.__setattr__(
            self,
            "anisotropy",
            _as_float(self.anisotropy, "Frame.anisotropy", minimum=0.0, strict=True),
        )
        d = tuple(self.d)
        if len(d) != 3:
            raise ValueError(f"Frame.d must have 3 entries for {self.id!r}, got {self.d!r}")
        object.__setattr__(
            self, "d", tuple(_as_int(v, "Frame.d", minimum=None) for v in d)
        )


@dataclass(frozen=True)
class Level:
    """One lattice energy level in a frame/irrep/row, with reference frame id."""

    id: str
    frame_id: str
    irrep: str
    row: int
    energy_at: float
    energy_unc_at: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _as_label(self.id, "Level.id"))
        object.__setattr__(self, "frame_id", _as_label(self.frame_id, "Level.frame_id"))
        object.__setattr__(self, "irrep", _as_label(self.irrep, "Level.irrep"))
        object.__setattr__(self, "row", _as_int(self.row, "Level.row", minimum=0))
        object.__setattr__(self, "energy_at", _as_float(self.energy_at, "Level.energy_at"))
        object.__setattr__(
            self, "energy_unc_at", _as_float(self.energy_unc_at, "Level.energy_unc_at", minimum=0.0)
        )


@dataclass(frozen=True, eq=False)
class Spectrum:
    """A full coupled-channel spectrum with an explicit covariance matrix.

    ``covariance`` is ordered like ``levels``.  Validation at construction:
    every ``Level.frame_id`` resolves, level ids are unique, and the covariance
    is square, symmetric and (by default) positive definite.
    """

    ensembles: tuple[Ensemble, ...]
    hadrons: tuple[Hadron, ...]
    channels: tuple[Channel, ...]
    frames: tuple[Frame, ...]
    levels: tuple[Level, ...]
    covariance: np.ndarray
    irrep_map: dict[str, dict[str, str]] = field(default_factory=dict)
    units: str = TEMPORAL_LATTICE_UNITS
    require_positive_definite: bool = field(default=True, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name, cls in (
            ("ensembles", Ensemble),
            ("hadrons", Hadron),
            ("channels", Channel),
            ("frames", Frame),
            ("levels", Level),
        ):
            values = tuple(getattr(self, name))
            for value in values:
                if not isinstance(value, cls):
                    raise ValueError(f"Spectrum.{name} entries must be {cls.__name__}, got {value!r}")
            object.__setattr__(self, name, values)
        if not isinstance(self.irrep_map, dict):
            raise ValueError(f"Spectrum.irrep_map must be a dict, got {self.irrep_map!r}")
        object.__setattr__(
            self,
            "irrep_map",
            {str(k): {str(a): str(b) for a, b in dict(v).items()} for k, v in self.irrep_map.items()},
        )
        object.__setattr__(self, "units", _as_spectrum_units(self.units, "Spectrum.units"))

        _check_unique((h.label for h in self.hadrons), "hadron label")
        _check_unique((c.label for c in self.channels), "channel label")
        _check_unique((f.id for f in self.frames), "frame id")
        _check_unique((e.id for e in self.ensembles), "ensemble id")
        _check_unique((lv.id for lv in self.levels), "level id")

        frame_ids = {f.id for f in self.frames}
        for level in self.levels:
            if level.frame_id not in frame_ids:
                raise ValueError(
                    f"level {level.id!r} references unknown frame {level.frame_id!r}"
                )

        covariance = np.asarray(self.covariance, dtype=float)
        n = len(self.levels)
        if covariance.shape != (n, n):
            raise ValueError(
                f"Spectrum.covariance shape {covariance.shape} must be {(n, n)} for {n} levels"
            )
        if not np.all(np.isfinite(covariance)):
            raise ValueError("Spectrum.covariance must be finite")
        if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-12):
            raise ValueError("Spectrum.covariance must be symmetric")
        if n and self.require_positive_definite:
            smallest = float(np.linalg.eigvalsh(covariance)[0])
            if smallest <= 0.0:
                raise ValueError(
                    f"Spectrum.covariance must be positive definite, min eigenvalue {smallest:.6g}"
                )
        object.__setattr__(self, "covariance", covariance)

    # -- lookups ---------------------------------------------------------
    def ensemble(self, ensemble_id: str) -> Ensemble:
        return _lookup(self.ensembles, ensemble_id, "ensemble", lambda e: e.id)

    def hadron(self, label: str) -> Hadron:
        return _lookup(self.hadrons, label, "hadron", lambda h: h.label)

    def channel(self, label: str) -> Channel:
        return _lookup(self.channels, label, "channel", lambda c: c.label)

    def frame(self, frame_id: str) -> Frame:
        return _lookup(self.frames, frame_id, "frame", lambda f: f.id)

    def index(self, level_id: str) -> int:
        for i, level in enumerate(self.levels):
            if level.id == level_id:
                return i
        raise KeyError(f"no level {level_id!r} in spectrum")

    @property
    def n_levels(self) -> int:
        return len(self.levels)

    def __len__(self) -> int:
        return len(self.levels)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Spectrum):
            return NotImplemented
        return bool(
            self.units == other.units
            and self.ensembles == other.ensembles
            and self.hadrons == other.hadrons
            and self.channels == other.channels
            and self.frames == other.frames
            and self.levels == other.levels
            and self.irrep_map == other.irrep_map
            and self.covariance.shape == other.covariance.shape
            and np.array_equal(self.covariance, other.covariance)
        )


def _check_unique(values, what: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"duplicate {what} {value!r}")
        seen.add(value)


def _lookup(items, key, what: str, getter):
    for item in items:
        if getter(item) == key:
            return item
    raise KeyError(f"no {what} {key!r} in spectrum")
