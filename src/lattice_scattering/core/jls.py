"""JLS sector enumeration and block layout in channel-major order."""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

from .model import Channel

__all__ = ["Sector", "enumerate_sectors", "jls_blocks", "jls_shape"]


def _as_int(value: object, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}: {value!r}")
    out = int(value)
    if minimum is not None and out < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {out}")
    return out


@dataclass(frozen=True)
class Sector:
    """A ``(channel_index, ell, twice_S)`` orbital/partial-wave sector."""

    channel_index: int
    ell: int
    twice_S: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "channel_index", _as_int(self.channel_index, "Sector.channel_index", minimum=0)
        )
        object.__setattr__(self, "ell", _as_int(self.ell, "Sector.ell", minimum=0))
        object.__setattr__(
            self, "twice_S", _as_int(self.twice_S, "Sector.twice_S", minimum=0)
        )


def enumerate_sectors(
    channels,
    ell_max: int,
    *,
    exchange: bool = True,
    parity_target: int | None = None,
    ell_min: int = 0,
) -> tuple[Sector, ...]:
    """Enumerate allowed ``(channel, ell, S)`` sectors, channel-major.

    Ordering is ``channel index`` -> ``ell`` ascending -> ``twice_S`` ascending.

    Parameters
    ----------
    channels:
        Sequence of :class:`~lattice_scattering.core.model.Channel`.
    ell_max:
        Largest orbital angular momentum (inclusive).
    exchange:
        Apply each channel's exchange constraint when True.
    parity_target:
        If given, keep only sectors with
        ``channel.intrinsic_parity * (-1)**ell == parity_target``.
    ell_min:
        Smallest orbital angular momentum (inclusive).
    """
    ell_max = _as_int(ell_max, "ell_max", minimum=0)
    ell_min = _as_int(ell_min, "ell_min", minimum=0)
    if ell_min > ell_max:
        raise ValueError(f"ell_min {ell_min} exceeds ell_max {ell_max}")
    if parity_target is not None:
        parity_target = _as_int(parity_target, "parity_target")
        if parity_target not in (1, -1):
            raise ValueError(f"parity_target must be +1 or -1, got {parity_target}")

    sectors: list[Sector] = []
    for index, channel in enumerate(channels):
        if not isinstance(channel, Channel):
            raise ValueError(f"channel {index} must be a Channel, got {channel!r}")
        for ell in range(ell_min, ell_max + 1):
            if parity_target is not None and channel.intrinsic_parity * (-1) ** ell != parity_target:
                continue
            for twice_s in channel.allowed_twice_S:
                if exchange and not channel.exchange_allowed(ell, twice_s):
                    continue
                sectors.append(Sector(index, ell, twice_s))
    return tuple(sectors)


def _participates(ell: int, twice_s: int, twice_j: int) -> bool:
    """JLS triangular criterion with parity (same parity as ``2ell + 2S``)."""
    lower = abs(2 * ell - twice_s)
    upper = 2 * ell + twice_s
    if twice_j < lower or twice_j > upper:
        return False
    return (twice_j - upper) % 2 == 0


def jls_blocks(sectors) -> dict[int, tuple[Sector, ...]]:
    """Map ``twice_J`` -> participating sectors (input order preserved).

    Keys cover ``0 .. 2*ell_max + max(twice_S)`` of the given sectors; blocks
    without members map to an empty tuple.  Empty input yields ``{}``.
    """
    sectors = tuple(sectors)
    for sector in sectors:
        if not isinstance(sector, Sector):
            raise ValueError(f"sectors entries must be Sector, got {sector!r}")
    if not sectors:
        return {}
    twice_j_max = 2 * max(s.ell for s in sectors) + max(s.twice_S for s in sectors)
    blocks: dict[int, tuple[Sector, ...]] = {}
    for twice_j in range(twice_j_max + 1):
        blocks[twice_j] = tuple(
            s for s in sectors if _participates(s.ell, s.twice_S, twice_j)
        )
    return blocks


def jls_shape(sectors, twice_J: int) -> int:
    """Number of sectors contributing to the ``twice_J`` block."""
    twice_J = _as_int(twice_J, "twice_J", minimum=0)
    return len(jls_blocks(sectors).get(twice_J, ()))
