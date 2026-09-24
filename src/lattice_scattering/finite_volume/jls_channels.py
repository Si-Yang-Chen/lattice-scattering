"""Channel layouts and exact selective-total-J projections for the JLS kernel.

The public JLS API historically used one ``(ell, twice_S)`` list and one
intrinsic parity for every channel.  This module contains the small amount of
layout algebra needed by the extended path:

* every channel may carry its own sector list and intrinsic parity;
* a ``twice_J`` block contains only the requested active incidences;
* the finite-volume carrier remains channel-major, with each channel's local
  sectors in caller order; and
* the selected-J row space is obtained by projecting a genuine irrep row basis
  with the active LS projector and orthonormalising its range.

The module deliberately has no dependency on :mod:`jls_matrix`, so the root
scanner and the assembly wrapper can adopt these helpers without introducing a
cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Mapping

import numpy as np

from ..core.jls import Sector
from ..symmetry.angular_momentum import coupled_j_basis
from ..symmetry.groups import subduce
from .box import _dimension, _sectors

__all__ = [
    "ChannelLayout",
    "assemble_active_inverse",
    "channel_row_basis",
    "normalize_channel_layout",
    "normalise_channel_layout",
]


def _integer(value, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}")
    return value


def _parity(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) not in (-1, 1):
        raise ValueError(f"{name} must be +1 or -1, got {value!r}")
    return int(value)


def _sector_pair(value, channel: int) -> tuple[int, int]:
    """Convert one pair or one channel-labelled ``Sector`` to ``(ell, 2S)``."""
    if isinstance(value, Sector):
        if value.channel_index != channel:
            raise ValueError(
                f"channel {channel} received Sector for channel {value.channel_index}"
            )
        return int(value.ell), int(value.twice_S)
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(
            "channel_sectors entries must be (ell, twice_S) pairs or Sector objects, "
            f"got {value!r}"
        )
    return _integer(value[0], "ell"), _integer(value[1], "twice_S")


def _channel_pairs(raw, channel: int, budget: int) -> tuple[tuple[int, int], ...]:
    if isinstance(raw, (str, bytes)):
        raise ValueError(f"channel {channel} sectors must be a sequence of sectors")
    try:
        pairs = tuple(_sector_pair(item, channel) for item in raw)
    except TypeError as exc:
        raise ValueError(f"channel {channel} sectors must be a sequence of sectors") from exc
    # Reuse the finite-volume sector validator for degree, spin, duplicate and
    # per-channel dimension checks.  It also returns canonical integer pairs.
    return _sectors(pairs, budget)[0]


def _channel_sequence(channel_sectors, channels: int | None):
    """Return ``(number_of_channels, per-channel raw sequences)``."""
    if isinstance(channel_sectors, Mapping):
        keys = []
        for key in channel_sectors:
            keys.append(_integer(key, "channel index"))
        inferred = (max(keys) + 1) if keys else 0
        if channels is None:
            channels = inferred
        if channels != inferred or set(keys) != set(range(channels)):
            raise ValueError(
                "channel_sectors mapping must contain exactly channel indices "
                f"0..{channels - 1}"
            )
        return int(channels), tuple(channel_sectors[index] for index in range(channels))

    if isinstance(channel_sectors, (str, bytes)):
        raise ValueError("channel_sectors must be a sequence of per-channel sector lists")
    try:
        raw = tuple(channel_sectors)
    except TypeError as exc:
        raise ValueError("channel_sectors must be a sequence of per-channel sector lists") from exc
    if channels is None:
        channels = len(raw)
    if len(raw) != channels:
        raise ValueError(
            f"channel_sectors must contain one sector list per channel ({channels}), got {len(raw)}"
        )
    return int(channels), raw


def _allowed_local(pairs: tuple[tuple[int, int], ...]) -> dict[int, tuple[int, ...]]:
    allowed: dict[int, list[int]] = {}
    for index, (ell, twice_s) in enumerate(pairs):
        for twice_j in range(abs(2 * ell - twice_s), 2 * ell + twice_s + 1, 2):
            allowed.setdefault(twice_j, []).append(index)
    return {twice_j: tuple(indices) for twice_j, indices in allowed.items()}


@dataclass(frozen=True)
class ChannelLayout:
    """Canonical channel-major carrier and active-incidence layout."""

    channel_pairs: tuple[tuple[tuple[int, int], ...], ...]
    channel_sizes: tuple[tuple[int, ...], ...]
    channel_offsets: tuple[int, ...]
    local_offsets: tuple[tuple[int, ...], ...]
    intrinsic_parities: tuple[int, ...]
    allowed: dict[int, tuple[tuple[int, int], ...]]
    active: dict[int, tuple[tuple[int, int], ...]]
    total_dimension: int

    @property
    def channels(self) -> int:
        return len(self.channel_pairs)

    def channel_dimension(self, channel: int) -> int:
        return int(sum(self.channel_sizes[channel]))

    def sector_slice(self, channel: int, sector_index: int) -> slice:
        start = self.channel_offsets[channel] + self.local_offsets[channel][sector_index]
        stop = self.channel_offsets[channel] + self.local_offsets[channel][sector_index + 1]
        return slice(start, stop)

    def incidence_label(self, incidence: tuple[int, int]) -> tuple[int, int, int]:
        channel, sector_index = incidence
        ell, twice_s = self.channel_pairs[channel][sector_index]
        return channel, ell, twice_s

    def labels(self, twice_j: int) -> tuple[tuple[int, int, int], ...]:
        return tuple(self.incidence_label(item) for item in self.active[twice_j])

    @property
    def active_labels(self) -> dict[int, tuple[tuple[int, int, int], ...]]:
        """Canonical JSON-friendly active incidence labels by ``twice_J``."""
        return {twice_j: self.labels(twice_j) for twice_j in self.active}


def _parse_incidence(value) -> tuple[int, int, int]:
    if isinstance(value, Sector):
        return int(value.channel_index), int(value.ell), int(value.twice_S)
    if isinstance(value, Mapping):
        names = ("channel", "ell", "twice_S")
        if any(name not in value for name in names):
            raise ValueError(
                "selected_j_sectors incidence mappings require channel, ell and twice_S"
            )
        value = tuple(value[name] for name in names)
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(
            "selected_j_sectors entries must be (channel, ell, twice_S) triples"
        )
    return (
        _integer(value[0], "selected incidence channel"),
        _integer(value[1], "selected incidence ell"),
        _integer(value[2], "selected incidence twice_S"),
    )


def _selected_active(
    selected_j_sectors,
    *,
    channel_pairs: tuple[tuple[tuple[int, int], ...], ...],
    allowed: dict[int, tuple[tuple[int, int], ...]],
) -> dict[int, tuple[tuple[int, int], ...]]:
    """Validate and canonicalise the selected incidence mapping."""
    if selected_j_sectors is None:
        return {twice_j: tuple(entries) for twice_j, entries in allowed.items()}
    if not isinstance(selected_j_sectors, Mapping):
        raise ValueError("selected_j_sectors must be a mapping {twice_J: incidence list}")
    if not selected_j_sectors:
        raise ValueError(
            "selected_j_sectors must retain at least one J; omit the option for full J content"
        )

    pair_indices = [
        {pair: index for index, pair in enumerate(pairs)} for pairs in channel_pairs
    ]
    active: dict[int, tuple[tuple[int, int], ...]] = {}
    for raw_j, raw_entries in selected_j_sectors.items():
        twice_j = _integer(raw_j, "selected twice_J")
        if twice_j not in allowed:
            raise ValueError(
                f"selected twice_J={twice_j} is unknown; allowed values are {sorted(allowed)}"
            )
        if isinstance(raw_entries, (str, bytes)):
            raise ValueError(f"selected_j_sectors[{twice_j}] must be an incidence sequence")
        try:
            entries = tuple(_parse_incidence(value) for value in raw_entries)
        except TypeError as exc:
            raise ValueError(
                f"selected_j_sectors[{twice_j}] must be an incidence sequence"
            ) from exc
        if not entries:
            raise ValueError(
                f"selected_j_sectors[{twice_j}] cannot be empty; omit the key to omit that J"
            )
        requested: set[tuple[int, int]] = set()
        for channel, ell, twice_s in entries:
            if channel >= len(channel_pairs):
                raise ValueError(
                    f"selected incidence {(channel, ell, twice_s)} has unknown channel"
                )
            sector_index = pair_indices[channel].get((ell, twice_s))
            if sector_index is None:
                raise ValueError(
                    f"selected incidence {(channel, ell, twice_s)} is absent from channel sectors"
                )
            incidence = (channel, sector_index)
            if incidence in requested:
                raise ValueError(
                    f"duplicate selected incidence {(channel, ell, twice_s)} for twice_J={twice_j}"
                )
            if incidence not in allowed[twice_j]:
                raise ValueError(
                    f"selected incidence {(channel, ell, twice_s)} does not participate in "
                    f"twice_J={twice_j}"
                )
            requested.add(incidence)
        # Reorder user entries into the contract's channel-major sector order.
        active[twice_j] = tuple(item for item in allowed[twice_j] if item in requested)
    return {twice_j: entries for twice_j, entries in active.items() if entries}


def normalise_channel_layout(
    sectors=None,
    *,
    channels: int | None = None,
    channel_sectors=None,
    channel_intrinsic_parities=None,
    intrinsic_parity: int = 1,
    selected_j_sectors=None,
    max_dimension: int = 256,
) -> ChannelLayout:
    """Validate and lower common or heterogeneous channel inputs.

    ``sectors`` is the historical common list.  When ``channel_sectors`` is
    present it replaces it and ``sectors`` must be ``None``.  The returned
    ``active`` entries are pairs of
    ``(channel_index, local_sector_index)`` in channel-major order.
    """
    budget = _dimension(max_dimension)
    if channels is not None:
        channels = _integer(channels, "channels", minimum=1)

    if channel_sectors is None:
        if sectors is None:
            raise ValueError("sectors is required when channel_sectors is not supplied")
        common = _sectors(sectors, budget)[0]
        if channels is None:
            channels = 1
        raw_channels = tuple(common for _ in range(channels))
    else:
        if sectors is not None:
            raise ValueError(
                "pass either the common sectors argument or channel_sectors, not both"
            )
        channels, raw_channels = _channel_sequence(channel_sectors, channels)
        if channels < 1:
            raise ValueError("at least one channel is required")

    channel_pairs = tuple(
        _channel_pairs(raw, channel, budget) for channel, raw in enumerate(raw_channels)
    )
    if not channel_pairs or any(not pairs for pairs in channel_pairs):
        raise ValueError("at least one sector is required for every channel")

    if channel_intrinsic_parities is None:
        default_parity = _parity(intrinsic_parity, "intrinsic_parity")
        parities = (default_parity,) * channels
    else:
        if intrinsic_parity != 1:
            raise ValueError(
                "intrinsic_parity cannot be combined with channel_intrinsic_parities; "
                "put all channel parities in channel_intrinsic_parities"
            )
        try:
            raw_parities = tuple(channel_intrinsic_parities)
        except TypeError as exc:
            raise ValueError("channel_intrinsic_parities must be one parity per channel") from exc
        if len(raw_parities) != channels:
            raise ValueError(
                "channel_intrinsic_parities must contain one value per channel "
                f"({channels}), got {len(raw_parities)}"
            )
        parities = tuple(
            _parity(value, f"channel_intrinsic_parities[{channel}]")
            for channel, value in enumerate(raw_parities)
        )

    channel_sizes = tuple(
        tuple((2 * ell + 1) * (twice_s + 1) for ell, twice_s in pairs)
        for pairs in channel_pairs
    )
    total_dimension = int(sum(sum(sizes) for sizes in channel_sizes))
    if total_dimension > budget:
        raise ValueError(
            f"channel-major mixed basis of dimension {total_dimension} exceeds the budget {budget}"
        )
    local_offsets = tuple(
        tuple(np.cumsum([0] + list(sizes)).astype(int).tolist()) for sizes in channel_sizes
    )
    channel_offsets = tuple(
        np.cumsum([0] + [sum(sizes) for sizes in channel_sizes]).astype(int).tolist()
    )

    allowed_by_channel = tuple(_allowed_local(pairs) for pairs in channel_pairs)
    all_j = sorted({twice_j for allowed in allowed_by_channel for twice_j in allowed})
    allowed: dict[int, tuple[tuple[int, int], ...]] = {
        twice_j: tuple(
            (channel, sector_index)
            for channel, allowed_local in enumerate(allowed_by_channel)
            for sector_index in allowed_local.get(twice_j, ())
        )
        for twice_j in all_j
    }
    active = _selected_active(
        selected_j_sectors,
        channel_pairs=channel_pairs,
        allowed=allowed,
    )
    return ChannelLayout(
        channel_pairs=channel_pairs,
        channel_sizes=channel_sizes,
        channel_offsets=channel_offsets,
        local_offsets=local_offsets,
        intrinsic_parities=parities,
        allowed=allowed,
        active=active,
        total_dimension=total_dimension,
    )


# Both spellings are kept as a small convenience for callers; the public
# package historically uses British spelling in ``normalise`` helpers.
normalize_channel_layout = normalise_channel_layout


def parity_allowed(layout: ChannelLayout, left: tuple[int, int], right: tuple[int, int]) -> bool:
    """Whether two LS incidences have equal physical total parity."""
    left_channel, left_sector = left
    right_channel, right_sector = right
    left_ell = layout.channel_pairs[left_channel][left_sector][0]
    right_ell = layout.channel_pairs[right_channel][right_sector][0]
    return (
        layout.intrinsic_parities[left_channel] * (-1) ** left_ell
        == layout.intrinsic_parities[right_channel] * (-1) ** right_ell
    )


def assemble_active_inverse(
    layout: ChannelLayout,
    validated,
    *,
    scale: float,
    weighting: str,
    per_channel=None,
    max_dimension: int = 256,
) -> np.ndarray:
    """Assemble ``R`` from already validated active ``J`` blocks."""
    budget = _dimension(max_dimension)
    result = np.zeros((layout.total_dimension, layout.total_dimension), dtype=complex)
    if weighting == "threshold":
        if per_channel is None:
            raise ValueError("threshold weighting requires one 2k value per channel")
        per_channel = np.asarray(per_channel)
        if per_channel.shape != (layout.channels,):
            raise ValueError(
                f"one k^2-derived value per channel required ({layout.channels}), got {per_channel.shape}"
            )

    for twice_j, (entries, value) in validated.items():
        bases = {
            incidence: coupled_j_basis(
                2 * layout.channel_pairs[incidence[0]][incidence[1]][0],
                layout.channel_pairs[incidence[0]][incidence[1]][1],
                twice_j,
                max_dimension=budget,
            )
            for incidence in entries
        }
        for a, left_incidence in enumerate(entries):
            left_channel, left_sector = left_incidence
            left = layout.sector_slice(left_channel, left_sector)
            left_ell = layout.channel_pairs[left_channel][left_sector][0]
            for b, right_incidence in enumerate(entries):
                right_channel, right_sector = right_incidence
                right = layout.sector_slice(right_channel, right_sector)
                right_ell = layout.channel_pairs[right_channel][right_sector][0]
                if weighting == "threshold":
                    factor = per_channel[left_channel] ** (-left_ell)
                    factor = factor * per_channel[right_channel] ** (-right_ell)
                else:
                    factor = scale ** (left_ell + right_ell + 1)
                result[left, right] += (
                    factor * value[a, b] * (bases[left_incidence] @ bases[right_incidence].conj().T)
                )
    if not np.all(np.isfinite(result)):
        raise ArithmeticError("nonfinite assembled inverse in channel-major JLS carrier")
    return result


def _block_diagonal(blocks, shape: tuple[int, int]) -> np.ndarray:
    result = np.zeros(shape, dtype=complex)
    row_offset = column_offset = 0
    for block in blocks:
        rows, columns = block.shape
        result[row_offset : row_offset + rows, column_offset : column_offset + columns] = block
        row_offset += rows
        column_offset += columns
    return result


def channel_row_basis(layout: ChannelLayout, group, irrep, row: int, *, max_dimension: int = 256):
    """Return ``(W, V, P)`` for the requested row and active J content.

    ``V`` is the block-diagonal per-channel row basis, ``P`` is the exact
    active LS projector, and columns of ``W`` are an orthonormal basis for
    ``range(P @ V)``.  A row absent from every channel or removed entirely by
    the selected-J projection raises a descriptive ``ValueError``.
    """
    budget = _dimension(max_dimension)
    if isinstance(row, bool) or not isinstance(row, Integral):
        raise ValueError(f"row must be an integer, got {row!r}")
    row = int(row)
    try:
        target = group.irrep(irrep)
    except (AttributeError, KeyError) as exc:
        raise ValueError(f"unknown irrep {irrep!r} for group {getattr(group, 'name', group)!r}") from exc
    if not 0 <= row < target.dimension:
        raise ValueError(
            f"invalid row {row!r} for irrep {irrep!r} of dimension {target.dimension} "
            f"in group {getattr(group, 'name', group)!r}"
        )

    row_blocks = []
    for channel, pairs in enumerate(layout.channel_pairs):
        result = subduce(
            [Sector(channel, ell, twice_s) for ell, twice_s in pairs],
            group,
            irrep,
            intrinsic_parity=layout.intrinsic_parities[channel],
            max_dimension=budget,
        )
        row_blocks.append(result["row_bases"][row])
    multiplicity = sum(block.shape[1] for block in row_blocks)
    if multiplicity == 0:
        raise ValueError(
            f"irrep {irrep!r} is absent from all channel sectors after subduction"
        )
    V = _block_diagonal(row_blocks, (layout.total_dimension, multiplicity))

    P = np.zeros((layout.total_dimension, layout.total_dimension), dtype=complex)
    for twice_j, entries in layout.active.items():
        for channel, sector_index in entries:
            ell, twice_s = layout.channel_pairs[channel][sector_index]
            basis = coupled_j_basis(
                2 * ell, twice_s, twice_j, max_dimension=budget
            )
            block = layout.sector_slice(channel, sector_index)
            P[block, block] += basis @ basis.conj().T
    P = 0.5 * (P + P.conj().T)

    projected = P @ V
    if projected.shape[1] == 0:
        raise ValueError(
            f"irrep {irrep!r} is absent after the selected-J projection"
        )
    left, singular, _ = np.linalg.svd(projected, full_matrices=False)
    if singular.size == 0:
        raise ValueError(
            f"irrep {irrep!r} is absent after the selected-J projection"
        )
    if np.any(np.minimum(np.abs(singular), np.abs(singular - 1.0)) > 1e-8):
        raise ArithmeticError(
            f"selected-J projector does not preserve irrep {irrep!r} row space; "
            f"singular values were {singular.tolist()}"
        )
    tolerance = 1e-10 * max(1.0, float(singular[0]))
    rank = int(np.count_nonzero(singular > tolerance))
    if rank == 0:
        raise ValueError(
            f"irrep {irrep!r} is absent after the selected-J projection"
        )
    W = left[:, :rank]
    return W, V, P
