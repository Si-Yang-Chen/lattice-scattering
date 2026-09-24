"""Adapters from single-channel K models to explicit JLS partial-wave blocks.

The 2102.04973 convention is

``t_l^-1 = (2 k)^(-2 l) K_l^-1 + I(s)``

with ``Im I = -2 k / E``.  The JLS ``scale`` callback must therefore return

``R_l = E / 2**(2*l + 1) * K_l^-1 + E/2 * (k**2)**l * Re I``.

This module deliberately consumes ``K^-1`` from the existing amplitude model and
does not call ``effective_inverse_k``: that helper already adds Chew-Mandelstam
phase space and would double-count it here.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral

import numpy as np

from ..finite_volume.jls_channels import normalise_channel_layout
from .phase_space import cm_real_axis

__all__ = ["PartialWaveJMatrixAdapter"]


def _real_scalar(value, name: str) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or np.iscomplexobj(value)
        or np.ndim(value) != 0
    ):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


def _mass_pairs(channel_masses) -> tuple[tuple[float, float], ...]:
    try:
        complex_masses = np.iscomplexobj(channel_masses)
    except (TypeError, ValueError):
        complex_masses = False
    if isinstance(channel_masses, (str, bytes)) or complex_masses:
        raise ValueError("channel_masses must contain positive finite (mass1, mass2) pairs")
    try:
        raw = tuple(channel_masses)
    except TypeError as exc:
        raise ValueError(
            "channel_masses must contain positive finite (mass1, mass2) pairs"
        ) from exc
    if not raw:
        raise ValueError("at least one channel mass pair is required")
    result = []
    for channel, pair in enumerate(raw):
        if isinstance(pair, (str, bytes)):
            raise ValueError(f"channel_masses[{channel}] must be a mass pair")
        try:
            values = tuple(pair)
        except TypeError as exc:
            raise ValueError(f"channel_masses[{channel}] must be a mass pair") from exc
        if len(values) != 2:
            raise ValueError(f"channel_masses[{channel}] must contain exactly two masses")
        masses = tuple(_real_scalar(value, f"channel_masses[{channel}]") for value in values)
        if any(mass <= 0.0 for mass in masses):
            raise ValueError("channel masses must be positive")
        result.append(masses)
    return tuple(result)


def _incidence_key(value) -> tuple[int, int, int, int]:
    if not isinstance(value, tuple) or len(value) != 4:
        raise ValueError(
            "model keys must be (channel, ell, twice_S, twice_J) incidence tuples"
        )
    names = ("channel", "ell", "twice_S", "twice_J")
    result = []
    for item, name in zip(value, names):
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, Integral):
            raise ValueError(f"incidence {name} must be a non-negative integer")
        integer = int(item)
        if integer < 0:
            raise ValueError(f"incidence {name} must be a non-negative integer")
        result.append(integer)
    return tuple(result)


def _inverse_scalar(model, s: float, incidence) -> float:
    inverse_method = getattr(model, "inverse", None)
    if not callable(inverse_method):
        raise ValueError(f"model at incidence {incidence} must provide inverse(s)")
    try:
        value = np.asarray(inverse_method(s))
        if value.shape != (1, 1):
            raise ValueError(
                f"K model at incidence {incidence} must return an exact (1, 1) inverse; "
                f"got {value.shape}"
            )
        if not np.issubdtype(value.dtype, np.number) or not np.all(np.isfinite(value)):
            raise ValueError(f"K inverse at incidence {incidence} must be finite and numeric")
        scalar = complex(value[0, 0])
    except (TypeError, OverflowError) as exc:
        raise ValueError(
            f"K inverse at incidence {incidence} must be a finite (1, 1) value"
        ) from exc
    if abs(scalar.imag) > 1e-12 * max(1.0, abs(scalar)):
        raise ValueError(
            f"K inverse at incidence {incidence} must be real for real s; "
            f"got imaginary part {scalar.imag!r}"
        )
    return float(scalar.real)


class PartialWaveJMatrixAdapter:
    """Turn single-channel K models into scale-weighted JLS blocks.

    ``models`` maps each active ``(channel, ell, twice_S, twice_J)`` incidence to
    an existing single-channel model implementing ``inverse(s)`` with shape
    ``(1, 1)``.  ``channel_sectors`` uses the same per-channel sector layout as
    :func:`lattice_scattering.finite_volume.row_matrix`.  If
    ``selected_j_sectors`` is omitted, every triangle-allowed incidence is
    active; otherwise the model keys must match that explicit selection exactly.

    For ``phase_space="chew-mandelstam"``, ``subtraction_points`` is required
    and maps every active incidence to either ``"threshold"`` or
    ``("k-pole", s_at2)``.  The latter explicitly declares the source's
    K-pole subtraction point; the adapter does not guess it from model
    parameter names.  It evaluates ``Re[I(s) - I(s_sub)]`` through the
    package's :func:`cm_real_axis`.
    ``phase_space="simple"`` uses ``Re I = 0`` and rejects subtraction points.

    The returned ``blocks(s)`` mapping is scale-weighted and diagonal: a
    single-channel K model has no inter-incidence mixing parameters.  Pass
    ``adapter.channel_sectors`` and ``adapter.selected_j_sectors`` to the JLS
    kernel alongside the adapter.  This class does not choose an irrep or row.
    """

    weighting = "scale"

    def __init__(
        self,
        models,
        *,
        channel_masses,
        channel_sectors,
        phase_space: str = "simple",
        subtraction_points=None,
        selected_j_sectors=None,
    ) -> None:
        masses = _mass_pairs(channel_masses)
        if phase_space not in ("simple", "chew-mandelstam"):
            raise ValueError("phase_space must be 'simple' or 'chew-mandelstam'")
        if not isinstance(models, Mapping) or not models:
            raise ValueError("models must be a non-empty incidence-to-K-model mapping")

        layout = normalise_channel_layout(
            None,
            channels=len(masses),
            channel_sectors=channel_sectors,
            selected_j_sectors=selected_j_sectors,
        )

        normalised_models = {}
        for raw_incidence, model in models.items():
            incidence = _incidence_key(raw_incidence)
            if incidence in normalised_models:
                raise ValueError(f"duplicate model incidence {incidence}")
            if not callable(getattr(model, "inverse", None)):
                raise ValueError(f"model at incidence {incidence} must provide inverse(s)")
            normalised_models[incidence] = model

        active = {
            (channel, *layout.channel_pairs[channel][sector_index], twice_j)
            for twice_j, incidences in layout.active.items()
            for channel, sector_index in incidences
        }
        supplied = set(normalised_models)
        if supplied != active:
            missing = sorted(active - supplied)
            extra = sorted(supplied - active)
            raise ValueError(
                "models must cover exactly the active JLS incidences; "
                f"missing={missing}, extra={extra}"
            )

        if phase_space == "simple":
            if subtraction_points is not None:
                raise ValueError("simple phase space does not accept subtraction_points")
            normalised_subtractions = {}
        else:
            if not isinstance(subtraction_points, Mapping):
                raise ValueError(
                    "chew-mandelstam phase space requires one subtraction point per active incidence"
                )
            normalised_subtractions = {}
            for raw_incidence, point in subtraction_points.items():
                incidence = _incidence_key(raw_incidence)
                if incidence not in active:
                    raise ValueError(
                        f"subtraction point supplied for inactive incidence {incidence}"
                    )
                if isinstance(point, str):
                    if point != "threshold":
                        raise ValueError(
                            "subtraction points must be 'threshold' or ('k-pole', s_at2)"
                        )
                else:
                    if (
                        not isinstance(point, tuple)
                        or len(point) != 2
                        or point[0] != "k-pole"
                    ):
                        raise ValueError(
                            "subtraction points must be 'threshold' or ('k-pole', s_at2)"
                        )
                    s_sub = _real_scalar(
                        point[1], f"subtraction_points[{incidence}]['k-pole']"
                    )
                    channel = incidence[0]
                    m1, m2 = masses[channel]
                    if s_sub <= (m1 - m2) ** 2:
                        raise ValueError(
                            f"subtraction point at incidence {incidence} must lie above "
                            "the channel pseudothreshold"
                        )
                    point = s_sub
                normalised_subtractions[incidence] = point
            if set(normalised_subtractions) != active:
                missing = sorted(active - set(normalised_subtractions))
                raise ValueError(
                    "chew-mandelstam phase space requires one subtraction point per active "
                    f"incidence; missing={missing}"
                )

        self.channel_masses = masses
        self.channel_sectors = layout.channel_pairs
        self.phase_space = phase_space
        self._layout = layout
        self._models = normalised_models
        self._subtraction_points = normalised_subtractions

    @property
    def selected_j_sectors(self) -> dict[int, tuple[tuple[int, int, int], ...]]:
        """The exact active incidence map to pass to a JLS forward call."""
        return {
            twice_j: tuple(
                (channel, *self._layout.channel_pairs[channel][sector_index])
                for channel, sector_index in incidences
            )
            for twice_j, incidences in self._layout.active.items()
        }

    def _real_cm(self, s: float, incidence: tuple[int, int, int, int]) -> float:
        point = self._subtraction_points[incidence]
        channel = incidence[0]
        m1, m2 = self.channel_masses[channel]
        s_sub = (m1 + m2) ** 2 if point == "threshold" else float(point)
        offset = -float(np.real(cm_real_axis(s_sub, m1, m2, subtraction=0.0)))
        value = cm_real_axis(s, m1, m2, subtraction=offset)
        return float(np.real(value))

    def blocks(self, s_at2: float) -> dict[int, np.ndarray]:
        """Return ``{twice_J: R^J(s_at2)}`` in active channel-major order."""
        s = _real_scalar(s_at2, "s_at2")
        if s <= 0.0:
            raise ValueError("s_at2 must be positive")
        energy = float(np.sqrt(s))
        result = {}
        for twice_j, incidences in self._layout.active.items():
            block = np.zeros((len(incidences), len(incidences)), dtype=float)
            for index, (channel, sector_index) in enumerate(incidences):
                ell, twice_s = self._layout.channel_pairs[channel][sector_index]
                incidence = (channel, ell, twice_s, twice_j)
                m1, m2 = self.channel_masses[channel]
                pseudothreshold = (m1 - m2) ** 2
                if s <= pseudothreshold:
                    raise ValueError(
                        f"s_at2 must be above channel {channel}'s pseudothreshold"
                    )
                k_squared = (s - (m1 + m2) ** 2) * (s - pseudothreshold) / (4.0 * s)
                if not np.isfinite(k_squared):
                    raise ValueError(f"non-finite k^2 in channel {channel}")
                inverse_k = _inverse_scalar(self._models[incidence], s, incidence)
                real_i = (
                    self._real_cm(s, incidence)
                    if self.phase_space == "chew-mandelstam"
                    else 0.0
                )
                value = energy / (2.0 ** (2 * ell + 1)) * inverse_k
                value += energy / 2.0 * k_squared**ell * real_i
                if not np.isfinite(value):
                    raise ValueError(f"non-finite reduced inverse at incidence {incidence}")
                block[index, index] = value
            result[twice_j] = block
        return result
