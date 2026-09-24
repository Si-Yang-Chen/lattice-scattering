"""The channel-basis and JLS kernels must agree, frame by frame.

This is the gate for having a single kernel.  The old end-to-end test compared against a
stored spectrum, which couples the assertion to whatever the scan produced when that
spectrum was generated -- so it could not tell a fixed kernel from a regression.  This test
compares the two kernels against each other on the same frame, which is what "the
channel-basis condition is a special case of the JLS one" actually means, and it costs
seconds rather than tens of minutes.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.slow

from lattice_scattering.kinematics import LatticeFrame, two_body_point
from lattice_scattering.amplitudes.models import ConstantK
from lattice_scattering.amplitudes.phase_space import effective_inverse_k
from lattice_scattering.finite_volume import Q2_MAX, RootScanError, coupled_s_roots
from lattice_scattering.finite_volume.jls_matrix import (
    WEIGHTING_THRESHOLD,
    quantization_roots,
)
from lattice_scattering.symmetry.groups import double_cover, little_group

SAMPLES = 200


def _equal_mass_energy_at_q2(frame, mass, q2):
    """Lab energy whose equal-mass CM momentum has the requested dimensionless q^2."""
    momentum_squared = sum(component**2 for component in frame.momentum_at)
    k_squared_at2 = q2 * (2.0 * np.pi / frame.length_at) ** 2
    return float(np.sqrt(momentum_squared + 4.0 * (mass**2 + k_squared_at2)))


def _jls_roots(frame, masses, k_matrix, window, phase_space, subtractions):
    """The same condition, expressed as JLS sectors and handed to the JLS kernel."""

    def reduced_inverse(s_at2):
        inverse = effective_inverse_k(
            k_matrix, masses, phase_space, s_at2, subtractions=subtractions
        )
        return (np.sqrt(s_at2) * frame.length_at / (4.0 * np.pi)) * inverse

    group = double_cover(little_group(frame.d))
    irrep = "A1g" if frame.d == (0, 0, 0) else "A1"
    return list(
        quantization_roots(
            window,
            frame,
            masses,
            ((0, 0),),
            {0: reduced_inverse},
            group=group,
            irrep=irrep,
            samples=SAMPLES,
            weighting=WEIGHTING_THRESHOLD,
        )
    )


def _assert_same(tag, frame, masses, k_matrix, window, phase_space, subtractions=()):
    model = ConstantK(matrix=k_matrix)
    channel = list(
        coupled_s_roots(
            frame, masses, model, phase_space, window,
            subtractions=subtractions, samples=SAMPLES,
        )
    )
    jls = _jls_roots(frame, masses, model, window, phase_space, subtractions)
    assert len(channel) == len(jls), f"{tag}: {len(channel)} roots vs {len(jls)}"
    worst = max((abs(a - b) for a, b in zip(channel, jls)), default=0.0)
    assert worst <= 1e-9, f"{tag}: worst deviation {worst:.3e}"


@pytest.mark.parametrize(
    "tag, sites, anisotropy, d, window, phase_space, subtractions",
    [
        ("rest-simple", 24, 1.0, (0, 0, 0), (0.61, 0.95), "simple", ()),
        ("moving-simple", 24, 1.0, (0, 0, 1), (0.61, 1.15), "simple", ()),
        ("moving-diagonal-simple", 24, 1.0, (1, 1, 0), (0.61, 1.15), "simple", ()),
        ("rest-chew-mandelstam", 24, 1.0, (0, 0, 0), (0.61, 0.95), "chew-mandelstam", (0.0,)),
        ("subtracted-chew-mandelstam", 24, 1.0, (0, 0, 0), (0.61, 0.95), "chew-mandelstam", (0.5,)),
    ],
)
def test_single_channel_kernels_agree(tag, sites, anisotropy, d, window, phase_space, subtractions):
    frame = LatticeFrame(sites, anisotropy, d)
    _assert_same(tag, frame, [(0.3, 0.3)], [[2.0]], window, phase_space, subtractions)


def test_anisotropic_single_channel_kernels_agree_within_zeta_domain():
    frame = LatticeFrame(24, 1.2, (1, 1, 0))
    mass = 0.3
    boundary = _equal_mass_energy_at_q2(frame, mass, Q2_MAX)
    supported_high = np.nextafter(boundary, 0.0)
    point = two_body_point(
        energy_lab_at=supported_high, mass1_at=mass, mass2_at=mass, frame=frame
    )
    assert point.q_squared <= Q2_MAX
    _assert_same(
        "anisotropic-simple",
        frame,
        [(mass, mass)],
        [[2.0]],
        (0.61, supported_high),
        "simple",
    )


def test_anisotropic_window_above_supported_q2_fails_closed():
    frame = LatticeFrame(24, 1.2, (1, 1, 0))
    mass = 0.3
    boundary = _equal_mass_energy_at_q2(frame, mass, Q2_MAX)
    unsupported_low = np.nextafter(boundary, np.inf)
    original_high = 1.15
    assert two_body_point(
        energy_lab_at=unsupported_low, mass1_at=mass, mass2_at=mass, frame=frame
    ).q_squared > Q2_MAX
    assert two_body_point(
        energy_lab_at=original_high, mass1_at=mass, mass2_at=mass, frame=frame
    ).q_squared > Q2_MAX
    with pytest.raises(RootScanError, match="outside supported range") as captured:
        coupled_s_roots(
            frame,
            [(mass, mass)],
            ConstantK(matrix=[[2.0]]),
            "simple",
            (unsupported_low, original_high),
            samples=4,
        )
    assert captured.value.energy >= unsupported_low
    assert isinstance(captured.value.cause, ValueError)


@pytest.mark.parametrize(
    "tag, phase_space, subtractions",
    [
        ("two-channel-simple", "simple", ()),
        ("two-channel-chew-mandelstam", "chew-mandelstam", (0.0, 0.3)),
    ],
)
def test_two_channel_kernels_agree(tag, phase_space, subtractions):
    """Two channels put a root below the second threshold, where k^2 < 0."""
    frame = LatticeFrame(24, 1.0, (0, 0, 0))
    _assert_same(
        tag, frame, [(0.3, 0.3), (0.35, 0.35)], [[2.0, 0.3], [0.3, 1.5]],
        (0.61, 1.00), phase_space, subtractions,
    )


def test_unequal_masses_agree():
    frame = LatticeFrame(24, 1.0, (0, 0, 0))
    _assert_same(
        "unequal-simple", frame, [(0.2, 0.45)], [[2.0]], (0.66, 0.95), "simple"
    )
    _assert_same(
        "unequal-chew-mandelstam", frame, [(0.2, 0.45)], [[2.0]], (0.66, 0.95),
        "chew-mandelstam", (0.4,),
    )


def test_unknown_phase_space_is_refused():
    frame = LatticeFrame(24, 1.0, (0, 0, 0))
    with pytest.raises(ValueError, match="unknown phase space"):
        coupled_s_roots(
            frame, [(0.3, 0.3)], ConstantK(matrix=[[2.0]]), "nonsense", (0.6, 0.9)
        )
