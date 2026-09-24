"""Independent scientific acceptance tests for generalized JLS capabilities."""

from __future__ import annotations

import numpy as np
import pytest

from lattice_scattering.finite_volume import (
    assemble_inverse,
    normalise_channel_layout,
    quantization_roots,
    row_matrix,
)
from lattice_scattering.finite_volume.box import mixed_box
from lattice_scattering.finite_volume.jls_channels import channel_row_basis
from lattice_scattering.kinematics import LatticeFrame, two_body_point
from lattice_scattering.symmetry import double_cover, little_group
from lattice_scattering.symmetry.angular_momentum import coupled_j_basis


FRAME = LatticeFrame(8, 1.0, (0, 0, 1))
GROUP = double_cover(little_group(FRAME.d))
SP_SECTORS = ((0, 1), (1, 1))
ONE_MASS = ((0.2, 0.3),)
TWO_MASSES = ((0.2, 0.3), (0.25, 0.35))


def _constant(value):
    matrix = np.asarray(value, dtype=float)
    return lambda _s: matrix


@pytest.mark.parametrize("irrep", ["G1", "G2"])
def test_generalized_all_active_is_unitarily_equivalent_to_legacy(irrep):
    blocks = {
        1: _constant([[1.2, 0.0], [0.0, 0.9]]),
        3: _constant([[0.8]]),
    }
    legacy = row_matrix(
        1.2,
        FRAME,
        ONE_MASS,
        SP_SECTORS,
        blocks,
        group=GROUP,
        irrep=irrep,
        intrinsic_parity=-1,
    )
    generalized = row_matrix(
        1.2,
        FRAME,
        ONE_MASS,
        None,
        blocks,
        group=GROUP,
        irrep=irrep,
        channel_sectors=(SP_SECTORS,),
        channel_intrinsic_parities=(-1,),
    )
    assert generalized.shape == legacy.shape
    np.testing.assert_allclose(
        np.linalg.eigvalsh(generalized), np.linalg.eigvalsh(legacy), atol=1e-10, rtol=1e-10
    )
    np.testing.assert_allclose(
        np.linalg.det(generalized), np.linalg.det(legacy), atol=1e-10, rtol=1e-10
    )


def test_selective_s31_p33_projection_matches_explicit_active_formula():
    selected = {1: ((0, 0, 1),), 3: ((0, 1, 1),)}
    layout = normalise_channel_layout(
        None,
        channels=1,
        channel_sectors=(SP_SECTORS,),
        channel_intrinsic_parities=(-1,),
        selected_j_sectors=selected,
    )
    assert layout.active_labels == selected

    W, V, P = channel_row_basis(layout, GROUP, "G1", 0)
    np.testing.assert_allclose(P @ P, P, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(P, P.conj().T, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(W.conj().T @ W, np.eye(W.shape[1]), atol=1e-10, rtol=1e-10)
    row_projector_spectrum = np.linalg.eigvalsh(V.conj().T @ P @ V)
    distance_to_binary = np.minimum(
        np.abs(row_projector_spectrum), np.abs(row_projector_spectrum - 1.0)
    )
    assert np.max(distance_to_binary) < 1e-10

    matrices = {1: np.array([[1.2]]), 3: np.array([[0.8]])}
    callbacks = {twice_j: _constant(value) for twice_j, value in matrices.items()}
    actual = row_matrix(
        1.2,
        FRAME,
        ONE_MASS,
        None,
        callbacks,
        group=GROUP,
        irrep="G1",
        channel_sectors=(SP_SECTORS,),
        channel_intrinsic_parities=(-1,),
        selected_j_sectors=selected,
    )
    assert actual.shape == (2, 2)

    full = assemble_inverse(
        None,
        scale=FRAME.length_at / (2 * np.pi),
        j_matrices=matrices,
        channels=1,
        channel_sectors=(SP_SECTORS,),
        channel_intrinsic_parities=(-1,),
        selected_j_sectors=selected,
    )
    point = two_body_point(
        energy_lab_at=1.2, mass1_at=ONE_MASS[0][0], mass2_at=ONE_MASS[0][1], frame=FRAME
    )
    full -= mixed_box(
        point.q_squared,
        SP_SECTORS,
        d=FRAME.d,
        gamma=point.gamma,
        alpha=point.alpha,
    )
    expected = W.conj().T @ full @ W
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)

    with pytest.raises(ValueError, match="order 1"):
        row_matrix(
            1.2,
            FRAME,
            ONE_MASS,
            None,
            {1: _constant(np.eye(2)), 3: _constant([[0.8]])},
            group=GROUP,
            irrep="G1",
            channel_sectors=(SP_SECTORS,),
            channel_intrinsic_parities=(-1,),
            selected_j_sectors=selected,
        )
    with pytest.raises(ValueError, match="exactly the active blocks"):
        row_matrix(
            1.2,
            FRAME,
            ONE_MASS,
            None,
            {1: _constant([[1.2]]), 3: _constant([[0.8]]), 5: _constant([[0.4]])},
            group=GROUP,
            irrep="G1",
            channel_sectors=(SP_SECTORS,),
            channel_intrinsic_parities=(-1,),
            selected_j_sectors=selected,
        )
    with pytest.raises(ValueError, match="absent"):
        channel_row_basis(layout, GROUP, "A1", 0)


@pytest.mark.parametrize(("irrep", "expected_rank"), [("G1", 2), ("G2", 1)])
def test_s31_p33_active_projector_excludes_p31_and_intersects_each_row(irrep, expected_rank):
    # Input key order is intentionally not the canonical sector order.  The
    # retained space is S31 (J=1/2) plus P33 (J=3/2), with P31 (J=1/2) absent.
    selected = {3: ((0, 1, 1),), 1: ((0, 0, 1),)}
    layout = normalise_channel_layout(
        None,
        channels=1,
        channel_sectors=(SP_SECTORS,),
        channel_intrinsic_parities=(-1,),
        selected_j_sectors=selected,
    )
    W, V, P = channel_row_basis(layout, GROUP, irrep, 0)

    s_wave = layout.sector_slice(0, 0)
    p_wave = layout.sector_slice(0, 1)
    u_s31 = coupled_j_basis(0, 1, 1)
    u_p31 = coupled_j_basis(2, 1, 1)
    u_p33 = coupled_j_basis(2, 1, 3)
    np.testing.assert_allclose(P[s_wave, s_wave] @ u_s31, u_s31, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(P[p_wave, p_wave] @ u_p31, 0.0, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(
        P[p_wave, p_wave] @ u_p33, u_p33, atol=1e-10, rtol=1e-10
    )
    np.testing.assert_allclose(P @ W, W, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(V.conj().T @ V, np.eye(V.shape[1]), atol=1e-10, rtol=1e-10)
    row_projector = V @ V.conj().T
    np.testing.assert_allclose(row_projector @ W, W, atol=1e-10, rtol=1e-10)
    assert W.shape[1] == expected_rank
    assert np.linalg.matrix_rank(P @ V, tol=1e-9) == expected_rank

    spectrum = np.linalg.eigvalsh(V.conj().T @ P @ V)
    assert np.count_nonzero(spectrum > 1.0 - 1e-9) == expected_rank
    assert np.count_nonzero(np.abs(spectrum) < 1e-9) == V.shape[1] - expected_rank


def test_selected_incidence_entries_are_canonicalized_before_block_assembly():
    channel_sectors = (((0, 1),), ((1, 1),))
    canonical = {1: ((0, 0, 1), (1, 1, 1))}
    reversed_entries = {1: ((1, 1, 1), (0, 0, 1))}
    options = dict(
        sectors=None,
        channels=2,
        channel_sectors=channel_sectors,
        channel_intrinsic_parities=(1, -1),
    )
    expected_labels = {1: ((0, 0, 1), (1, 1, 1))}
    assert normalise_channel_layout(
        **options, selected_j_sectors=reversed_entries
    ).active_labels == expected_labels

    matrix = np.array([[1.4, 0.35], [0.35, 0.7]])
    canonical_result = assemble_inverse(
        **options,
        scale=1.3,
        j_matrices={1: matrix},
        selected_j_sectors=canonical,
    )
    reversed_result = assemble_inverse(
        **options,
        scale=1.3,
        j_matrices={1: matrix},
        selected_j_sectors=reversed_entries,
    )
    np.testing.assert_allclose(reversed_result, canonical_result, atol=1e-12, rtol=1e-12)


def test_selected_projection_rejects_an_irrep_row_removed_by_the_selection():
    layout = normalise_channel_layout(
        None,
        channels=1,
        channel_sectors=(SP_SECTORS,),
        channel_intrinsic_parities=(-1,),
        selected_j_sectors={1: ((0, 0, 1),)},
    )
    # G2 occurs in the untruncated P33 sector, but the selected S31-only space
    # contains no G2 row.
    with pytest.raises(ValueError, match="absent after the selected-J projection"):
        channel_row_basis(layout, GROUP, "G2", 0)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"selected_j_sectors": {}}, "retain at least one J"),
        ({"selected_j_sectors": {1: ()}}, "cannot be empty"),
        ({"selected_j_sectors": {99: ((0, 0, 1),)}}, "unknown; allowed values"),
        (
            {"selected_j_sectors": {1: ((0, 0, 1), (0, 0, 1))}},
            "duplicate selected incidence",
        ),
        ({"selected_j_sectors": {1: ((0, 2, 1),)}}, "absent from channel sectors"),
        ({"selected_j_sectors": {3: ((0, 0, 1),)}}, "does not participate"),
        ({"selected_j_sectors": {1: ((1, 0, 1),)}}, "unknown channel"),
        ({"sectors": SP_SECTORS}, "either the common sectors argument or channel_sectors"),
    ],
    ids=(
        "empty-selected-map",
        "empty-selected-j-entry",
        "unknown-twice-j",
        "duplicate-incidence",
        "absent-channel-sector",
        "triangle-rule-violation",
        "unknown-channel",
        "common-and-per-channel-layouts",
    ),
)
def test_invalid_selective_layouts_fail_closed(overrides, error):
    arguments = {
        "sectors": None,
        "channels": 1,
        "channel_sectors": (SP_SECTORS,),
        "channel_intrinsic_parities": (-1,),
        "selected_j_sectors": {1: ((0, 0, 1),)},
    }
    arguments.update(overrides)
    with pytest.raises(ValueError, match=error):
        normalise_channel_layout(**arguments)


def test_heterogeneous_intrinsic_parity_uses_total_parity_rule():
    channel_sectors = (((0, 1),), ((1, 1),))
    selected = {1: ((0, 0, 1), (1, 1, 1))}
    coupled = {1: _constant([[1.2, 0.4], [0.4, 0.8]])}
    result = row_matrix(
        1.2,
        FRAME,
        TWO_MASSES,
        None,
        coupled,
        group=GROUP,
        irrep="G1",
        channel_sectors=channel_sectors,
        channel_intrinsic_parities=(1, -1),
        selected_j_sectors=selected,
    )
    assert result.shape == (2, 2)
    np.testing.assert_allclose(result, result.conj().T, atol=1e-10, rtol=1e-10)

    same_ell_sectors = (((0, 1),), ((0, 1),))
    same_ell_selected = {1: ((0, 0, 1), (1, 0, 1))}
    with pytest.raises(ValueError, match="equal physical parity"):
        row_matrix(
            1.2,
            FRAME,
            TWO_MASSES,
            None,
            coupled,
            group=GROUP,
            irrep="G1",
            channel_sectors=same_ell_sectors,
            channel_intrinsic_parities=(1, -1),
            selected_j_sectors=same_ell_selected,
        )
    uncoupled = {1: _constant([[1.2, 0.0], [0.0, 0.8]])}
    accepted = row_matrix(
        1.2,
        FRAME,
        TWO_MASSES,
        None,
        uncoupled,
        group=GROUP,
        irrep="G1",
        channel_sectors=same_ell_sectors,
        channel_intrinsic_parities=(1, -1),
        selected_j_sectors=same_ell_selected,
    )
    assert accepted.shape == (2, 2)


def test_active_inverse_scale_and_threshold_weights_match_direct_embedding():
    channel_sectors = (((0, 1),), ((1, 1),))
    selected = {1: ((0, 0, 1), (1, 1, 1))}
    inverse = np.array([[1.2, 0.4], [0.4, 0.8]])
    scale = 1.7
    U0 = coupled_j_basis(0, 1, 1)
    U1 = coupled_j_basis(2, 1, 1)

    def embedded(f0, f1):
        expected = np.zeros((8, 8), dtype=complex)
        expected[:2, :2] = f0 * f0 * inverse[0, 0] * (U0 @ U0.conj().T)
        expected[:2, 2:] = f0 * f1 * inverse[0, 1] * (U0 @ U1.conj().T)
        expected[2:, :2] = f1 * f0 * inverse[1, 0] * (U1 @ U0.conj().T)
        expected[2:, 2:] = f1 * f1 * inverse[1, 1] * (U1 @ U1.conj().T)
        return expected

    scaled = assemble_inverse(
        None,
        scale=scale,
        j_matrices={1: inverse},
        channels=2,
        channel_sectors=channel_sectors,
        channel_intrinsic_parities=(1, -1),
        selected_j_sectors=selected,
    )
    np.testing.assert_allclose(scaled, embedded(scale**0.5, scale**1.5), atol=1e-12, rtol=1e-12)

    threshold = assemble_inverse(
        None,
        scale=scale,
        j_matrices={1: inverse},
        channels=2,
        channel_sectors=channel_sectors,
        channel_intrinsic_parities=(1, -1),
        selected_j_sectors=selected,
        weighting="threshold",
        channel_k_squared_at2=(-0.1, 0.2),
    )
    p_factor = 1.0 / (2.0 * np.sqrt(0.2))
    np.testing.assert_allclose(threshold, embedded(1.0, p_factor), atol=1e-12, rtol=1e-12)
    with pytest.raises(ValueError, match="channel 1 is below its threshold"):
        assemble_inverse(
            None,
            scale=scale,
            j_matrices={1: inverse},
            channels=2,
            channel_sectors=channel_sectors,
            channel_intrinsic_parities=(1, -1),
            selected_j_sectors=selected,
            weighting="threshold",
            channel_k_squared_at2=(0.1, -0.2),
        )


def test_heterogeneous_roots_are_density_stable_and_have_small_matrix_residuals():
    selected = {1: ((0, 0, 1), (1, 1, 1))}
    callbacks = {1: _constant([[1.2, 0.4], [0.4, 0.8]])}
    common = dict(
        frame=FRAME,
        channel_masses=TWO_MASSES,
        sectors=None,
        j_amplitudes=callbacks,
        group=GROUP,
        irrep="G1",
        channel_sectors=(((0, 1),), ((1, 1),)),
        channel_intrinsic_parities=(1, -1),
        selected_j_sectors=selected,
    )
    sparse = quantization_roots((1.08, 1.3), samples=8, **common)
    dense = quantization_roots((1.08, 1.3), samples=16, **common)
    np.testing.assert_allclose(sparse, dense, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(
        dense,
        (1.09240502776297, 1.10647496980251, 1.16991929609188),
        atol=1e-9,
        rtol=0.0,
    )
    for energy in dense:
        matrix = row_matrix(energy, **common)
        singular = np.linalg.svd(matrix, compute_uv=False)
        relative_residual = singular[-1] / max(1.0, singular[0])
        assert relative_residual < 1e-10
