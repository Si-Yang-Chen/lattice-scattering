"""Independent tests for the explicit-incidence 2102 partial-wave adapter."""

from __future__ import annotations

from functools import lru_cache
import io
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import subprocess
import sys
import tarfile
import tempfile

import numpy as np
import pytest

from lattice_scattering.amplitudes import (
    ConstantK,
    PartialWaveJMatrixAdapter,
    cm_real_axis,
)
from lattice_scattering.finite_volume import assemble_inverse, row_matrix
from lattice_scattering.finite_volume.box import mixed_box
from lattice_scattering.finite_volume.zeta import harmonic_zeta_wide
from lattice_scattering.finite_volume.jls_channels import (
    channel_row_basis,
    normalise_channel_layout,
)
from lattice_scattering.kinematics import LatticeFrame, two_body_point
from lattice_scattering.symmetry import double_cover, little_group, oh_group


MASSES = ((0.2, 0.3),)
SPINLESS_DPI_MASSES = ((0.31, 0.46),)


class _KWithPole:
    """One-channel K matrix whose inverse has a genuine zero at ``pole_s``."""

    def __init__(self, pole_s, residue=0.7):
        self.pole_s = float(pole_s)
        self.residue = float(residue)

    def inverse(self, s):
        return np.array([[(self.pole_s - float(s)) / self.residue]])


def _source_inverse_from_phase(ell, k, ecm, delta):
    """Construct K^-1 from Khat^-1=(2k/Ecm)cot(delta), independently."""
    cot_delta = 1.0 / np.tan(delta)
    khat_inverse = (2.0 * k / ecm) * cot_delta
    return (2.0 * k) ** (2 * ell) * khat_inverse


def _validate_historical_archive_members(members):
    """Allow only safe members under the pinned ``src/lattice_scattering`` tree."""
    package = ("src", "lattice_scattering")
    ancestors = {("src",), package}
    for member in members:
        name = member.name
        if not isinstance(name, str) or not name or "\\" in name:
            raise AssertionError(f"unexpected path in pinned historical source archive: {name!r}")
        member_path = PurePosixPath(name)
        parts = member_path.parts
        if (
            member_path.is_absolute()
            or ".." in parts
            or member_path.as_posix().rstrip("/") != name.rstrip("/")
        ):
            raise AssertionError(f"unexpected path in pinned historical source archive: {name}")
        if parts in ancestors:
            allowed_type = member.isdir()
        elif parts[:2] == package:
            allowed_type = member.isfile() or member.isdir()
        else:
            allowed_type = False
        if not allowed_type:
            raise AssertionError(f"unexpected path in pinned historical source archive: {name}")


@lru_cache(maxsize=1)
def _historical_coupled_axial_source_tree():
    """Extract the pinned historical package for isolated test subprocesses."""
    repository = Path(__file__).resolve().parents[2] / "lattice-scattering-software"
    revision = "5c562e3aace00ea74444b89abe65de6b0297c027"
    source_path = "src/lattice_scattering/finite_volume/coupled_axial.py"
    expected_blob = "cd5513b09bfc210c92e60d316386ba4484e59085"
    actual_blob = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", f"{revision}:{source_path}"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    if actual_blob != expected_blob:
        raise AssertionError(
            f"historical coupled_axial source blob changed: expected {expected_blob}, got {actual_blob}"
        )
    archive = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "archive",
            "--format=tar",
            revision,
            "src/lattice_scattering",
        ],
        check=True,
        capture_output=True,
    ).stdout
    temporary = tempfile.TemporaryDirectory(prefix="wave-adapter-historical-")
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as source_archive:
        members = source_archive.getmembers()
        _validate_historical_archive_members(members)
        source_archive.extractall(temporary.name, members=members)
    return temporary, Path(temporary.name) / "src"


def _historical_quantization_matrix(energy, s_inverse, p_reduced_inverse, masses, frame):
    """Call the pinned 5c562e3 kernel in an isolated Python process."""
    temporary, source_root = _historical_coupled_axial_source_tree()
    payload = {
        "energy": float(energy),
        "s_inverse": float(s_inverse),
        "p_reduced_inverse": np.asarray(p_reduced_inverse, dtype=float).tolist(),
        "masses": [list(pair) for pair in masses],
        "spatial_sites": int(frame.spatial_sites),
        "anisotropy": float(frame.anisotropy),
        "d": list(frame.d),
    }
    runner = r"""
import json
import sys
import numpy as np
from lattice_scattering.kinematics import LatticeFrame
from lattice_scattering.finite_volume.coupled_axial import quantization_matrix

class _InverseModel:
    def __init__(self, value):
        self.value = float(value)
    def inverse(self, _s):
        return np.array([[self.value]])

data = json.loads(sys.argv[1])
frame = LatticeFrame(data["spatial_sites"], data["anisotropy"], tuple(data["d"]))
matrix = quantization_matrix(
    data["energy"],
    _InverseModel(data["s_inverse"]),
    np.asarray(data["p_reduced_inverse"], dtype=float),
    data["masses"],
    frame,
)
print(json.dumps(np.asarray(matrix).real.tolist()))
"""
    environment = os.environ.copy()
    inherited_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(source_root), inherited_pythonpath) if part
    )
    result = subprocess.run(
        [sys.executable, "-c", runner, json.dumps(payload)],
        check=True,
        capture_output=True,
        text=True,
        cwd=temporary.name,
        env=environment,
    )
    return np.asarray(json.loads(result.stdout), dtype=float)


def _archive_member(name, member_type):
    member = tarfile.TarInfo(name)
    member.type = member_type
    return member


def test_historical_archive_validator_accepts_git_archive_package_ancestors():
    _validate_historical_archive_members(
        [
            _archive_member("src", tarfile.DIRTYPE),
            _archive_member("src/lattice_scattering", tarfile.DIRTYPE),
            _archive_member("src/lattice_scattering/amplitudes", tarfile.DIRTYPE),
            _archive_member("src/lattice_scattering/__init__.py", tarfile.REGTYPE),
        ]
    )


@pytest.mark.parametrize(
    ("name", "member_type"),
    [
        ("../outside.py", tarfile.REGTYPE),
        ("src/../outside.py", tarfile.REGTYPE),
        ("/src/lattice_scattering/outside.py", tarfile.REGTYPE),
        ("src/lattice_scattering_extra/module.py", tarfile.REGTYPE),
        ("src/other.py", tarfile.REGTYPE),
        (r"src\..\outside.py", tarfile.REGTYPE),
        ("src", tarfile.REGTYPE),
        ("src/lattice_scattering/link", tarfile.SYMTYPE),
    ],
)
def test_historical_archive_validator_rejects_unsafe_or_out_of_package_members(
    name, member_type
):
    with pytest.raises(AssertionError, match="unexpected path"):
        _validate_historical_archive_members([_archive_member(name, member_type)])


def _one_wave(ell, *, phase_space="simple", subtraction_point=None, model=None):
    twice_j = 2 * ell
    incidence = (0, ell, 0, twice_j)
    subtraction_points = (
        None if phase_space == "simple" else {incidence: subtraction_point}
    )
    adapter = PartialWaveJMatrixAdapter(
        {incidence: ConstantK([[2.0]]) if model is None else model},
        channel_masses=MASSES,
        channel_sectors=(((ell, 0),),),
        phase_space=phase_space,
        subtraction_points=subtraction_points,
    )
    return adapter, incidence, twice_j


@pytest.mark.parametrize(
    ("ell", "phase_space", "subtraction_point"),
    [
        (0, "simple", None),
        (1, "simple", None),
        (0, "chew-mandelstam", "threshold"),
        (1, "chew-mandelstam", ("k-pole", 0.49)),
    ],
)
def test_eq_2102_3_matches_independent_t_inverse(
    ell, phase_space, subtraction_point
):
    adapter, incidence, twice_j = _one_wave(
        ell,
        phase_space=phase_space,
        subtraction_point=subtraction_point,
        model=(
            _KWithPole(subtraction_point[1])
            if isinstance(subtraction_point, tuple)
            else None
        ),
    )
    s = 0.64
    m1, m2 = MASSES[0]
    k_squared = (s - (m1 + m2) ** 2) * (s - (m1 - m2) ** 2) / (4.0 * s)
    rho = 2.0 * np.sqrt(k_squared) / np.sqrt(s)
    model = _KWithPole(subtraction_point[1]) if isinstance(subtraction_point, tuple) else ConstantK([[2.0]])
    inverse_k = model.inverse(s)[0, 0]

    real_i = 0.0
    if phase_space == "chew-mandelstam":
        s_sub = (
            (m1 + m2) ** 2
            if subtraction_point == "threshold"
            else subtraction_point[1]
        )
        real_i = np.real(
            cm_real_axis(s, m1, m2)
            - cm_real_axis(s_sub, m1, m2)
        )

    # Eq. (2102-2): t^-1 = (2k)^(-2l) K^-1 + I, with Im I = -rho.
    t_inverse_real = (2.0 * np.sqrt(k_squared)) ** (-2 * ell) * inverse_k + real_i
    cot_delta = t_inverse_real / rho
    independent = np.sqrt(s) / 2.0 * k_squared**ell * rho * cot_delta
    actual = adapter.blocks(s)[twice_j][0, 0]
    np.testing.assert_allclose(actual, independent, rtol=1e-13, atol=1e-13)

    if ell == 1 and phase_space == "simple":
        np.testing.assert_allclose(actual, np.sqrt(s) * inverse_k / 8.0, atol=1e-14)


@pytest.mark.parametrize("phase_space", ["simple", "chew-mandelstam"])
def test_p_wave_has_finite_exact_threshold_limit(phase_space):
    adapter, _, twice_j = _one_wave(
        1,
        phase_space=phase_space,
        subtraction_point=None if phase_space == "simple" else "threshold",
    )
    threshold = sum(MASSES[0]) ** 2
    actual = adapter.blocks(threshold)[twice_j][0, 0]
    expected = np.sqrt(threshold) * ConstantK([[2.0]]).inverse(threshold)[0, 0] / 8.0
    assert np.isfinite(actual)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-14)


def test_selected_active_incidences_are_exposed_and_block_shapes_are_exact():
    sectors = (((0, 1), (1, 1)),)
    selected = {1: ((0, 0, 1),), 3: ((0, 1, 1),)}
    models = {
        (0, 0, 1, 1): ConstantK([[2.0]]),
        (0, 1, 1, 3): ConstantK([[3.0]]),
    }
    adapter = PartialWaveJMatrixAdapter(
        models,
        channel_masses=MASSES,
        channel_sectors=sectors,
        selected_j_sectors=selected,
    )
    assert adapter.selected_j_sectors == selected
    blocks = adapter.blocks(0.64)
    assert set(blocks) == {1, 3}
    assert blocks[1].shape == (1, 1)
    assert blocks[3].shape == (1, 1)


def test_p_wave_phase_shift_fixes_physical_k_cubed_normalization():
    frame = LatticeFrame(16, 1.0, (0, 0, 0))
    masses = SPINLESS_DPI_MASSES[0]
    q_squared = 0.37
    q = np.sqrt(q_squared)
    k = q / (frame.length_at / (2.0 * np.pi))
    ecm = np.sqrt(masses[0] ** 2 + k**2) + np.sqrt(masses[1] ** 2 + k**2)
    delta_1 = 0.57
    cot_delta_1 = 1.0 / np.tan(delta_1)
    khat_inverse = (2.0 * k / ecm) * cot_delta_1
    source_inverse = (2.0 * k) ** 2 * khat_inverse
    adapter = PartialWaveJMatrixAdapter(
        {(0, 1, 0, 2): ConstantK([[1.0 / source_inverse]])},
        channel_masses=(masses,),
        channel_sectors=(((1, 0),),),
        selected_j_sectors={2: ((0, 1, 0),)},
    )

    r_physical = k**3 * cot_delta_1
    q_physical = q**3 * cot_delta_1
    actual = adapter.blocks(ecm**2)[2][0, 0]
    np.testing.assert_allclose(actual, r_physical, rtol=1e-13, atol=0.0)
    np.testing.assert_allclose(
        (frame.length_at / (2.0 * np.pi)) ** 3 * r_physical,
        q_physical,
        rtol=1e-13,
        atol=0.0,
    )


@pytest.mark.parametrize("q_squared", [0.04, 0.55])
def test_rest_spinless_t1u_row_matches_scalar_luescher_oracle(q_squared):
    frame = LatticeFrame(16, 1.0, (0, 0, 0))
    masses = SPINLESS_DPI_MASSES[0]
    scale = frame.length_at / (2.0 * np.pi)
    k = np.sqrt(q_squared) / scale
    ecm = np.sqrt(masses[0] ** 2 + k**2) + np.sqrt(masses[1] ** 2 + k**2)
    delta_1 = 0.43
    cot_delta_1 = 1.0 / np.tan(delta_1)
    source_inverse = _source_inverse_from_phase(1, k, ecm, delta_1)
    adapter = PartialWaveJMatrixAdapter(
        {(0, 1, 0, 2): ConstantK([[1.0 / source_inverse]])},
        channel_masses=(masses,),
        channel_sectors=(((1, 0),),),
        selected_j_sectors={2: ((0, 1, 0),)},
    )

    catalog = double_cover(oh_group())
    assert "T1u" in {irrep.label for irrep in catalog.irreps}
    assert q_squared > 0.0 and not np.isclose(q_squared, round(q_squared))
    matrix = row_matrix(
        ecm,
        frame,
        (masses,),
        None,
        adapter,
        group=catalog,
        irrep="T1u",
        channel_sectors=adapter.channel_sectors,
        channel_intrinsic_parities=(1,),
        selected_j_sectors=adapter.selected_j_sectors,
    )
    z00 = harmonic_zeta_wide(
        q_squared, 0, d=(0, 0, 0), gamma=1.0, alpha=0.0
    ).values[0].real
    expected = q_squared**1.5 * cot_delta_1 - q_squared * z00 / np.pi**1.5
    eigenvalue = np.linalg.eigvalsh(matrix)[0]
    assert matrix.shape == (1, 1)
    np.testing.assert_allclose(
        eigenvalue,
        expected,
        rtol=1e-11,
        atol=1e-12 * max(1.0, abs(expected)),
    )


def test_moving_frame_spinless_s_p_row_matches_independent_source_oracle():
    frame = LatticeFrame(16, 1.0, (0, 0, 1))
    masses = SPINLESS_DPI_MASSES[0]
    energy = 1.1
    point = two_body_point(
        energy_lab_at=energy,
        mass1_at=masses[0],
        mass2_at=masses[1],
        frame=frame,
    )
    k = np.sqrt(point.k_squared_at2)
    ecm = np.sqrt(point.s_at2)
    delta_0, delta_1 = 0.37, 0.61
    source_s_inverse = _source_inverse_from_phase(0, k, ecm, delta_0)
    source_p_inverse = _source_inverse_from_phase(1, k, ecm, delta_1)
    r_s = k / np.tan(delta_0)
    r_p = k**3 / np.tan(delta_1)
    sectors = (((0, 0), (1, 0)),)
    selected = {0: ((0, 0, 0),), 2: ((0, 1, 0),)}
    s_model = ConstantK([[1.0 / source_s_inverse]])
    group = double_cover(little_group(frame.d))
    adapter = PartialWaveJMatrixAdapter(
        {
            (0, 0, 0, 0): s_model,
            (0, 1, 0, 2): ConstantK([[1.0 / source_p_inverse]]),
        },
        channel_masses=(masses,),
        channel_sectors=sectors,
        selected_j_sectors=selected,
    )
    matrix = row_matrix(
        energy,
        frame,
        (masses,),
        None,
        adapter,
        group=group,
        irrep="A1",
        channel_sectors=adapter.channel_sectors,
        channel_intrinsic_parities=(1,),
        selected_j_sectors=adapter.selected_j_sectors,
    )
    historical = _historical_quantization_matrix(
        energy, source_s_inverse, np.array([[r_p]]), (masses,), frame
    )
    np.testing.assert_allclose(
        adapter.blocks(point.s_at2)[0][0, 0], r_s, rtol=1e-13, atol=0.0
    )
    np.testing.assert_allclose(
        adapter.blocks(point.s_at2)[2][0, 0], r_p, rtol=1e-13, atol=0.0
    )
    assert matrix.shape == historical.shape == (2, 2)
    np.testing.assert_allclose(
        np.linalg.eigvalsh(matrix),
        np.linalg.eigvalsh(historical),
        rtol=2e-11,
        atol=2e-10,
    )


def test_moving_p_only_row_matches_historical_coupled_axial_oracle():
    frame = LatticeFrame(16, 1.0, (0, 0, 1))
    masses = SPINLESS_DPI_MASSES[0]
    energy = 1.1
    point = two_body_point(
        energy_lab_at=energy,
        mass1_at=masses[0],
        mass2_at=masses[1],
        frame=frame,
    )
    k = np.sqrt(point.k_squared_at2)
    ecm = np.sqrt(point.s_at2)
    delta_1 = 0.61
    r_p = k**3 / np.tan(delta_1)
    source_inverse = _source_inverse_from_phase(1, k, ecm, delta_1)
    sectors = (((1, 0),),)
    selected = {2: ((0, 1, 0),)}
    adapter = PartialWaveJMatrixAdapter(
        {(0, 1, 0, 2): ConstantK([[1.0 / source_inverse]])},
        channel_masses=(masses,),
        channel_sectors=sectors,
        selected_j_sectors=selected,
    )
    matrix = row_matrix(
        energy,
        frame,
        (masses,),
        None,
        adapter,
        group=double_cover(little_group(frame.d)),
        irrep="A1",
        channel_sectors=adapter.channel_sectors,
        channel_intrinsic_parities=(1,),
        selected_j_sectors=adapter.selected_j_sectors,
    )
    historical = _historical_quantization_matrix(
        energy, 1.0, np.array([[r_p]]), (masses,), frame
    )
    assert matrix.shape == (1, 1)
    np.testing.assert_allclose(
        np.linalg.eigvalsh(matrix),
        historical[1:2, 1:2].diagonal().real,
        rtol=2e-11,
        atol=2e-10,
    )


@pytest.mark.parametrize(
    ("masses", "sectors", "models", "kwargs", "message"),
    [
        (((0.0, 0.3),), (((0, 0),),), {(0, 0, 0, 0): ConstantK([[1.0]])}, {}, "positive"),
        (((0.2, 0.3, 0.4),), (((0, 0),),), {(0, 0, 0, 0): ConstantK([[1.0]])}, {}, "exactly two"),
        (MASSES, (((1, 1),),), {(0, 1, 1, 1): ConstantK([[1.0]])}, {}, "missing=.*extra"),
        (
            MASSES,
            (((1, 1),),),
            {(0, 1, 1, 1): ConstantK([[1.0]]), (0, 1, 1, 3): ConstantK([[1.0]])},
            {"selected_j_sectors": {3: ((0, 1, 1),)}},
            "missing=.*extra",
        ),
        (
            MASSES,
            (((0, 1),),),
            {(0, 0, 1, 0): ConstantK([[1.0]])},
            {},
            "missing=.*extra",
        ),
    ],
)
def test_invalid_masses_spin_and_active_incidences_fail_closed(
    masses, sectors, models, kwargs, message
):
    with pytest.raises(ValueError, match=message):
        PartialWaveJMatrixAdapter(
            models,
            channel_masses=masses,
            channel_sectors=sectors,
            **kwargs,
        )


def test_non_integral_spin_is_rejected():
    with pytest.raises(ValueError, match="twice_S must be an integer"):
        PartialWaveJMatrixAdapter(
            {(0, 0, 0, 0): ConstantK([[1.0]])},
            channel_masses=MASSES,
            channel_sectors=(((0, 0.5),),),
        )


class _WrongShapeK:
    def inverse(self, _s):
        return np.eye(2)


class _ComplexK:
    def inverse(self, _s):
        return np.array([[1.0 + 0.1j]])


@pytest.mark.parametrize(
    "model, message",
    [(_WrongShapeK(), r"exact \(1, 1\)"), (_ComplexK(), "must be real")],
)
def test_inverse_shape_and_reality_are_checked_at_evaluation(model, message):
    adapter = PartialWaveJMatrixAdapter(
        {(0, 0, 0, 0): model},
        channel_masses=MASSES,
        channel_sectors=(((0, 0),),),
    )
    with pytest.raises(ValueError, match=message):
        adapter.blocks(0.64)


def test_phase_space_and_subtraction_contract_is_explicit():
    incidence = (0, 0, 0, 0)
    model = {incidence: ConstantK([[1.0]])}
    common = dict(channel_masses=MASSES, channel_sectors=(((0, 0),),))
    with pytest.raises(ValueError, match="does not accept subtraction"):
        PartialWaveJMatrixAdapter(
            model, **common, subtraction_points={incidence: "threshold"}
        )
    with pytest.raises(ValueError, match="requires one subtraction point"):
        PartialWaveJMatrixAdapter(model, **common, phase_space="chew-mandelstam")
    with pytest.raises(ValueError, match="pseudothreshold"):
        PartialWaveJMatrixAdapter(
            model,
            **common,
            phase_space="chew-mandelstam",
            subtraction_points={incidence: ("k-pole", 0.0)},
        )


def test_cm_subtraction_point_is_zero_and_the_row_kernel_does_not_add_it_again():
    incidence = (0, 0, 0, 0)
    subtraction = 0.49
    pole_model = _KWithPole(subtraction)
    adapter = PartialWaveJMatrixAdapter(
        {incidence: pole_model},
        channel_masses=MASSES,
        channel_sectors=(((0, 0),),),
        phase_space="chew-mandelstam",
        subtraction_points={incidence: ("k-pole", subtraction)},
    )
    at_subtraction = adapter.blocks(subtraction)[0][0, 0]
    inverse_only = pole_model.inverse(subtraction)[0, 0] * np.sqrt(subtraction) / 2.0
    np.testing.assert_allclose(pole_model.inverse(subtraction), [[0.0]], atol=0.0)
    np.testing.assert_allclose(at_subtraction, inverse_only, atol=1e-14)

    frame = LatticeFrame(8, 1.0, (0, 0, 0))
    group = double_cover(little_group(frame.d))
    energy = np.sqrt(subtraction)
    actual = row_matrix(
        energy,
        frame,
        MASSES,
        None,
        adapter,
        group=group,
        irrep="A1g",
        channel_sectors=adapter.channel_sectors,
        channel_intrinsic_parities=(1,),
        selected_j_sectors=adapter.selected_j_sectors,
    )

    point = two_body_point(
        energy_lab_at=energy,
        mass1_at=MASSES[0][0],
        mass2_at=MASSES[0][1],
        frame=frame,
    )
    layout = normalise_channel_layout(
        None,
        channels=1,
        channel_sectors=adapter.channel_sectors,
        channel_intrinsic_parities=(1,),
        selected_j_sectors=adapter.selected_j_sectors,
    )
    projection, _, _ = channel_row_basis(layout, group, "A1g", 0)
    full = assemble_inverse(
        None,
        scale=frame.length_at / (2.0 * np.pi),
        j_matrices=adapter.blocks(point.s_at2),
        channels=1,
        channel_sectors=adapter.channel_sectors,
        channel_intrinsic_parities=(1,),
        selected_j_sectors=adapter.selected_j_sectors,
    )
    full -= mixed_box(
        point.q_squared,
        layout.channel_pairs[0],
        d=frame.d,
        gamma=point.gamma,
        alpha=point.alpha,
    )
    expected = projection.conj().T @ full @ projection
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10)
