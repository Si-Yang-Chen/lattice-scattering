"""Synthetic equation and domain checks for the approved pair-B inverse models."""

from __future__ import annotations

import numpy as np
import pytest

from lattice_scattering.amplitudes import (
    RationalInverseK,
    ReducedERE,
    UnitaryChPTLO,
    amplitude_inverse,
    build,
    registered,
)
from lattice_scattering.amplitudes.models_extra import PolygonPcotdelta
from lattice_scattering.amplitudes.phase_space import (
    cm_real_axis,
    effective_inverse_k,
    physical_rho,
)
from lattice_scattering.finite_volume import harmonic_zeta_wide, row_matrix
from lattice_scattering.kinematics import LatticeFrame, two_body_point
from lattice_scattering.symmetry.groups import little_group


def _poly(coefficients, x):
    return sum(coefficient * x**power for power, coefficient in enumerate(coefficients))


@pytest.mark.parametrize(
    "numerator,denominator,s_ref,scale_squared,s",
    [
        ((0.4, -1.3), (1.0,), 0.2, 0.7, 0.9),
        ((1.2, -0.4), (1.0, 0.25), -0.3, 1.7, 0.8),
        ((-0.5, 0.1, 0.03), (2.0, -0.2), 0.1, 0.6, 1.1),
    ],
)
def test_rational_inverse_matches_independent_affine_and_rational_equations(
    numerator, denominator, s_ref, scale_squared, s
):
    model = RationalInverseK(
        numerator_coefficients=numerator,
        denominator_coefficients=denominator,
        s_ref=s_ref,
        scale_squared=scale_squared,
    )
    x = (s - s_ref) / scale_squared
    expected = _poly(numerator, x) / _poly(denominator, x)
    assert model.inverse(s).shape == (1, 1)
    assert model.inverse(s)[0, 0] == pytest.approx(expected, rel=1e-14)


def test_rational_inverse_handles_numerator_zero_and_reports_only_denominator_roots():
    model = RationalInverseK(
        numerator_coefficients=(1.0, 1.0),
        denominator_coefficients=(1.0, 0.5),
        s_ref=3.0,
        scale_squared=2.0,
    )

    # x=-1 makes N=0 and K infinite, while K^-1 itself remains regular.
    assert model.inverse(1.0)[0, 0] == 0.0
    with pytest.raises(ValueError, match="K has a pole"):
        model.k(1.0)
    # The denominator root is x=-2 -> s=-1. The numerator root is not a breakpoint.
    np.testing.assert_allclose(model.s_breakpoints(), [-1.0], rtol=0.0, atol=1e-14)
    with pytest.raises(ValueError, match=r"denominator D\(x\) vanishes"):
        model.inverse(-1.0)


def test_rational_inverse_validation_registry_and_complex_evaluation():
    with pytest.raises(ValueError, match="non-empty finite real sequence"):
        RationalInverseK(numerator_coefficients=(), denominator_coefficients=(1.0,))
    with pytest.raises(ValueError, match="non-empty finite real sequence"):
        RationalInverseK(numerator_coefficients=(1.0, np.inf), denominator_coefficients=(1.0,))
    with pytest.raises(ValueError, match="denominator_coefficients must not be identically zero"):
        RationalInverseK(numerator_coefficients=(1.0,), denominator_coefficients=(0.0, 0.0))
    with pytest.raises(ValueError, match="scale_squared must be positive"):
        RationalInverseK(
            numerator_coefficients=(1.0,), denominator_coefficients=(1.0,), scale_squared=0.0
        )

    params = {
        "numerator_coefficients": (1.0, 0.2),
        "denominator_coefficients": (1.0, -0.1),
        "s_ref": 0.3,
        "scale_squared": 0.8,
    }
    model = build("rational-inverse-k", params)
    assert isinstance(model, RationalInverseK)
    assert "rational-inverse-k" in registered()
    s = 0.6 + 0.04j
    x = (s - params["s_ref"]) / params["scale_squared"]
    expected = _poly(params["numerator_coefficients"], x) / _poly(
        params["denominator_coefficients"], x
    )
    assert model.inverse(s)[0, 0] == pytest.approx(expected, rel=1e-14)


@pytest.mark.parametrize("phase_space", ["simple", "chew-mandelstam"])
def test_rational_inverse_effective_inverse_conventions(phase_space):
    masses = (0.23, 0.41)
    model = RationalInverseK(
        numerator_coefficients=(1.4, -0.15),
        denominator_coefficients=(1.0, 0.1),
        s_ref=0.2,
        scale_squared=0.9,
    )
    threshold = sum(masses) ** 2
    inverse_at_threshold = model.inverse(threshold)[0, 0]
    if phase_space == "simple":
        np.testing.assert_allclose(
            effective_inverse_k(model, (masses,), phase_space, threshold),
            [[inverse_at_threshold]],
            rtol=0.0,
            atol=1e-14,
        )
    else:
        # The zero subtraction is the threshold-subtracted convention I(s_th)=0.
        assert cm_real_axis(threshold, *masses, subtraction=0.0) == 0.0
        np.testing.assert_allclose(
            effective_inverse_k(
                model, (masses,), phase_space, threshold, subtractions=(0.0,)
            ),
            [[inverse_at_threshold]],
            rtol=0.0,
            atol=1e-14,
        )
        s = threshold + 0.17
        expected = model.inverse(s)[0, 0] + cm_real_axis(s, *masses) + 1j * physical_rho(
            s, *masses
        )
        np.testing.assert_allclose(
            effective_inverse_k(model, (masses,), phase_space, s, subtractions=(0.0,)),
            [[expected.real]],
            rtol=1e-13,
            atol=1e-14,
        )


def test_reduced_s_wave_matches_polygon_inverse_above_and_below_threshold():
    masses = (0.25, 0.4)
    coefficients = (1.3, -0.6, 0.08)
    reduced = ReducedERE(mass1=masses[0], mass2=masses[1], ell=0, coefficients=coefficients)
    polygon = PolygonPcotdelta(
        coefficients=coefficients, mass1=masses[0], mass2=masses[1]
    )
    # Both energies are above the pseudothreshold; the first is below the physical threshold.
    for s in (0.2, 0.78):
        np.testing.assert_allclose(reduced.inverse(s), polygon.inverse(s), rtol=1e-14, atol=1e-14)


def test_reduced_p_wave_ere_and_threshold_view():
    masses = (0.3, 0.47)
    scattering_volume_inverse = 2.2
    effective_range = -0.8
    model = ReducedERE(
        mass1=masses[0],
        mass2=masses[1],
        ell=1,
        coefficients=(scattering_volume_inverse, 0.5 * effective_range),
    )
    threshold = sum(masses) ** 2
    s_below, s_above = 0.3, 1.1
    for s in (s_below, threshold, s_above):
        k2 = (s - threshold) * (s - (masses[0] - masses[1]) ** 2) / (4.0 * s)
        expected = scattering_volume_inverse + 0.5 * effective_range * k2
        assert model.reduced(s) == pytest.approx(expected, rel=1e-14, abs=1e-14)
    assert model.reduced(threshold) == pytest.approx(scattering_volume_inverse)
    with pytest.raises(ValueError, match="finite reduced ERE view"):
        model.inverse(threshold)

    # When the first ell coefficients vanish, the hatted inverse has a removable limit.
    removable = ReducedERE(mass1=masses[0], mass2=masses[1], ell=1, coefficients=(0.0, 1.5))
    expected_limit = 2.0 * 1.5 / np.sqrt(threshold)
    assert removable.inverse(threshold)[0, 0] == pytest.approx(expected_limit)


def test_reduced_p_wave_matches_independent_phase_shift_normalization_oracle():
    mass1, mass2, s = 0.31, 0.46, 1.24
    energy = np.sqrt(s)
    k2 = (s - (mass1 + mass2) ** 2) * (s - (mass1 - mass2) ** 2) / (4.0 * s)
    momentum = np.sqrt(k2)
    delta = 0.63
    cot_delta = 1.0 / np.tan(delta)
    physical_reduced_inverse = momentum**3 * cot_delta
    physical_hatted_inverse = 2.0 * momentum / energy * cot_delta

    model = ReducedERE(
        mass1=mass1,
        mass2=mass2,
        ell=1,
        coefficients=(physical_reduced_inverse,),
    )
    # The expected values use only kinematics and the chosen phase shift, not the model output.
    assert model.reduced(s) == pytest.approx(physical_reduced_inverse, rel=1e-14)
    assert model.inverse(s)[0, 0] == pytest.approx(physical_hatted_inverse, rel=1e-14)
    assert (energy / 2.0) * momentum**2 * physical_hatted_inverse == pytest.approx(
        physical_reduced_inverse, rel=1e-14
    )


def test_reduced_p_wave_rest_frame_t1u_row_matches_direct_z00_scalar_oracle():
    """Compare a complete JLS P-wave row with the independent scalar Luescher form."""
    mass1, mass2 = 0.3, 0.4
    frame = LatticeFrame(spatial_sites=24, anisotropy=1.0, d=(0, 0, 0))
    length_factor = frame.length_at / (2.0 * np.pi)
    ere_coefficients = (0.004, -0.001)
    model = ReducedERE(
        mass1=mass1,
        mass2=mass2,
        ell=1,
        coefficients=ere_coefficients,
    )
    rest_group = little_group(frame.d)
    assert rest_group.name == "Oh"
    # Confirm the fixed-product Oh catalog label; this is the rest-frame T1u row.
    assert rest_group.irrep("T1u").label == "T1u"

    j_callbacks = {
        2: lambda s_at2: np.asarray([[model.jls_scale(s_at2)]], dtype=float)
    }
    for q_squared in (0.15, 0.82):
        assert 0.0 < q_squared < 1.0  # Positive, below the first nonzero free pole.
        k_squared = q_squared / length_factor**2
        momentum = np.sqrt(k_squared)
        energy_cm = np.sqrt(mass1**2 + k_squared) + np.sqrt(mass2**2 + k_squared)
        point = two_body_point(
            energy_lab_at=energy_cm,
            mass1_at=mass1,
            mass2_at=mass2,
            frame=frame,
        )
        assert point.q_squared == pytest.approx(q_squared, rel=2e-14, abs=2e-15)

        ere_r = ere_coefficients[0] + ere_coefficients[1] * k_squared
        cot_delta = ere_r / momentum**3
        z00 = harmonic_zeta_wide(
            q_squared,
            0,
            d=(0, 0, 0),
            gamma=1.0,
            alpha=0.0,
        ).values[0].real
        scalar_oracle = q_squared**1.5 * cot_delta - q_squared * z00 / np.pi**1.5

        jls_row = row_matrix(
            energy_cm,
            frame,
            ((mass1, mass2),),
            ((1, 0),),
            j_callbacks,
            group=rest_group,
            irrep="T1u",
            row=0,
            weighting="scale",
        )
        assert jls_row.shape == (1, 1)
        np.testing.assert_allclose(
            jls_row[0, 0],
            scalar_oracle,
            rtol=1e-11,
            atol=1e-12 * max(1.0, abs(scalar_oracle)),
        )


@pytest.mark.parametrize("ell", [-1, 1.5, True])
def test_reduced_ere_rejects_invalid_ell(ell):
    params = {"mass1": 0.3, "mass2": 0.4, "ell": 1, "coefficients": (1.0,)}
    params["ell"] = ell
    with pytest.raises(ValueError):
        ReducedERE(**params)


def test_reduced_ere_rejects_invalid_masses_and_empty_coefficients():
    with pytest.raises(ValueError, match="mass1 must be positive"):
        ReducedERE(mass1=0.0, mass2=0.4, ell=0, coefficients=(1.0,))
    with pytest.raises(ValueError, match="non-empty finite real sequence"):
        ReducedERE(mass1=0.3, mass2=0.4, ell=1, coefficients=())
    with pytest.raises(ValueError, match="non-empty finite real sequence"):
        ReducedERE(mass1=0.3, mass2=0.4, ell=1, coefficients=(1.0 + 0.2j,))


def _direct_chpt_inverse(mpi, md, decay_constant, alpha, mu, s):
    polynomial = 3.0 * s**2 - 2.0 * s * (md**2 + mpi**2) - (md**2 - mpi**2) ** 2
    v0 = -polynomial / (4.0 * s * decay_constant**2)
    subtraction_constant = alpha / np.pi + (2.0 / np.pi) * (
        md / (mpi + md) * np.log(md / mpi) + np.log(mpi / mu)
    )
    return 1.0 / (-v0 / (16.0 * np.pi)) + subtraction_constant


@pytest.mark.parametrize(
    "mpi,md,decay_constant,alpha,mu,s",
    [
        (0.24, 0.73, 0.12, -0.2, 0.31, 1.3),
        (0.29, 0.91, 0.17, 0.4, 0.38, 2.1),
        (0.36, 1.12, 0.2, 0.1, 0.44, 1.8),
    ],
)
def test_unitarized_chpt_inverse_matches_direct_source_formula(
    mpi, md, decay_constant, alpha, mu, s
):
    model = UnitaryChPTLO(
        pion_mass=mpi,
        heavy_mass=md,
        decay_constant=decay_constant,
        alpha=alpha,
        mu=mu,
    )
    expected = _direct_chpt_inverse(mpi, md, decay_constant, alpha, mu, s)
    assert model.inverse(s)[0, 0] == pytest.approx(expected, rel=2e-14, abs=1e-14)


def test_unitarized_chpt_threshold_subtraction_poles_and_registry():
    params = {
        "pion_mass": 0.27,
        "heavy_mass": 0.82,
        "decay_constant": 0.14,
        "alpha": -0.15,
        "mu": 0.33,
    }
    model = build("unitarized-chpt-lo", params)
    assert isinstance(model, UnitaryChPTLO)
    assert "unitarized-chpt-lo" in registered()
    threshold = (params["pion_mass"] + params["heavy_mass"]) ** 2
    assert cm_real_axis(threshold, params["pion_mass"], params["heavy_mass"]) == 0.0
    np.testing.assert_allclose(model.t_inverse(threshold), model.inverse(threshold), atol=1e-14)

    s = threshold + 0.11
    expected = model.inverse(s) + np.asarray(
        [[cm_real_axis(s, params["pion_mass"], params["heavy_mass"])]]
    )
    np.testing.assert_allclose(model.t_inverse(s), expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(
        amplitude_inverse(
            model,
            ((params["pion_mass"], params["heavy_mass"]),),
            "chew-mandelstam",
            s,
            subtractions=(0.0,),
        ),
        expected,
        rtol=1e-14,
        atol=1e-14,
    )

    with pytest.raises(ValueError, match="undefined at s=0"):
        model.inverse(0.0)
    breakpoints = model.s_breakpoints()
    assert len(breakpoints) == 2
    for point in breakpoints:
        with pytest.raises(ValueError, match=r"denominator D\(x\) vanishes"):
            model.inverse(float(point))


@pytest.mark.parametrize(
    "field,value",
    [
        ("pion_mass", 0.0),
        ("heavy_mass", np.inf),
        ("decay_constant", -0.1),
        ("mu", 0.0),
        ("alpha", np.nan),
    ],
)
def test_unitarized_chpt_validates_masses_and_parameters(field, value):
    params = {
        "pion_mass": 0.27,
        "heavy_mass": 0.82,
        "decay_constant": 0.14,
        "alpha": -0.15,
        "mu": 0.33,
    }
    params[field] = value
    with pytest.raises(ValueError):
        UnitaryChPTLO(**params)


@pytest.mark.parametrize(
    "model_name,field,value",
    [
        ("rational", "scale_squared", True),
        ("reduced", "mass1", True),
        ("chpt", "decay_constant", True),
        ("chpt", "alpha", False),
    ],
)
@pytest.mark.parametrize("use_numpy_bool", [False, True], ids=["python-bool", "numpy-bool"])
def test_shared_scalar_validator_rejects_python_and_numpy_booleans(
    model_name, field, value, use_numpy_bool
):
    scalar = np.bool_(value) if use_numpy_bool else value
    if model_name == "rational":
        with pytest.raises(ValueError, match=field):
            RationalInverseK(
                numerator_coefficients=(1.0,),
                denominator_coefficients=(1.0,),
                scale_squared=scalar,
            )
    elif model_name == "reduced":
        with pytest.raises(ValueError, match=field):
            ReducedERE(mass1=scalar, mass2=0.4, ell=0, coefficients=(1.0,))
    else:
        params = {
            "pion_mass": 0.27,
            "heavy_mass": 0.82,
            "decay_constant": 0.14,
            "alpha": -0.15,
            "mu": 0.33,
        }
        params[field] = scalar
        with pytest.raises(ValueError, match=field):
            UnitaryChPTLO(**params)


def test_unitarized_chpt_fit_parameterization_varies_only_f_and_alpha():
    fixed = {"pion_mass": 0.25, "heavy_mass": 0.8, "mu": 0.32}
    base = UnitaryChPTLO(**fixed, decay_constant=0.13, alpha=-0.2)
    changed_f = UnitaryChPTLO(**fixed, decay_constant=0.16, alpha=-0.2)
    changed_alpha = UnitaryChPTLO(**fixed, decay_constant=0.13, alpha=0.35)
    s = 1.4
    constant = -0.2 / np.pi + (2.0 / np.pi) * (
        fixed["heavy_mass"] / (fixed["pion_mass"] + fixed["heavy_mass"])
        * np.log(fixed["heavy_mass"] / fixed["pion_mass"])
        + np.log(fixed["pion_mass"] / fixed["mu"])
    )
    base_varying = base.inverse(s)[0, 0] - constant
    f_varying = changed_f.inverse(s)[0, 0] - constant
    assert f_varying / base_varying == pytest.approx((0.16 / 0.13) ** 2, rel=2e-14)
    assert changed_alpha.inverse(s)[0, 0] - base.inverse(s)[0, 0] == pytest.approx(
        (0.35 - (-0.2)) / np.pi, rel=2e-14
    )
