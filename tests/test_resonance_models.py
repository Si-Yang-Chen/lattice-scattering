"""Independent acceptance tests for the Chung/P33 models and their JLS bridge."""

from __future__ import annotations

import numpy as np
import pytest

from lattice_scattering.amplitudes import (
    ChungBW,
    ConformalMapK,
    HattedKJLSAdapter,
    P33BW,
    build,
    registered,
)
from lattice_scattering.finite_volume import (
    WEIGHTING_SCALE,
    WEIGHTING_THRESHOLD,
    normalise_channel_layout,
    quantization_roots,
    row_matrix,
)
from lattice_scattering.finite_volume.zeta import harmonic_zeta_wide
from lattice_scattering.kinematics import LatticeFrame, two_body_point
from lattice_scattering.symmetry import double_cover, little_group


def _k2(s, mass1, mass2):
    return (s - (mass1 + mass2) ** 2) * (s - (mass1 - mass2) ** 2) / (4.0 * s)


def _f1_squared(k2, range_parameter):
    kr2 = k2 * range_parameter**2
    return 2.0 * kr2 / (1.0 + kr2)


@pytest.mark.parametrize("ell", [0, 1])
def test_chung_inverse_matches_independent_eq13_at_multiple_points(ell):
    parameters = dict(
        pole_mass=0.9,
        coupling=0.7,
        mass1=0.25,
        mass2=0.35,
        ell=ell,
    )
    range_parameter = 1.2
    if ell == 1:
        parameters["range_parameter"] = range_parameter
    model = ChungBW(**parameters)
    threshold_s = (parameters["mass1"] + parameters["mass2"]) ** 2
    pole_s = parameters["pole_mass"] ** 2
    k2_alpha = _k2(pole_s, parameters["mass1"], parameters["mass2"])
    f2_alpha = 1.0 if ell == 0 else _f1_squared(k2_alpha, range_parameter)

    for s in (threshold_s + 0.035, threshold_s + 0.12, pole_s + 0.04, pole_s + 0.16):
        k2 = _k2(s, parameters["mass1"], parameters["mass2"])
        barrier_squared = 1.0 if ell == 0 else _f1_squared(k2, range_parameter) / f2_alpha
        expected_inverse = (pole_s - s) / (parameters["coupling"] ** 2 * barrier_squared)
        np.testing.assert_allclose(model.inverse(s), [[expected_inverse]], rtol=2e-14, atol=2e-14)
        if s != pole_s:
            expected_khat = parameters["coupling"] ** 2 * barrier_squared / (pole_s - s)
            np.testing.assert_allclose(model.k(s), [[expected_khat]], rtol=2e-14, atol=2e-14)

    np.testing.assert_array_equal(model.inverse(pole_s), [[0.0]])
    with pytest.raises(ValueError, match="bare K pole"):
        model.k(pole_s)
    if ell == 0:
        for s in (threshold_s + 0.02, 0.6, 1.1):
            np.testing.assert_allclose(model.inverse(s), [[(pole_s - s) / parameters["coupling"] ** 2]])


def test_chung_complex_continuation_uses_the_same_analytic_equation():
    model = ChungBW(
        pole_mass=0.9,
        coupling=0.7,
        mass1=0.25,
        mass2=0.35,
        ell=1,
        range_parameter=1.2,
    )
    s = 0.72 + 0.11j
    pole_s = 0.9**2
    k2 = _k2(s, 0.25, 0.35)
    k2_alpha = _k2(pole_s, 0.25, 0.35)
    barrier_squared = _f1_squared(k2, 1.2) / _f1_squared(k2_alpha, 1.2)
    expected = (pole_s - s) / (0.7**2 * barrier_squared)
    actual = model.inverse(s)
    assert np.iscomplexobj(actual)
    np.testing.assert_allclose(actual, [[expected]], rtol=2e-14, atol=2e-14)


def test_chung_inverse_uses_analytic_limit_at_barrier_denominator_zero():
    model = ChungBW(
        pole_mass=1.4,
        coupling=0.7,
        mass1=0.5,
        mass2=0.5,
        ell=1,
        range_parameter=4.0,
    )
    # For equal masses, k^2=(s-4m^2)/4. At s=0.75 this gives k^2 R^2=-1.
    np.testing.assert_array_equal(model.inverse(0.75), [[0.0]])
    with pytest.raises(ValueError, match="Blatt-Weisskopf denominator"):
        model.k(0.75)


def test_p_wave_threshold_limits_breakpoints_and_regular_bare_poles():
    mass1 = mass2 = 0.3
    threshold_s = (mass1 + mass2) ** 2
    chung = ChungBW(
        pole_mass=0.9,
        coupling=0.8,
        mass1=mass1,
        mass2=mass2,
        ell=1,
        range_parameter=1.1,
    )
    p33 = P33BW(pole_mass=0.9, coupling=0.8, mass1=mass1, mass2=mass2)
    for model in (chung, p33):
        assert model.s_breakpoints().tolist() == [threshold_s]
        with pytest.raises(ValueError, match="threshold"):
            model.inverse(threshold_s)
        assert model.k(threshold_s).shape == (1, 1)
        np.testing.assert_array_equal(model.inverse(model.pole_mass**2), [[0.0]])
        with pytest.raises(ValueError, match="bare K pole"):
            model.k(model.pole_mass**2)

    chung_adapter = HattedKJLSAdapter(chung, ell=1, twice_S=0, twice_J=2)
    p33_adapter = HattedKJLSAdapter(p33, ell=1, twice_S=1, twice_J=3)
    chung_expected = (
        np.sqrt(threshold_s)
        / 2
        * (chung.pole_mass**2 - threshold_s)
        / chung.coupling**2
        * _f1_squared(_k2(chung.pole_mass**2, mass1, mass2), chung.range_parameter)
        / (2 * chung.range_parameter**2)
    )
    p33_expected = (
        np.sqrt(threshold_s)
        / 2
        * 12
        * np.pi
        * (p33.pole_mass**2 - threshold_s)
        / p33.coupling**2
    )
    np.testing.assert_allclose(chung_adapter.reduced_inverse(threshold_s), [[chung_expected]])
    np.testing.assert_allclose(p33_adapter.reduced_inverse(threshold_s), [[p33_expected]])


def test_p33_inverse_matches_both_paper_forms_and_threshold_scaling():
    model = P33BW(pole_mass=0.78, coupling=0.63, mass1=0.2, mass2=0.3)
    threshold_s = (model.mass1 + model.mass2) ** 2
    for s in (threshold_s + 0.03, 0.54, model.pole_mass**2 + 0.06, 0.78**2 + 0.15):
        k2 = _k2(s, model.mass1, model.mass2)
        k = np.sqrt(k2)
        ecm = np.sqrt(s)
        gamma = model.coupling**2 * k**3 / (6 * np.pi * s)
        rho = 2 * k / ecm
        khat_from_width = ecm * gamma / ((model.pole_mass**2 - s) * rho)
        khat_direct = model.coupling**2 * k2 / (12 * np.pi * (model.pole_mass**2 - s))
        inverse_expected = 12 * np.pi * (model.pole_mass**2 - s) / (model.coupling**2 * k2)
        np.testing.assert_allclose(model.k(s), [[khat_from_width]], rtol=3e-14, atol=3e-14)
        np.testing.assert_allclose(model.k(s), [[khat_direct]], rtol=3e-14, atol=3e-14)
        np.testing.assert_allclose(model.inverse(s), [[inverse_expected]], rtol=3e-14, atol=3e-14)

    epsilon = 1e-8
    k2_near = _k2(threshold_s + epsilon, model.mass1, model.mass2)
    np.testing.assert_allclose(
        model.k(threshold_s + epsilon) / k2_near,
        [[model.coupling**2 / (12 * np.pi * (model.pole_mass**2 - threshold_s))]],
        rtol=2e-7,
        atol=2e-7,
    )
    subthreshold_pole = P33BW(pole_mass=0.45, coupling=0.63, mass1=0.2, mass2=0.3)
    assert subthreshold_pole.pole_mass < subthreshold_pole.mass1 + subthreshold_pole.mass2
    assert np.all(np.isfinite(subthreshold_pole.inverse(threshold_s + 0.04)))
    unequal_masses = P33BW(pole_mass=0.8, coupling=0.63, mass1=0.2, mass2=0.35)
    assert unequal_masses.s_breakpoints().tolist() == [
        (unequal_masses.mass1 - unequal_masses.mass2) ** 2,
        (unequal_masses.mass1 + unequal_masses.mass2) ** 2,
    ]


def _paper_chung_khat(model, s):
    k2 = _k2(s, model.mass1, model.mass2)
    if model.ell == 0:
        barrier_squared = 1.0
    else:
        k2_alpha = _k2(model.pole_mass**2, model.mass1, model.mass2)
        barrier_squared = _f1_squared(k2, model.range_parameter) / _f1_squared(
            k2_alpha, model.range_parameter
        )
    return model.coupling**2 * barrier_squared / (model.pole_mass**2 - s)


def test_chung_and_p33_adapters_match_direct_paper_and_physical_phase_formulas():
    chung_s = ChungBW(pole_mass=0.9, coupling=0.7, mass1=0.25, mass2=0.35, ell=0)
    chung_p = ChungBW(
        pole_mass=0.9,
        coupling=0.7,
        mass1=0.25,
        mass2=0.35,
        ell=1,
        range_parameter=1.2,
    )
    p33 = P33BW(pole_mass=0.78, coupling=0.63, mass1=0.2, mass2=0.3)
    cases = (
        (chung_s, HattedKJLSAdapter(chung_s, ell=0, twice_S=0, twice_J=0), (0.395, 0.48, 0.76, 0.85, 0.97)),
        (chung_p, HattedKJLSAdapter(chung_p, ell=1, twice_S=0, twice_J=2), (0.395, 0.48, 0.76, 0.85, 0.97)),
        (p33, HattedKJLSAdapter(p33, ell=1, twice_S=1, twice_J=3), (0.28, 0.45, 0.5684, 0.6684, 0.7584)),
    )

    for model, adapter, invariant_energies_squared in cases:
        for s in invariant_energies_squared:
            k2 = _k2(s, model.mass1, model.mass2)
            k = np.sqrt(k2)
            ecm = np.sqrt(s)
            rho = 2 * k / ecm
            if isinstance(model, ChungBW):
                khat = _paper_chung_khat(model, s)
                if model.ell == 0:
                    paper_scale = ecm * (model.pole_mass**2 - s) / (2 * model.coupling**2)
                else:
                    k2_alpha = _k2(model.pole_mass**2, model.mass1, model.mass2)
                    barrier_squared = _f1_squared(k2, model.range_parameter) / _f1_squared(
                        k2_alpha, model.range_parameter
                    )
                    paper_scale = (
                        ecm * k2 * (model.pole_mass**2 - s)
                        / (2 * model.coupling**2 * barrier_squared)
                    )
            else:
                khat = model.coupling**2 * k2 / (
                    12 * np.pi * (model.pole_mass**2 - s)
                )
                # Equation (7), with the paper's 6*pi coefficient exposed.
                paper_scale = 6 * np.pi * ecm * (model.pole_mass**2 - s) / model.coupling**2

            tan_delta = rho * khat
            physical_inverse = k ** (2 * model.ell + 1) / tan_delta
            adapted = adapter.reduced_inverse(s)[0, 0]
            np.testing.assert_allclose(adapted, physical_inverse, rtol=1e-13, atol=1e-14)
            np.testing.assert_allclose(adapted, paper_scale, rtol=1e-13, atol=1e-14)


def test_adapted_chung_p_and_p33_approach_finite_threshold_limits():
    mass1 = mass2 = 0.3
    threshold_s = (mass1 + mass2) ** 2
    chung = ChungBW(
        pole_mass=0.9,
        coupling=0.8,
        mass1=mass1,
        mass2=mass2,
        ell=1,
        range_parameter=1.1,
    )
    p33 = P33BW(pole_mass=0.9, coupling=0.8, mass1=mass1, mass2=mass2)
    models_and_limits = (
        (
            chung,
            HattedKJLSAdapter(chung, ell=1, twice_S=0, twice_J=2),
            np.sqrt(threshold_s)
            / 2
            * (chung.pole_mass**2 - threshold_s)
            / chung.coupling**2
            * _f1_squared(_k2(chung.pole_mass**2, mass1, mass2), chung.range_parameter)
            / (2 * chung.range_parameter**2),
        ),
        (
            p33,
            HattedKJLSAdapter(p33, ell=1, twice_S=1, twice_J=3),
            np.sqrt(threshold_s)
            / 2
            * 12
            * np.pi
            * (p33.pole_mass**2 - threshold_s)
            / p33.coupling**2,
        ),
    )
    epsilons = (1e-3, 1e-5, 1e-7)
    for _model, adapter, analytic_limit in models_and_limits:
        sequence = np.asarray(
            [adapter.reduced_inverse(threshold_s + epsilon)[0, 0] for epsilon in epsilons]
        )
        errors = np.abs(sequence - analytic_limit)
        assert errors[2] < errors[1] < errors[0]
        np.testing.assert_allclose(sequence[-1], analytic_limit, rtol=5e-7, atol=1e-10)


@pytest.mark.parametrize("s", [(0.2 + 0.3) ** 2, (0.3 - 0.2) ** 2])
def test_p_wave_threshold_block_rejects_zero_momentum(s):
    model = P33BW(pole_mass=0.9, coupling=0.8, mass1=0.2, mass2=0.3)
    adapter = HattedKJLSAdapter(model, ell=1, twice_S=1, twice_J=3)

    with pytest.raises(ValueError, match="threshold weighting is singular at k\\^2=0"):
        adapter.threshold_block(s, scale=2.0)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.inf, np.nan, True, np.bool_(True), 1.0 + 0.0j, 1.0 + 0.2j])
def test_conformal_alpha_requires_finite_positive_real_scalar(bad):
    with pytest.raises(ValueError):
        ConformalMapK(
            coefficients=(0.4, -0.2, 0.1),
            alpha=bad,
            s0=1.4,
            s_adler=0.3,
            delta_kpi=0.15,
        )


@pytest.mark.parametrize("alpha", [1.0, 1.3])
def test_conformal_map_alpha_matches_independent_real_axis_expression(alpha):
    coefficients = (0.4, -0.2, 0.1)
    s0 = 1.4
    s_adler = 0.3
    delta = 0.15
    model = ConformalMapK(
        coefficients=coefficients,
        alpha=alpha,
        s0=s0,
        s_adler=s_adler,
        delta_kpi=delta,
    )

    def independent(s):
        y = ((s - delta) / (s + delta)) ** 2
        y0 = ((s0 - delta) / (s0 + delta)) ** 2
        omega = (np.sqrt(y) - alpha * np.sqrt(y0 - y)) / (
            np.sqrt(y) + alpha * np.sqrt(y0 - y)
        )
        inverse = sum(coefficient * omega**n for n, coefficient in enumerate(coefficients)) / (
            s - s_adler
        )
        return omega, inverse

    for s in (0.25, 0.6, 1.0, s0):
        expected_omega, expected_inverse = independent(s)
        np.testing.assert_allclose(model.omega(s), expected_omega, rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(model.inverse(s), [[expected_inverse]], rtol=2e-14, atol=2e-14)
    with pytest.raises(ValueError, match="s <= s0"):
        model.omega(s0 + 0.1)
    with pytest.raises(ValueError, match="Adler zero"):
        model.inverse(s_adler)


@pytest.mark.parametrize("field", ["pole_mass", "coupling", "mass1", "mass2"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, 1.0 + 0.0j, True, np.bool_(True)])
def test_resonance_scalar_parameters_reject_nonreal_nonfinite_and_boolean(field, bad):
    chung_parameters = dict(
        pole_mass=0.9,
        coupling=0.7,
        mass1=0.25,
        mass2=0.35,
        ell=0,
    )
    p33_parameters = {key: value for key, value in chung_parameters.items() if key != "ell"}
    chung_parameters[field] = bad
    p33_parameters[field] = bad
    with pytest.raises(ValueError):
        ChungBW(**chung_parameters)
    with pytest.raises(ValueError):
        P33BW(**p33_parameters)


@pytest.mark.parametrize("field", ["mass1", "mass2"])
def test_chung_and_p33_reject_negative_particle_masses(field):
    chung_parameters = dict(
        pole_mass=0.9,
        coupling=0.7,
        mass1=0.25,
        mass2=0.35,
        ell=0,
    )
    p33_parameters = {key: value for key, value in chung_parameters.items() if key != "ell"}
    chung_parameters[field] = -0.25
    p33_parameters[field] = -0.25
    with pytest.raises(ValueError):
        ChungBW(**chung_parameters)
    with pytest.raises(ValueError):
        P33BW(**p33_parameters)


@pytest.mark.parametrize(
    "bad", [-1.0, np.inf, np.nan, 1.0 + 0.0j, 1.0 + 0.2j, True, np.bool_(True)]
)
def test_chung_p_wave_rejects_invalid_range_parameter(bad):
    with pytest.raises(ValueError):
        ChungBW(
            pole_mass=0.9,
            coupling=0.7,
            mass1=0.25,
            mass2=0.35,
            ell=1,
            range_parameter=bad,
        )


def test_resonance_model_validation_and_registry_construction():
    with pytest.raises(ValueError, match="nonzero"):
        ChungBW(pole_mass=0.9, coupling=0, mass1=0.2, mass2=0.3, ell=0)
    with pytest.raises(ValueError, match="nonzero"):
        P33BW(pole_mass=0.9, coupling=0, mass1=0.2, mass2=0.3)
    with pytest.raises(ValueError, match="positive"):
        ChungBW(pole_mass=0.9, coupling=0.7, mass1=0.2, mass2=0.3, ell=1, range_parameter=0)
    with pytest.raises(ValueError, match="ell must be 0 or 1"):
        ChungBW(pole_mass=0.9, coupling=0.7, mass1=0.2, mass2=0.3, ell=2)
    with pytest.raises(ValueError, match=r"F_1\(k_alpha\)=0"):
        ChungBW(
            pole_mass=0.5,
            coupling=0.7,
            mass1=0.2,
            mass2=0.3,
            ell=1,
            range_parameter=1.0,
        )
    with pytest.raises(ValueError, match="branch point"):
        P33BW(pole_mass=0.5, coupling=0.7, mass1=0.2, mass2=0.3)
    with pytest.raises(ValueError, match="positive"):
        P33BW(pole_mass=-0.9, coupling=0.7, mass1=0.2, mass2=0.3)
    with pytest.raises(ValueError, match="positive"):
        ChungBW(pole_mass=0.9, coupling=0.7, mass1=0.0, mass2=0.3, ell=0)

    chung = build(
        "chung-bw",
        dict(pole_mass=0.9, coupling=0.7, mass1=0.2, mass2=0.3, ell=1, range_parameter=1.0),
    )
    p33 = build("p33-bw", dict(pole_mass=0.9, coupling=0.7, mass1=0.2, mass2=0.3))
    assert isinstance(chung, ChungBW)
    assert isinstance(p33, P33BW)
    assert registered()["chung-bw"]["required"] == ["pole_mass", "coupling", "mass1", "mass2", "ell"]
    assert registered()["p33-bw"]["required"] == ["pole_mass", "coupling", "mass1", "mass2"]


class _MalformedHattedK:
    ell = 0
    mass1 = 0.2
    mass2 = 0.3

    def __init__(self, matrix):
        self.matrix = matrix

    def k(self, s):
        return self.matrix

    def inverse(self, s):
        return self.matrix


def test_adapter_enforces_incidence_masses_and_exact_one_channel_matrix():
    valid = ChungBW(pole_mass=0.9, coupling=0.7, mass1=0.2, mass2=0.3, ell=0)
    with pytest.raises(ValueError, match="does not match model ell"):
        HattedKJLSAdapter(valid, ell=1, twice_S=0, twice_J=2)
    with pytest.raises(ValueError, match="triangle allowed"):
        HattedKJLSAdapter(valid, ell=0, twice_S=0, twice_J=2)
    with pytest.raises(ValueError, match="masses must match"):
        HattedKJLSAdapter(valid, ell=0, twice_S=0, twice_J=0, mass1=0.21)
    malformed = HattedKJLSAdapter(
        _MalformedHattedK(np.eye(2)), ell=0, twice_S=0, twice_J=0
    )
    with pytest.raises(ValueError, match=r"exact shape \(1, 1\)"):
        malformed.reduced_inverse(0.64)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("ell", 1.0),
        ("ell", -1),
        ("ell", np.nan),
        ("ell", np.inf),
        ("ell", 1 + 0j),
        ("ell", True),
        ("twice_S", -1),
        ("twice_J", -1),
        ("twice_J", np.inf),
    ],
)
def test_adapter_rejects_invalid_incidence_parameters(field, bad):
    model = ChungBW(pole_mass=0.9, coupling=0.7, mass1=0.2, mass2=0.3, ell=0)
    parameters = dict(ell=0, twice_S=0, twice_J=0)
    parameters[field] = bad
    with pytest.raises(ValueError):
        HattedKJLSAdapter(model, **parameters)


@pytest.mark.parametrize("field", ["mass1", "mass2"])
@pytest.mark.parametrize("bad", [-0.2, np.inf, np.nan, 0.2 + 0.1j, True])
def test_adapter_rejects_invalid_explicit_masses(field, bad):
    model = ChungBW(pole_mass=0.9, coupling=0.7, mass1=0.2, mass2=0.3, ell=0)
    parameters = dict(ell=0, twice_S=0, twice_J=0, mass1=0.2, mass2=0.3)
    parameters[field] = bad
    with pytest.raises(ValueError):
        HattedKJLSAdapter(model, **parameters)


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (np.asarray([[np.nan]]), "non-finite"),
        (np.asarray([[np.inf]]), "non-finite"),
        (np.asarray([[1.0 + 0.2j]]), "complex for a real supported point"),
    ],
)
def test_adapter_rejects_nonfinite_or_complex_inverse_outputs(matrix, message):
    malformed = HattedKJLSAdapter(
        _MalformedHattedK(matrix), ell=0, twice_S=0, twice_J=0
    )
    with pytest.raises(ValueError, match=message):
        malformed.reduced_inverse(0.64)


class _PhaseShiftHattedK:
    """Synthetic elastic model fixed only by an independent phase shift."""

    def __init__(self, mass1, mass2, delta):
        self.mass1 = mass1
        self.mass2 = mass2
        self.ell = 1
        self.delta = delta

    def inverse(self, s):
        k2 = _k2(s, self.mass1, self.mass2)
        if k2 <= 0:
            raise ValueError("phase-shift oracle is defined above threshold only")
        return np.asarray([[2 * np.sqrt(k2) / np.sqrt(s) / np.tan(self.delta)]])

    def k(self, s):
        return np.linalg.inv(self.inverse(s))


def test_adapter_recovers_independent_k_cubed_cot_delta_oracle():
    mass1 = mass2 = 0.3
    delta = 0.37
    model = _PhaseShiftHattedK(mass1, mass2, delta)
    adapter = HattedKJLSAdapter(model, ell=1, twice_S=0, twice_J=2)
    for s in (0.5, 0.8, 1.1):
        k = np.sqrt(_k2(s, mass1, mass2))
        expected_physical_inverse = k**3 / np.tan(delta)
        actual = adapter.reduced_inverse(s)[0, 0]
        np.testing.assert_allclose(actual, expected_physical_inverse, rtol=1e-13, atol=1e-14)

        frame = LatticeFrame(16, 1.0)
        q = k * frame.length_at / (2 * np.pi)
        np.testing.assert_allclose(
            (frame.length_at / (2 * np.pi)) ** 3 * actual,
            q**3 / np.tan(delta),
            rtol=1e-13,
            atol=1e-14,
        )


@pytest.mark.parametrize("q2", [0.2, 1.4])
def test_rest_t1u_row_matches_standard_scalar_p_wave_condition(q2):
    mass = 0.3
    frame = LatticeFrame(16, 1.0)
    lam = frame.length_at / (2 * np.pi)
    q = np.sqrt(q2)
    k = q / lam
    s = 4 * (mass**2 + k**2)
    energy = np.sqrt(s)
    delta = 0.37
    adapter = HattedKJLSAdapter(
        _PhaseShiftHattedK(mass, mass, delta), ell=1, twice_S=0, twice_J=2
    )
    group = double_cover(little_group(frame.d))
    irrep = next(rep.label for rep in group.irreps if rep.dimension == 3 and rep.label == "T1u")

    row = row_matrix(
        energy,
        frame,
        ((mass, mass),),
        ((1, 0),),
        {2: adapter},
        group=group,
        irrep=irrep,
        weighting=WEIGHTING_SCALE,
    )
    threshold_row = row_matrix(
        energy,
        frame,
        ((mass, mass),),
        ((1, 0),),
        {2: _threshold_callback(adapter.model, frame)},
        group=group,
        irrep=irrep,
        weighting=WEIGHTING_THRESHOLD,
    )
    z00 = harmonic_zeta_wide(q2, 0, d=(0, 0, 0), gamma=1.0, alpha=0.0).values[0]
    expected = q**3 / np.tan(delta) - q2 * z00 / np.pi**1.5
    assert row.shape == (1, 1)
    np.testing.assert_allclose(
        row, threshold_row, rtol=1e-11, atol=1e-12 * max(1.0, np.max(np.abs(row)))
    )
    np.testing.assert_allclose(row[0, 0], expected, rtol=1e-11, atol=1e-12 * max(1.0, abs(expected)))


def test_moving_p_wave_matches_pinned_historical_coupled_axial_p_entry():
    """Source-bound oracle for the old axial P diagonal; no historical import.

    This is the P-only entry obtained from
    ``coupled_axial.quantization_matrix`` at software commit
    ``5c562e3aace00ea74444b89abe65de6b0297c027`` (file blob
    ``cd5513b09bfc210c92e60d316386ba4484e59085``): that routine starts from
    ``-axial_sp_box`` and adds ``(L/2pi)^3 * p_reduced_inverse`` to the P
    diagonal. The pinned ``axial_sp_box`` defines that diagonal as
    ``(q2*Z00 + 2*Z20/sqrt(5))/(gamma*pi^(3/2))``. This test encodes that
    historical source contract directly and keeps the old module out of the
    production and test import graphs.
    """

    mass = 0.3
    delta = 0.39
    frame = LatticeFrame(24, 1.0, (0, 0, 1))
    lam = frame.length_at / (2 * np.pi)
    group = double_cover(little_group(frame.d))
    model = _PhaseShiftHattedK(mass, mass, delta)
    adapter = HattedKJLSAdapter(model, ell=1, twice_S=0, twice_J=2)

    for target_q2 in (0.35, 1.4):
        k_target = np.sqrt(target_q2) / lam
        s_target = 4 * (mass**2 + k_target**2)
        energy = np.sqrt(s_target + sum(momentum**2 for momentum in frame.momentum_at))
        point = two_body_point(
            energy_lab_at=energy, mass1_at=mass, mass2_at=mass, frame=frame
        )
        q2 = point.q_squared
        k = np.sqrt(point.k_squared_at2)
        adapted_r_phys = adapter.reduced_inverse(point.s_at2)[0, 0]
        np.testing.assert_allclose(
            adapted_r_phys, k**3 / np.tan(delta), rtol=1e-13, atol=1e-14
        )

        z00 = harmonic_zeta_wide(
            q2, 0, d=frame.d, gamma=point.gamma, alpha=point.alpha
        ).values[0]
        z20 = harmonic_zeta_wide(
            q2, 2, d=frame.d, gamma=point.gamma, alpha=point.alpha
        ).values[2]
        historical_p_entry = lam**3 * adapted_r_phys - (
            q2 * z00 + 2 * z20 / np.sqrt(5)
        ) / (point.gamma * np.pi**1.5)
        row = row_matrix(
            energy,
            frame,
            ((mass, mass),),
            ((1, 0),),
            {2: adapter},
            group=group,
            irrep="A1",
            weighting=WEIGHTING_SCALE,
        )
        assert row.shape == (1, 1)
        np.testing.assert_allclose(
            row[0, 0],
            historical_p_entry,
            rtol=1e-11,
            atol=1e-12 * max(1.0, abs(historical_p_entry)),
        )


def _scale_inverse(model, s):
    """Independent paper-form evaluation of the scale-weighted reduced inverse."""

    ecm = np.sqrt(s)
    if isinstance(model, ChungBW) and model.ell == 0:
        return ecm * (model.pole_mass**2 - s) / (2 * model.coupling**2)
    k2 = _k2(s, model.mass1, model.mass2)
    return ecm * k2 * model.inverse(s)[0, 0] / 2


def _threshold_callback(model, frame):
    lam = frame.length_at / (2 * np.pi)

    def evaluate(s):
        k2 = _k2(s, model.mass1, model.mass2)
        reduced_scale = _scale_inverse(model, s)
        return np.asarray([[(2 * np.sqrt(k2)) ** (2 * model.ell) * lam ** (2 * model.ell + 1) * reduced_scale]])

    return evaluate


def test_moving_chung_sp_mixing_has_equal_scale_and_threshold_rows_and_stable_roots():
    frame = LatticeFrame(24, 1.0, (0, 0, 1))
    masses = ((0.2, 0.3),)
    group = double_cover(little_group(frame.d))
    chung_s = ChungBW(pole_mass=0.75, coupling=0.5, mass1=0.2, mass2=0.3, ell=0)
    chung_p = ChungBW(
        pole_mass=0.82,
        coupling=0.65,
        mass1=0.2,
        mass2=0.3,
        ell=1,
        range_parameter=1.1,
    )
    scale_blocks = {
        0: HattedKJLSAdapter(chung_s, ell=0, twice_S=0, twice_J=0),
        2: HattedKJLSAdapter(chung_p, ell=1, twice_S=0, twice_J=2),
    }
    threshold_blocks = {0: _threshold_callback(chung_s, frame), 2: _threshold_callback(chung_p, frame)}
    selected_irrep = "A1"
    energy = 0.79

    relevant_rows = (("A1", 0), ("E", 0), ("E", 1))
    for irrep, row_index in relevant_rows:
        scale_row = row_matrix(
            energy,
            frame,
            masses,
            ((0, 0), (1, 0)),
            scale_blocks,
            group=group,
            irrep=irrep,
            row=row_index,
            weighting=WEIGHTING_SCALE,
        )
        threshold_row = row_matrix(
            energy,
            frame,
            masses,
            ((0, 0), (1, 0)),
            threshold_blocks,
            group=group,
            irrep=irrep,
            row=row_index,
            weighting=WEIGHTING_THRESHOLD,
        )
        np.testing.assert_allclose(scale_row, scale_row.conj().T, atol=1e-11, rtol=1e-11)
        if irrep == selected_irrep:
            assert scale_row.shape == (2, 2)
            assert abs(scale_row[0, 1]) > 1e-9
        np.testing.assert_allclose(
            scale_row,
            threshold_row,
            rtol=1e-11,
            atol=1e-12 * max(1.0, np.max(np.abs(scale_row))),
        )

    scale_row = row_matrix(
        energy,
        frame,
        masses,
        ((0, 0), (1, 0)),
        scale_blocks,
        group=group,
        irrep=selected_irrep,
        weighting=WEIGHTING_SCALE,
    )

    breakpoints = np.unique(np.concatenate((chung_s.s_breakpoints(), chung_p.s_breakpoints())))
    roots_40 = quantization_roots(
        (0.58, 0.99), frame, masses, ((0, 0), (1, 0)), scale_blocks,
        group=group, irrep=selected_irrep, samples=40, weighting=WEIGHTING_SCALE,
        breakpoints_at2=breakpoints,
    )
    roots_80 = quantization_roots(
        (0.58, 0.99), frame, masses, ((0, 0), (1, 0)), scale_blocks,
        group=group, irrep=selected_irrep, samples=80, weighting=WEIGHTING_SCALE,
        breakpoints_at2=breakpoints,
    )
    assert roots_40
    np.testing.assert_allclose(roots_40, roots_80, rtol=0.0, atol=2e-8)
    for root in roots_80:
        row = row_matrix(
            root, frame, masses, ((0, 0), (1, 0)), scale_blocks,
            group=group, irrep=selected_irrep, weighting=WEIGHTING_SCALE,
        )
        eigvals = np.linalg.eigvalsh(row)
        residual = np.min(np.abs(eigvals)) / max(1.0, np.linalg.norm(row, ord=2))
        assert residual < 2e-8


def test_p33_selected_j_has_only_s31_and_p33_and_equal_weighting_rows():
    frame = LatticeFrame(24, 1.0, (0, 0, 1))
    masses = ((0.2, 0.3),)
    sectors = ((0, 1), (1, 1))
    selected = {1: ((0, 0, 1),), 3: ((0, 1, 1),)}
    layout = normalise_channel_layout(
        None,
        channels=1,
        channel_sectors=(sectors,),
        channel_intrinsic_parities=(-1,),
        selected_j_sectors=selected,
    )
    assert layout.active_labels == selected
    assert layout.active_labels[1] == ((0, 0, 1),)
    assert layout.active_labels[3] == ((0, 1, 1),)
    assert layout.active[1] == ((0, 0),)
    assert layout.active[3] == ((0, 1),)
    assert (0, 1, 1) not in layout.active_labels[1]

    p33 = P33BW(pole_mass=0.82, coupling=0.7, mass1=0.2, mass2=0.3)
    p33_adapter = HattedKJLSAdapter(p33, ell=1, twice_S=1, twice_J=3)
    lam = frame.length_at / (2 * np.pi)
    scale_blocks = {
        1: lambda s: np.asarray([[0.5 * np.sqrt(s)]]),
        3: p33_adapter,
    }
    threshold_blocks = {
        1: lambda s: np.asarray([[lam * 0.5 * np.sqrt(s)]]),
        3: _threshold_callback(p33, frame),
    }
    group = double_cover(little_group(frame.d))
    kwargs = dict(
        group=group,
        irrep="G1",
        channel_sectors=(sectors,),
        channel_intrinsic_parities=(-1,),
        selected_j_sectors=selected,
    )
    for row_index in (0, 1):
        scale_row = row_matrix(
            0.86, frame, masses, None, scale_blocks, weighting=WEIGHTING_SCALE,
            row=row_index, **kwargs
        )
        threshold_row = row_matrix(
            0.86, frame, masses, None, threshold_blocks, weighting=WEIGHTING_THRESHOLD,
            row=row_index, **kwargs
        )
        np.testing.assert_allclose(scale_row, scale_row.conj().T, atol=1e-11, rtol=1e-11)
        np.testing.assert_allclose(
            scale_row,
            threshold_row,
            rtol=1e-11,
            atol=1e-12 * max(1.0, np.max(np.abs(scale_row))),
        )

    roots_40 = quantization_roots(
        (0.58, 0.99), frame, masses, None, scale_blocks,
        group=group, irrep="G1", samples=40, weighting=WEIGHTING_SCALE,
        breakpoints_at2=p33_adapter.s_breakpoints(),
        channel_sectors=(sectors,), channel_intrinsic_parities=(-1,),
        selected_j_sectors=selected,
    )
    roots_80 = quantization_roots(
        (0.58, 0.99), frame, masses, None, scale_blocks,
        group=group, irrep="G1", samples=80, weighting=WEIGHTING_SCALE,
        breakpoints_at2=p33_adapter.s_breakpoints(),
        channel_sectors=(sectors,), channel_intrinsic_parities=(-1,),
        selected_j_sectors=selected,
    )
    assert roots_40
    np.testing.assert_allclose(roots_40, roots_80, rtol=0.0, atol=2e-8)
    for root in roots_80:
        root_row = row_matrix(
            root, frame, masses, None, scale_blocks, group=group, irrep="G1",
            weighting=WEIGHTING_SCALE, channel_sectors=(sectors,),
            channel_intrinsic_parities=(-1,), selected_j_sectors=selected,
        )
        residual = np.min(np.abs(np.linalg.eigvalsh(root_row))) / max(
            1.0, np.linalg.norm(root_row, ord=2)
        )
        assert residual < 2e-8
