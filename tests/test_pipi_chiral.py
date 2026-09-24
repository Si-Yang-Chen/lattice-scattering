"""Independent synthetic acceptance checks for the 1107.5023 pi-pi amplitude."""

from cmath import log as complex_log
from math import isfinite, log, pi, sqrt

import numpy as np
import pytest

from lattice_scattering.amplitudes import (
    PipiChiralDomainError,
    PipiChiralPhaseObservablePredictor,
    PipiNLOChiral,
    build,
    pipi_chiral_constants_from_ell_at_fpi,
    registered,
)


def _source_reference_t(k, m_pi, f_pi, C1, C2, C4):
    """Direct Eq. (10) evaluator with no product helpers or threshold rewrites."""
    k2 = k * k
    m2 = m_pi * m_pi
    f2 = f_pi * f_pi
    t_lo = -(m2 + 2.0 * k2) / (8.0 * pi * f2)
    if k == 0.0:
        # The independent combined limits are L_t/u=-2 and
        # L_t^2(1+13m_pi^2/(12k^2))=13/3.
        t_nlo = -m2**2 / f2**2 * (C1 + 3.0 * log(m2 / f2) / (128.0 * pi**3))
        return t_lo, complex(t_nlo)

    u = sqrt(k2 / (k2 + m2))
    v = sqrt((k2 + m2) / k2)
    Ls = complex_log((u - 1.0) / (u + 1.0))
    Lt = log((v - 1.0) / (v + 1.0))
    t_nlo = -m2**2 / f2**2 * (C1 - 31.0 / (384.0 * pi**3))
    t_nlo += (k2 / f2) * (m2 / f2) * (
        301.0 / (1152.0 * pi**3) - C2 / (128.0 * pi**2) - C1 / 2.0
    )
    t_nlo += (k2**2 / f2**2) * (
        14.0 / (45.0 * pi**3)
        - (19.0 * C1 / 8.0 - 9.0 * C2 / (512.0 * pi**2) + 216.0 * pi * C4)
    )
    t_nlo -= (
        (3.0 * m2**2 / 32.0 + 5.0 * m2 * k2 / 12.0 + 5.0 * k2**2 / 9.0)
        / (4.0 * pi**3 * f2**2)
        * log(m2 / f2)
    )
    t_nlo += (
        (m2**2 / 4.0 + m2 * k2 + k2**2)
        / (16.0 * pi**3 * f2**2)
        * u
        * Ls
    )
    t_nlo += (
        (3.0 * m2**2 / 16.0 + 7.0 * m2 * k2 / 9.0 + 11.0 * k2**2 / 18.0)
        / (8.0 * pi**3 * f2**2)
        * v
        * Lt
    )
    t_nlo -= (
        m2**2
        / (128.0 * pi**3 * f2**2)
        * (1.0 + 13.0 * m2 / (12.0 * k2))
        * Lt**2
    )
    return t_lo, t_nlo


def _reference_kcot_over_mpi(s_phys, m_pi, t_lo, t_nlo):
    """Eq. (20), independent of the product's s-to-k and inverse adapters."""
    k = sqrt(s_phys / 4.0 - m_pi**2)
    return (
        sqrt(1.0 + k * k / m_pi**2)
        * (1.0 / t_lo - t_nlo / t_lo**2)
        + 1j * k / m_pi
    )


def _model():
    return PipiNLOChiral(pion_mass=0.31, decay_constant=0.47, C1=0.12, C2=-0.21, C4=0.004)


@pytest.mark.parametrize("k", [0.025, 0.14, 0.43])
def test_complete_amplitude_matches_independent_eq10_and_elastic_branch(k):
    model = _model()
    expected_lo, expected_nlo = _source_reference_t(
        k, model.m_pi, model.f_pi, model.C1, model.C2, model.C4
    )

    assert model.t_LO(k) == pytest.approx(expected_lo, rel=2e-14, abs=2e-15)
    assert model.t_NLO(k) == pytest.approx(expected_nlo, rel=3e-13, abs=3e-15)
    u = sqrt(k * k / (k * k + model.m_pi**2))
    assert model.t_NLO(k).imag == pytest.approx(u * model.t_LO(k) ** 2, rel=3e-13, abs=3e-15)
    assert model.t_NLO(k).imag > 0.0  # L_s uses Log(x+i0)=ln|x|+i*pi.


def test_eq9_eq20_p_cot_and_registered_inverse_have_no_extra_phase_space_factor():
    model = _model()
    for k in (0.0, 0.12, 0.35):
        s_phys = 4.0 * (model.m_pi**2 + k**2)
        t_lo, t_nlo = _source_reference_t(k, model.m_pi, model.f_pi, model.C1, model.C2, model.C4)
        expected_kcot_over_mpi = _reference_kcot_over_mpi(s_phys, model.m_pi, t_lo, t_nlo)
        p_cot = model.p_cot_delta(s_phys)
        assert expected_kcot_over_mpi.imag == pytest.approx(0.0, abs=2e-13)
        assert p_cot / model.m_pi == pytest.approx(expected_kcot_over_mpi.real, rel=5e-13, abs=2e-13)

        # From Eq. (9), t^{-1}=rho(cot(delta)-i), so the reduced inverse is
        # t_LO^{-1}-t_NLO/t_LO^2+i*2k/sqrt(s_phys), with rho=2k/sqrt(s_phys).
        rho = 2.0 * k / sqrt(s_phys)
        independent_reduced_inverse = (1.0 / t_lo - t_nlo / t_lo**2 + 1j * rho).real
        protocol_inverse = float(model.inverse(s_phys)[0, 0])
        assert protocol_inverse == pytest.approx(independent_reduced_inverse, rel=5e-13, abs=2e-13)
        assert protocol_inverse == pytest.approx(2.0 * p_cot / sqrt(s_phys), rel=2e-14, abs=2e-14)
        # The scale-weighted ell=0 reduced inverse is p cot(delta) itself.
        assert 0.5 * sqrt(s_phys) * protocol_inverse == pytest.approx(p_cot, rel=2e-14, abs=2e-14)


def test_threshold_limit_and_eq15_a_b_c_coefficients():
    model = _model()
    z = model.m_pi**2 / model.f_pi**2
    log_z = log(z)
    expected_mpi_a = z / (8.0 * pi) + z**2 * model.C1 + 3.0 * z**2 * log_z / (128.0 * pi**3)
    expected_mpi2_b = -z / (4.0 * pi) - z**2 * (
        model.C1 / 2.0 + model.C2 / (128.0 * pi**2) + 5.0 * log_z / (48.0 * pi**3)
    )
    expected_mpi4_c = -z**2 * (
        19.0 * model.C1 / 8.0
        - 9.0 * model.C2 / (512.0 * pi**2)
        + 216.0 * pi * model.C4
        + 5.0 * log_z / (36.0 * pi**3)
    )

    threshold = model.two_pion_threshold_s_phys
    t0 = (model.t_LO(0.0) + model.t_NLO(0.0)).real
    assert -t0 == pytest.approx(expected_mpi_a, rel=3e-13, abs=3e-14)
    assert isfinite(model.p_cot_delta(threshold))

    # Eq. (12) expanded in x=k^2/m_pi^2 is -m_pi*a + (m_pi^2*b)x
    # + (m_pi^4*c)x^2 + O(x^3). Check the independently printed Eq. (15)
    # coefficients at two scales so the omitted remainder falls as x^3.
    errors = []
    for q in (0.08, 0.04):
        x = q * q
        k = q * model.m_pi
        actual = (model.t_LO(k) + model.t_NLO(k)).real
        truncated = -expected_mpi_a + expected_mpi2_b * x + expected_mpi4_c * x**2
        errors.append(abs(actual - truncated))
    assert errors[0] > 0.0
    assert errors[1] < errors[0] / 40.0


def test_eq11_ell_conversion_matches_an_independent_formula_once():
    ell = (0.021, -0.014, 0.008, 0.003)
    l1, l2, l3, l4 = ell
    expected = (
        -(4.0 * l1 + 4.0 * l2 + l3 - l4) / (2.0 * pi) - 1.0 / (128.0 * pi**3),
        32.0 * pi * (12.0 * l1 + 4.0 * l2 + 7.0 * l3 - 3.0 * l4) + 31.0 / (6.0 * pi),
        (212.0 * l1 + 40.0 * l2 + 123.0 * l3 - 69.0 * l4) / (5184.0 * pi**2)
        + 701.0 / (622080.0 * pi**4),
    )
    assert pipi_chiral_constants_from_ell_at_fpi(
        ell1=l1, ell2=l2, ell3=l3, ell4=l4
    ) == pytest.approx(expected, rel=2e-15, abs=2e-15)
    converted = PipiNLOChiral.from_ell_at_fpi(
        pion_mass=0.31,
        decay_constant=0.47,
        ell1=l1,
        ell2=l2,
        ell3=l3,
        ell4=l4,
    )
    assert (converted.C1, converted.C2, converted.C4) == pytest.approx(expected)


def test_dimensionless_units_and_ordered_observable_adapter():
    model = _model()
    scale = 3.7
    scaled = PipiNLOChiral(
        pion_mass=scale * model.m_pi,
        decay_constant=scale * model.f_pi,
        C1=model.C1,
        C2=model.C2,
        C4=model.C4,
    )
    q_values = (0.32, 0.11, 0.24)  # Intentionally out of energy order.
    s_phys = np.asarray([4.0 * model.m_pi**2 * (1.0 + q**2) for q in q_values])
    scaled_s_phys = scale**2 * s_phys
    for s, s_scaled in zip(s_phys, scaled_s_phys):
        assert scaled.p_cot_delta(s_scaled) == pytest.approx(scale * model.p_cot_delta(s), rel=2e-13)
        assert scaled.inverse(s_scaled) == pytest.approx(model.inverse(s), rel=2e-13)

    predictor = PipiChiralPhaseObservablePredictor(model)
    predictions = predictor.predict(s_phys)
    independently_ordered = np.asarray(
        [model.dimensionless_k_cot_delta(s) for s in s_phys], dtype=float
    )
    assert predictions == pytest.approx(independently_ordered, rel=2e-14, abs=2e-14)
    assert predictor(s_phys[::-1]) == pytest.approx(predictions[::-1], rel=2e-14, abs=2e-14)
    assert PipiChiralPhaseObservablePredictor(scaled).predict(scaled_s_phys) == pytest.approx(
        predictions, rel=2e-13, abs=2e-13
    )


def test_registry_builds_the_amplitude_and_exposes_exact_breakpoints():
    parameters = {"pion_mass": 0.31, "decay_constant": 0.47, "C1": 0.12, "C2": -0.21, "C4": 0.004}
    assert "pipi-nlo-chiral" in registered()
    assert registered()["pipi-nlo-chiral"]["required"] == [
        "pion_mass",
        "decay_constant",
        "C1",
        "C2",
        "C4",
    ]
    model = build("pipi-nlo-chiral", parameters)
    m2 = model.m_pi**2
    assert model.s_breakpoints() == pytest.approx([0.0, 2.0 * m2, 4.0 * m2, 16.0 * m2])
    assert model.crossed_channel_branch_endpoint_s_phys == 0.0
    assert model.t_lo_zero_s_phys == 2.0 * m2
    assert model.two_pion_threshold_s_phys == 4.0 * m2
    assert model.four_pion_threshold_s_phys == 16.0 * m2

    with pytest.raises(PipiChiralDomainError, match="branch endpoint"):
        model.p_cot_delta(model.crossed_channel_branch_endpoint_s_phys)
    with pytest.raises(ZeroDivisionError, match="t_LO=0"):
        model.p_cot_delta(model.t_lo_zero_s_phys)
    assert isfinite(model.p_cot_delta(model.two_pion_threshold_s_phys))
    with pytest.raises(PipiChiralDomainError, match="below the four-pion threshold"):
        model.inverse(model.four_pion_threshold_s_phys)
    with pytest.raises(PipiChiralDomainError, match="below the four-pion threshold"):
        model.inverse(np.nextafter(model.four_pion_threshold_s_phys, np.inf))
    assert isfinite(model.inverse(np.nextafter(model.four_pion_threshold_s_phys, 0.0))[0, 0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pion_mass": 0.0},
        {"pion_mass": -0.1},
        {"pion_mass": np.nan},
        {"decay_constant": 0.0},
        {"decay_constant": np.inf},
        {"C1": np.inf},
        {"C2": 1.0 + 1.0j},
    ],
)
def test_invalid_masses_decay_constant_and_lecs_are_rejected(kwargs):
    parameters = {"pion_mass": 0.31, "decay_constant": 0.47, "C1": 0.12, "C2": -0.21, "C4": 0.004}
    parameters.update(kwargs)
    with pytest.raises(ValueError):
        PipiNLOChiral(**parameters)


def test_real_axis_branch_domain_elastic_boundary_and_true_poles_are_explicit():
    model = _model()
    with pytest.raises(PipiChiralDomainError, match="k >= 0"):
        model.t_LO(-0.1)
    with pytest.raises(PipiChiralDomainError, match="k >= 0"):
        model.t_NLO(-0.1)
    with pytest.raises(ValueError, match="finite real scalar"):
        model.t_NLO(0.1j)
    with pytest.raises(PipiChiralDomainError, match="branch endpoint"):
        model.p_cot_delta(0.0)
    with pytest.raises(PipiChiralDomainError, match="s_phys > 0"):
        model.p_cot_delta(-0.1)
    with pytest.raises(PipiChiralDomainError, match="subthreshold logarithm continuation is not specified"):
        model.p_cot_delta(3.0 * model.m_pi**2)
    with pytest.raises(PipiChiralDomainError, match=r"starts at s_phys=4 m_pi\^2"):
        model.inverse(3.0 * model.m_pi**2)
    with pytest.raises(PipiChiralDomainError, match="four-pion threshold"):
        model.validate_elastic_finite_volume_domain(16.0 * model.m_pi**2)


def test_unstated_subthreshold_log_branch_is_not_silently_continued():
    model = _model()
    # The frozen spec fixes Log(x+i0) for the physical elastic axis but gives
    # no subthreshold prescription; metadata still exposes its real cut ends.
    assert model.s_breakpoints()[0] == 0.0
    assert model.s_breakpoints()[2] == model.two_pion_threshold_s_phys
    with pytest.raises(PipiChiralDomainError, match="continuation is not specified"):
        model.p_cot_delta(3.5 * model.m_pi**2)
