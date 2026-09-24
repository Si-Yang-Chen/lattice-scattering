"""Unit tests for the v2 core data model (``v2/core/model.py``)."""
from __future__ import annotations

import numpy as np
import pytest

from lattice_scattering.core.model import (
    Channel,
    Ensemble,
    ExchangeRule,
    Frame,
    Hadron,
    Level,
    Spectrum,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _hadron(label="X", twice_spin=0, parity=-1, mass=0.3):
    return Hadron(label=label, twice_spin=twice_spin, parity=parity, mass_at=mass)


def _spectrum():
    h = _hadron()
    ch = Channel(label="XX", hadron_a=h, hadron_b=h, exchange=ExchangeRule.IDENTICAL)
    fr = Frame(id="L24_d000", spatial_sites=24, anisotropy=3.444, d=(0, 0, 0))
    levels = (
        Level(id="L1", frame_id=fr.id, irrep="A1g", row=0, energy_at=0.5, energy_unc_at=0.01),
        Level(id="L2", frame_id=fr.id, irrep="A1g", row=0, energy_at=0.7),
    )
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    return Spectrum(
        ensembles=(Ensemble(id="E1", xi=3.444, xi_unc=0.05),),
        hadrons=(h,),
        channels=(ch,),
        frames=(fr,),
        levels=levels,
        covariance=cov,
    )


# ---------------------------------------------------------------------------
# exchange rules
# ---------------------------------------------------------------------------
def test_distinguishable_always_allowed():
    a = _hadron("A", twice_spin=1, parity=-1)
    b = _hadron("B", twice_spin=0, parity=-1)
    ch = Channel(label="AB", hadron_a=a, hadron_b=b)
    assert ch.exchange is ExchangeRule.DISTINGUISHABLE
    assert ch.intrinsic_parity == 1
    for ell in range(4):
        for twice_s in range(0, 7, 2):
            assert ch.exchange_allowed(ell, twice_s) is True


def test_identical_boson_exchange_allowed_twice_spin_0_and_2():
    # boson: sign = (-1)^(ell+S) must be +1  <=>  (ell + S) even
    # <=>  (2*ell + twice_s) % 4 == 0
    for twice_spin in (0, 2):
        h = _hadron("b", twice_spin=twice_spin)
        ch = Channel(label="bb", hadron_a=h, hadron_b=h, exchange=ExchangeRule.IDENTICAL)
        for ell in range(5):
            for twice_s in range(0, 2 * twice_spin + 1, 2):
                expected = (2 * ell + twice_s) % 4 == 0
                assert ch.exchange_allowed(ell, twice_s) is expected, (twice_spin, ell, twice_s)


def test_identical_fermion_exchange_allowed_twice_spin_1_and_3():
    # fermion: the intrinsic (-1)^(-twice_spin) combines with the required
    # sign=-1 so the criterion is the same integer condition as for bosons
    for twice_spin in (1, 3):
        h = _hadron("f", twice_spin=twice_spin)
        ch = Channel(label="ff", hadron_a=h, hadron_b=h, exchange=ExchangeRule.IDENTICAL)
        for ell in range(5):
            for twice_s in range(0, 2 * twice_spin + 1, 2):
                expected = (2 * ell + twice_s) % 4 == 0
                assert ch.exchange_allowed(ell, twice_s) is expected, (twice_spin, ell, twice_s)


def test_identical_exchange_matches_explicit_sign_formula():
    # cross-check the integer criterion against (-1)^ell * (-1)^(S - twice_spin)
    for twice_spin in (0, 1, 2, 3):
        h = _hadron("x", twice_spin=twice_spin)
        ch = Channel(label="xx", hadron_a=h, hadron_b=h, exchange=ExchangeRule.IDENTICAL)
        required = 1 if twice_spin % 2 == 0 else -1
        for ell in range(6):
            for twice_s in range(0, 2 * twice_spin + 1, 2):
                half_S = twice_s / 2
                sign = (-1) ** ell * (-1) ** (half_S - twice_spin)
                assert ch.exchange_allowed(ell, twice_s) is (sign == required), (
                    twice_spin,
                    ell,
                    twice_s,
                )


def test_identical_requires_equal_label_and_spin():
    a = _hadron("A", twice_spin=0)
    b = _hadron("B", twice_spin=0)
    with pytest.raises(ValueError):
        Channel(label="AB", hadron_a=a, hadron_b=b, exchange=ExchangeRule.IDENTICAL)
    c = _hadron("A", twice_spin=2)
    with pytest.raises(ValueError):
        Channel(label="AA", hadron_a=a, hadron_b=c, exchange=ExchangeRule.IDENTICAL)


def test_particle_antiparticle_has_no_exchange_constraint_but_keeps_intrinsic_parity():
    d = _hadron("D", twice_spin=0, parity=-1, mass=0.33)
    dbar = Hadron(label="Dbar", twice_spin=0, parity=-1, mass_at=0.33)
    ch = Channel(label="DDbar", hadron_a=d, hadron_b=dbar, exchange=ExchangeRule.PARTICLE_ANTIPARTICLE)
    assert ch.intrinsic_parity == 1
    for ell in range(4):
        for twice_s in range(0, 5, 2):
            assert ch.exchange_allowed(ell, twice_s) is True


def test_channel_accepts_string_exchange_and_rejects_unknown():
    h = _hadron()
    ch = Channel(label="hh", hadron_a=h, hadron_b=h, exchange="identical")
    assert ch.exchange is ExchangeRule.IDENTICAL
    with pytest.raises(ValueError):
        Channel(label="hh", hadron_a=h, hadron_b=h, exchange="nonsense")


# ---------------------------------------------------------------------------
# derived quantities
# ---------------------------------------------------------------------------
def test_allowed_twice_S_range():
    a = _hadron("a", twice_spin=1, parity=1)
    b = _hadron("b", twice_spin=2, parity=1)
    ch = Channel(label="ab", hadron_a=a, hadron_b=b)
    assert ch.allowed_twice_S == (1, 3)
    a0 = _hadron("c", twice_spin=0)
    b0 = _hadron("d", twice_spin=0)
    assert Channel(label="cd", hadron_a=a0, hadron_b=b0).allowed_twice_S == (0,)


# ---------------------------------------------------------------------------
# field validation
# ---------------------------------------------------------------------------
def test_hadron_validation():
    with pytest.raises(ValueError):
        Hadron(label="", twice_spin=0, parity=1, mass_at=0.3)
    with pytest.raises(ValueError):
        Hadron(label="x", twice_spin=0.5, parity=1, mass_at=0.3)
    with pytest.raises(ValueError):
        Hadron(label="x", twice_spin=-1, parity=1, mass_at=0.3)
    with pytest.raises(ValueError):
        Hadron(label="x", twice_spin=0, parity=0, mass_at=0.3)
    with pytest.raises(ValueError):
        Hadron(label="x", twice_spin=0, parity=1, mass_at=0.0)
    with pytest.raises(ValueError):
        Hadron(label="x", twice_spin=0, parity=1, mass_at=0.3, mass_unc_at=-0.1)


def test_ensemble_and_frame_validation():
    Ensemble(id="E", xi=1.0)
    with pytest.raises(ValueError):
        Ensemble(id="E", xi=0.0)
    with pytest.raises(ValueError):
        Ensemble(id="E", xi=1.0, xi_unc=-1.0)
    with pytest.raises(ValueError):
        Ensemble(id="E", xi=1.0, mass_covariance=((1.0, 0.2), (0.0, 1.0)))
    Ensemble(id="E", xi=1.0, mass_covariance=((1.0, 0.2), (0.2, 1.0)))
    Frame(id="F", spatial_sites=24, anisotropy=1.0, d=(0, 0, 0))
    with pytest.raises(ValueError):
        Frame(id="F", spatial_sites=0, anisotropy=1.0, d=(0, 0, 0))
    with pytest.raises(ValueError):
        Frame(id="F", spatial_sites=24, anisotropy=1.0, d=(0, 1))


# ---------------------------------------------------------------------------
# Spectrum invariants + lookups
# ---------------------------------------------------------------------------
def test_spectrum_lookups_and_index():
    sp = _spectrum()
    assert sp.hadron("X").label == "X"
    assert sp.channel("XX").label == "XX"
    assert sp.frame("L24_d000").spatial_sites == 24
    assert sp.ensemble("E1").xi == pytest.approx(3.444)
    assert sp.index("L2") == 1
    assert sp.n_levels == 2 == len(sp)
    with pytest.raises(KeyError):
        sp.index("missing")
    with pytest.raises(KeyError):
        sp.frame("missing")


def test_spectrum_rejects_missing_frame_and_duplicate_ids():
    base = _spectrum()
    bad_level = Level(id="L3", frame_id="nope", irrep="A1g", row=0, energy_at=0.9)
    cov = np.zeros((3, 3))
    np.fill_diagonal(cov, 0.1)
    with pytest.raises(ValueError):
        Spectrum(
            ensembles=base.ensembles,
            hadrons=base.hadrons,
            channels=base.channels,
            frames=base.frames,
            levels=base.levels + (bad_level,),
            covariance=cov,
        )
    with pytest.raises(ValueError):
        Spectrum(
            ensembles=base.ensembles,
            hadrons=base.hadrons,
            channels=base.channels,
            frames=base.frames,
            levels=(base.levels[0], base.levels[0]),
            covariance=np.eye(2),
        )


def test_spectrum_covariance_shape_symmetry_and_positivity():
    base = _spectrum()
    with pytest.raises(ValueError):
        Spectrum(
            ensembles=base.ensembles,
            hadrons=base.hadrons,
            channels=base.channels,
            frames=base.frames,
            levels=base.levels,
            covariance=np.eye(3),
        )
    asymmetric = np.array([[0.04, 0.05], [0.01, 0.09]])
    with pytest.raises(ValueError):
        Spectrum(
            ensembles=base.ensembles,
            hadrons=base.hadrons,
            channels=base.channels,
            frames=base.frames,
            levels=base.levels,
            covariance=asymmetric,
        )
    indefinite = np.array([[0.04, 0.0], [0.0, -0.01]])
    with pytest.raises(ValueError):
        Spectrum(
            ensembles=base.ensembles,
            hadrons=base.hadrons,
            channels=base.channels,
            frames=base.frames,
            levels=base.levels,
            covariance=indefinite,
        )
    # explicitly opting out of the PD check permits a singular covariance
    ok = Spectrum(
        ensembles=base.ensembles,
        hadrons=base.hadrons,
        channels=base.channels,
        frames=base.frames,
        levels=base.levels,
        covariance=np.zeros((2, 2)),
        require_positive_definite=False,
    )
    assert ok.covariance.shape == (2, 2)


def test_spectrum_is_frozen_and_requires_correct_covariance_dtype():
    import dataclasses

    sp = _spectrum()
    with pytest.raises(dataclasses.FrozenInstanceError):
        sp.units = "other"
    assert isinstance(sp.covariance, np.ndarray)
