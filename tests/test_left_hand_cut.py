"""Independent exchange partial-wave singularity and domain/route tests."""
from dataclasses import replace
import numpy as np
import pytest
from scipy.integrate import quad

from lattice_scattering.amplitudes import EqualMassExchangeDomain, LeftHandCutDomainError
from lattice_scattering.finite_volume import quantization_roots
from lattice_scattering.finite_volume import jls_matrix
from lattice_scattering.kinematics import LatticeFrame
from lattice_scattering.symmetry.groups import oh_group


DOMAIN = EqualMassExchangeDomain(mass=0.3, exchange_mass=0.1, exchange_allowed=True,
                                coupling=1., mechanism="equal_mass_t_exchange")


def test_branch_from_angular_exchange_denominator():
    report = DOMAIN.check((0.355, 0.365))
    # Integrate [mu^2 + 2 k^2 (1-z)]^-1 independently. The z=-1 endpoint
    # becomes singular at k^2=-mu^2/4; approach from its regular side.
    assert report.branch_point_s == pytest.approx(0.35)
    k2 = report.branch_point_k2
    assert 0.1**2 + 4*k2 == pytest.approx(0, abs=1e-17)
    integrals = [quad(lambda z: 1/(0.1**2+2*k2*f*(1-z)), -1, 1)[0] for f in (0.9, 0.99, 0.999)]
    assert integrals[0] < integrals[1] < integrals[2]
    assert report.qc_status == "not_excluded"
    assert report.ere_status == "inside_exchange_radius"


def test_qc_and_ere_are_separate_and_boundary_excluded():
    branch = DOMAIN.check((0.36, 0.36)).branch_point_s
    for low in (branch-0.001, branch):
        with pytest.raises(LeftHandCutDomainError):
            DOMAIN.check((low, 0.36)).require_qc()
    report = DOMAIN.check((0.36, 0.38))
    report.require_qc()
    with pytest.raises(LeftHandCutDomainError, match="outside_exchange_radius"):
        report.require_ere()
    path_report = DOMAIN.check((0.355, 0.365), path_s=[0.36, 0.36+0.1j])
    path_report.require_path()
    with pytest.raises(LeftHandCutDomainError, match="outside_exchange_radius"):
        path_report.require_ere()


@pytest.mark.parametrize("sheet", ["I", "II"])
def test_segment_crossing_not_just_vertices(sheet):
    r = DOMAIN.check((0.355, 0.365), path_s=[0.34+0.1j, 0.34-0.1j], sheet=sheet)
    with pytest.raises(LeftHandCutDomainError, match="crosses_declared_cut"):
        r.require_path()
    DOMAIN.check((0.355, 0.365), path_s=[0.36+0.1j, 0.36-0.1j], sheet=sheet).require_path()


def test_unknowns_do_not_pass():
    for kwargs in ({"mechanism": "unequal_mass_u_exchange"}, {"coupling": None},
                   {"exchange_allowed": None}, {"exchange_mass": None}, {"exchange_mass": 1.}):
        report = replace(DOMAIN, **kwargs).check((0.355, 0.365))
        assert report.qc_status == "unknown"
        with pytest.raises(LeftHandCutDomainError):
            report.require_qc()
    with pytest.raises(LeftHandCutDomainError):
        DOMAIN.check((0.355, 0.365), path_s=[0.36+0.1j], sheet="LHC-II").require_path()


def test_decoupled_or_forbidden_exchange():
    for kwargs in ({"coupling": 0.}, {"exchange_allowed": False}):
        r = replace(DOMAIN, **kwargs).check((0.3, 0.4), path_s=[0.2])
        assert r.branch_point_s is None
        r.require_qc().require_ere().require_path()


def test_unit_conversion_preserves_diagnostic():
    physical = replace(DOMAIN, mass=1., exchange_mass=0.2, units="GeV")
    converted = physical.to_temporal_lattice(0.15)
    r = physical.check((3.9, 4.1))
    c = converted.check(tuple(x*0.15**2 for x in (3.9, 4.1)))
    assert c.branch_point_s == pytest.approx(r.branch_point_s*0.15**2)
    assert c.maximum_ere_radius_ratio == pytest.approx(r.maximum_ere_radius_ratio)
    assert c.qc_status == r.qc_status


def test_scan_checks_whole_cm_window_before_matrix(monkeypatch):
    def forbidden(*a, **kw):
        pytest.fail("matrix called before domain rejection")
    monkeypatch.setattr(jls_matrix, "row_matrix", forbidden)
    with pytest.raises(LeftHandCutDomainError, match="excluded"):
        quantization_roots((0.59, 0.61), LatticeFrame(24, 1, (0,0,0)), ((0.3,0.3),),
                           ((0,0),), {0: lambda s: np.eye(1)}, group=oh_group(), irrep="A1g", lhc_domain=DOMAIN)


@pytest.mark.parametrize("kwargs", [{"mass": -1}, {"coupling": float("nan")}, {"units": "MeV"}, {"exchange_allowed": 1}])
def test_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        replace(DOMAIN, **kwargs)


def test_pole_rectangle_rejected_before_any_model_evaluation():
    from lattice_scattering.amplitudes import solve_poles
    class Forbidden:
        def inverse(self, s):
            pytest.fail("model evaluated before rectangle preflight")
    with pytest.raises(LeftHandCutDomainError, match="crosses_declared_cut"):
        solve_poles(Forbidden(), ((0.3,0.3),), "simple", ((0.34,-0.01),(0.37,0.01)), lhc_domain=DOMAIN)


@pytest.mark.parametrize("example", ["roots.json", "roots-amplitude.json"])
def test_cli_roots_reject_excluded_window(tmp_path, capsys, example):
    import json
    from pathlib import Path
    from dataclasses import asdict
    from lattice_scattering.__main__ import main
    data = json.loads((Path(__file__).parents[1]/"examples"/example).read_text())
    data.update(energy_window_at=[0.59,0.61], lhc_domain=asdict(DOMAIN), root_method="eigenvalues", subdivisions=0)
    target = tmp_path/"roots.json"
    target.write_text(json.dumps(data))
    assert main(["roots", str(target)]) == 2
    assert "standard QC excluded" in capsys.readouterr().err


def test_cli_fit_rejects_excluded_condition(tmp_path, capsys):
    import json
    from pathlib import Path
    from dataclasses import asdict
    from lattice_scattering.__main__ import main
    data = json.loads((Path(__file__).parents[1]/"examples/fit.json").read_text())
    data["conditions"][0].update(energy_window_at=[0.59,0.61], lhc_domain=asdict(DOMAIN))
    target = tmp_path/"fit.json"
    target.write_text(json.dumps(data))
    assert main(["fit", str(target)]) == 2
    assert "standard QC excluded" in capsys.readouterr().err
