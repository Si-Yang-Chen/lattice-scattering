"""Fix acceptance for the two defects identified in the defect review.

* **defect A** (harmonic degree cap): the cap was raised from 8 to 12 and must
  remain *correct*, not merely unguarded.  Acceptance uses two criteria that are
  independent of how the harmonic sums are computed: exact ``O_h`` covariance of
  the box matrix, and the analytic rest-frame selection rule.
* **defect C** (scan cost): ``harmonic_zeta_wide`` is memoised.  Acceptance
  checks the two properties that make memoisation safe and useful: identical
  arguments share one read-only result, and a fit-like access pattern (tiny
  parameter steps) actually hits the cache.

The scan-cost fix is explicitly **not** claimed to speed up a single scan: every
grid point has a distinct ``q2``, so a cold scan is unchanged.  That limitation
is asserted below rather than glossed over.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.slow
from scipy.linalg import block_diag

from lattice_scattering.symmetry.rotations import partial_wave_rotation
from lattice_scattering.symmetry.solid_harmonics import MAX_DEGREE
from lattice_scattering.finite_volume import zeta as zeta_module
from lattice_scattering.finite_volume.box import orbital_box
from lattice_scattering.finite_volume.zeta import MAX_ELL, harmonic_zeta_wide
from lattice_scattering.symmetry.groups import oh_group


# ---------------------------------------------------------------------------
# defect A: correctness of the raised harmonic cap
# ---------------------------------------------------------------------------

def test_the_caps_are_consistent_across_the_harmonic_kernel_and_the_zeta_kernel():
    """The Zeta cap may not exceed the harmonic kernel's own validated cap."""
    assert MAX_ELL == MAX_DEGREE


@pytest.mark.parametrize("ells", [(5,), (1, 5), (6,), (2, 6)])
def test_criterion_a_cubic_covariance_at_the_new_degrees(ells):
    """Exact group theory: ``B(R d) == D(R) B(d) D(R)^dagger`` for all of O_h.

    Independent of the harmonic-sum implementation, so this is the strongest
    available check that the raised cap did not silently break the algebra.
    """
    group = oh_group()
    worst = 0.0
    checked = 0
    for d in ((0, 0, 1), (0, 1, 1), (1, 1, 1)):
        base = orbital_box(0.4, ells, d=d, gamma=1.0, alpha=0.0)
        scale = max(1.0, float(np.max(np.abs(base))))
        for index, rotation in enumerate(group.elements):
            rotated = tuple(int(x) for x in rotation @ np.array(d))
            if sum(x * x for x in rotated) > 36:
                continue
            matrix = block_diag(
                *[
                    partial_wave_rotation(
                        np.asarray(group.lifts[index], dtype=float),
                        0,
                        (ell,),
                        inverted=bool(np.linalg.det(rotation) < 0),
                        intrinsic_parity=1,
                    )
                    for ell in ells
                ]
            )
            actual = orbital_box(0.4, ells, d=rotated, gamma=1.0, alpha=0.0)
            worst = max(worst, float(np.max(np.abs(actual - matrix @ base @ matrix.conj().T))) / scale)
            checked += 1
    assert checked >= 100
    assert worst <= 1e-12, f"cubic covariance violated by {worst:.3e} for ells={ells}"


@pytest.mark.parametrize("ell", [9, 10, 11, 12])
def test_criterion_b_rest_frame_selection_rule_at_the_new_degrees(ell):
    """Analytic rule: odd ``ell`` vanishes, even ``ell`` keeps only ``m % 4 == 0``."""
    values = harmonic_zeta_wide(-0.5, ell, d=(0, 0, 0), gamma=1.0, alpha=0.0).values
    scale = max(1.0, float(np.max(np.abs(values))))
    nonzero = sorted(
        int(m) for m in np.arange(-ell, ell + 1) if abs(values[m + ell]) > 1e-9 * scale
    )
    if ell % 2:
        assert nonzero == []
    else:
        assert nonzero == [m for m in range(-ell, ell + 1) if m % 4 == 0]


@pytest.mark.parametrize("ell", [6, 8, 10, 12])
def test_criterion_c_split_independence_at_the_new_degrees(ell):
    reference = harmonic_zeta_wide(-3.0, ell, d=(0, 0, 1), gamma=1.1, alpha=0.3, split=1.0).values
    scale = max(1.0, float(np.max(np.abs(reference))))
    for split in (0.5, 2.0):
        other = harmonic_zeta_wide(-3.0, ell, d=(0, 0, 1), gamma=1.1, alpha=0.3, split=split).values
        assert float(np.max(np.abs(other - reference))) / scale <= 1e-9


def test_degrees_beyond_the_cap_are_refused_by_both_layers():
    from lattice_scattering.symmetry.solid_harmonics import solid_harmonics

    with pytest.raises(ValueError, match=f"0..{MAX_DEGREE}"):
        solid_harmonics([[0.0, 0.0, 1.0]], MAX_DEGREE + 1)
    with pytest.raises(ValueError, match=f"0..{MAX_ELL}"):
        harmonic_zeta_wide(0.0, MAX_ELL + 1, d=(0, 0, 1), gamma=1.1, alpha=0.2)


# ---------------------------------------------------------------------------
# defect C: memoisation is safe and helps fits (but not a cold scan)
# ---------------------------------------------------------------------------

def test_identical_calls_share_one_read_only_result():
    """A shared result must not be mutable, or one caller corrupts every other."""
    zeta_module.zeta_cache_clear()
    first = harmonic_zeta_wide(-3.0, 0, d=(0, 0, 1), gamma=1.1, alpha=0.3)
    second = harmonic_zeta_wide(-3.0, 0, d=(0, 0, 1), gamma=1.1, alpha=0.3)
    assert first is second
    assert first.values.flags.writeable is False
    with pytest.raises(ValueError):
        first.values[0] = 1.0
    other = harmonic_zeta_wide(-3.0, 0, d=(0, 0, 1), gamma=1.2, alpha=0.3)
    assert other is not first


def test_diagnostics_arrays_are_write_protected_too():
    """The diagnostics carry arrays that callers could otherwise mutate in place."""
    zeta_module.zeta_cache_clear()
    result = harmonic_zeta_wide(-3.0, 0, d=(0, 0, 1), gamma=1.1, alpha=0.3)
    for name in ("direct", "image", "tolerance_per_m"):
        array = result.diagnostics[name]
        assert array.flags.writeable is False, name


def test_exact_duplicate_arguments_hit_the_memo():
    """The memo must actually bite for repeated identical calls."""
    zeta_module.zeta_cache_clear()
    before = zeta_module.zeta_cache_info()
    for _ in range(5):
        harmonic_zeta_wide(-3.0, 0, d=(0, 0, 1), gamma=1.1, alpha=0.3)
    after = zeta_module.zeta_cache_info()
    assert after.hits - before.hits >= 4
    assert after.misses - before.misses == 1


def test_a_cold_scan_is_not_faster_because_every_point_is_distinct():
    """Honest limitation: memoisation cannot help a *cold* single scan.

    Every scan point sits at a different ``q2``, so the hit rate is near zero.
    This is asserted so the fix is never described as a scan-speed fix.
    """
    zeta_module.zeta_cache_clear()
    n = 200
    for index in range(n):
        harmonic_zeta_wide(-3.0 + index * 1e-9, 0, d=(0, 0, 1), gamma=1.1, alpha=0.3)
    info = zeta_module.zeta_cache_info()
    assert info.misses == n
    assert info.hits <= 2, f"expected a cold scan to miss, got {info.hits} hits"


def test_a_fit_like_pattern_does_hit_the_memo():
    """Tiny parameter steps revisit the same lattice points, which is where it pays."""
    zeta_module.zeta_cache_clear()
    base = np.linspace(-3.0, -2.0, 300)
    for _ in range(3):
        for q2 in base:
            harmonic_zeta_wide(float(q2), 0, d=(0, 0, 1), gamma=1.1, alpha=0.3)
    info = zeta_module.zeta_cache_info()
    assert info.hits >= 2 * len(base), f"expected many hits on a repeated sweep, got {info.hits}"
