"""Unit tests for the v2 JLS sector layout (``v2/core/jls.py``)."""
from __future__ import annotations

import pytest

from lattice_scattering.core.jls import (
    Sector,
    enumerate_sectors,
    jls_blocks,
    jls_shape,
)
from lattice_scattering.core.model import Channel, ExchangeRule, Hadron


def _h(label, twice_spin, parity=-1, mass=0.3):
    return Hadron(label=label, twice_spin=twice_spin, parity=parity, mass_at=mass)


# ---------------------------------------------------------------------------
# ordering / channel-major enumeration
# ---------------------------------------------------------------------------
def test_enumerate_sectors_is_channel_major_then_ell_then_twice_s():
    aa = Channel(
        label="aa",
        hadron_a=_h("a", 2, parity=1),
        hadron_b=_h("a", 2, parity=1),
        exchange=ExchangeRule.IDENTICAL,
    )
    bb = Channel(
        label="bb",
        hadron_a=_h("b", 1, parity=-1),
        hadron_b=_h("b", 1, parity=-1),
        exchange=ExchangeRule.IDENTICAL,
    )
    sectors = enumerate_sectors((aa, bb), ell_max=1, exchange=False)
    # aa: 2S in {0,2,4}; bb: 2S in {0,2}; 2 ells each
    # channel-major: every sector of channel 0 precedes channel 1
    assert [s.channel_index for s in sectors] == [0] * 6 + [1] * 4
    # within a channel, ell ascending then twice_S ascending
    assert [s.ell for s in sectors] == [0, 0, 0, 1, 1, 1, 0, 0, 1, 1]
    assert [s.twice_S for s in sectors] == [0, 2, 4, 0, 2, 4, 0, 2, 0, 2]


def test_enumerate_sectors_uses_allowed_twice_S_per_channel():
    ch = Channel(label="ab", hadron_a=_h("a", 1, 1), hadron_b=_h("b", 2, 1))
    sectors = enumerate_sectors((ch,), ell_max=0)
    assert {(s.ell, s.twice_S) for s in sectors} == {(0, 1), (0, 3)}


# ---------------------------------------------------------------------------
# exchange filter
# ---------------------------------------------------------------------------
def test_enumerate_sectors_exchange_filter_matches_channel_rule():
    boson = Channel(
        label="bb",
        hadron_a=_h("b", 0, 1),
        hadron_b=_h("b", 0, 1),
        exchange=ExchangeRule.IDENTICAL,
    )
    with_exchange = enumerate_sectors((boson,), ell_max=4, exchange=True)
    assert [(s.ell, s.twice_S) for s in with_exchange] == [(0, 0), (2, 0), (4, 0)]
    without = enumerate_sectors((boson,), ell_max=4, exchange=False)
    assert [(s.ell, s.twice_S) for s in without] == [(ell, 0) for ell in range(5)]


def test_enumerate_sectors_fermion_exchange_filter():
    fermion = Channel(
        label="ff",
        hadron_a=_h("f", 1, -1),
        hadron_b=_h("f", 1, -1),
        exchange=ExchangeRule.IDENTICAL,
    )
    sectors = enumerate_sectors((fermion,), ell_max=3)
    assert [(s.ell, s.twice_S) for s in sectors] == [(0, 0), (1, 2), (2, 0), (3, 2)]


# ---------------------------------------------------------------------------
# parity filter
# ---------------------------------------------------------------------------
def test_enumerate_sectors_parity_target_filter():
    ch = Channel(label="ab", hadron_a=_h("a", 0, -1), hadron_b=_h("b", 0, -1))
    assert ch.intrinsic_parity == 1
    even = enumerate_sectors((ch,), ell_max=4, parity_target=+1, exchange=False)
    assert [(s.ell, s.twice_S) for s in even] == [(0, 0), (2, 0), (4, 0)]
    odd = enumerate_sectors((ch,), ell_max=3, parity_target=-1, exchange=False)
    assert [(s.ell, s.twice_S) for s in odd] == [(1, 0), (3, 0)]


def test_enumerate_sectors_rejects_bad_arguments():
    ch = Channel(label="ab", hadron_a=_h("a", 0), hadron_b=_h("b", 0))
    with pytest.raises(ValueError):
        enumerate_sectors((ch,), ell_max=-1)
    with pytest.raises(ValueError):
        enumerate_sectors((ch,), ell_max=1, ell_min=2)
    with pytest.raises(ValueError):
        enumerate_sectors((ch,), ell_max=1, parity_target=0)


# ---------------------------------------------------------------------------
# triangular criterion
# ---------------------------------------------------------------------------
def test_jls_triangular_criterion():
    sectors = (Sector(channel_index=0, ell=1, twice_S=2),)
    blocks = jls_blocks(sectors)
    # |2ell - 2S| <= 2J <= 2ell + 2S  with 2J same parity as 2ell + 2S
    # here: 0 <= 2J <= 4, 2J even -> {0, 2, 4}
    active = sorted(j for j, members in blocks.items() if members)
    assert active == [0, 2, 4]
    for j in active:
        assert blocks[j] == sectors
    assert blocks[1] == ()
    assert blocks[3] == ()


def test_jls_blocks_covers_full_range_and_preserves_order():
    s0 = Sector(0, 0, 0)   # only 2J=0
    s1 = Sector(0, 2, 2)   # 2J in {2, 4, 6}
    s2 = Sector(1, 1, 2)   # 2J in {0, 2, 4}
    blocks = jls_blocks((s0, s1, s2))
    # twice_J = 0 .. 2*ell_max + max(twice_S) = 2*2 + 2 = 6
    assert sorted(blocks) == list(range(7))
    assert blocks[0] == (s0, s2)
    assert blocks[1] == ()
    assert blocks[2] == (s1, s2)
    assert blocks[3] == ()
    assert blocks[4] == (s1, s2)
    assert blocks[5] == ()
    assert blocks[6] == (s1,)


def test_jls_blocks_empty_input():
    assert jls_blocks(()) == {}
    assert jls_shape((), 0) == 0


def test_jls_shape_matches_block_lengths():
    sectors = (
        Sector(0, 0, 0),
        Sector(1, 0, 2),
        Sector(0, 2, 0),
    )
    for twice_j in range(0, 7):
        assert jls_shape(sectors, twice_j) == len(jls_blocks(sectors)[twice_j])
    with pytest.raises(ValueError):
        jls_shape(sectors, -1)


def test_jls_to_j2_completeness():
    # For each sector, the number of allowed 2J values equals 2*min(ell,S_iso)+1
    # counted with the correct parity; here we just assert the sum over blocks
    # equals the number of (sector, J) incidences.
    sectors = tuple(Sector(channel_index=c, ell=ell, twice_S=2) for c in range(2) for ell in (1, 3))
    incidences = sum(len(members) for members in jls_blocks(sectors).values())
    expected = sum(
        len([j for j in range(0, 2 * s.ell + s.twice_S + 1) if (j - (2 * s.ell + s.twice_S)) % 2 == 0 and abs(2 * s.ell - s.twice_S) <= j])
        for s in sectors
    )
    assert incidences == expected


def test_sector_validation():
    with pytest.raises(ValueError):
        Sector(channel_index=-1, ell=0, twice_S=0)
    with pytest.raises(ValueError):
        Sector(channel_index=0, ell=-1, twice_S=0)
    with pytest.raises(ValueError):
        Sector(channel_index=0, ell=0, twice_S=-2)
