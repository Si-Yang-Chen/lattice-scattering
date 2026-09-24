"""Numerical contracts for real-axis amplitude models."""

from __future__ import annotations

import numpy as np
import pytest

from lattice_scattering.amplitudes import PolePolynomialK


def _constant_background_model():
    return PolePolynomialK(
        mass=0.72,
        couplings=np.array([0.30, 0.25]),
        coefficients=(np.diag([-0.40, -0.55]),),
    )


@pytest.mark.parametrize("s", [0.50, 0.54, 0.52 + 0.03j])
def test_woodbury_inverse_matches_direct_inverse_away_from_the_bare_pole(s):
    """The stable form preserves the ordinary real and complex amplitude values."""
    model = _constant_background_model()
    direct = np.linalg.solve(model.k(s), np.eye(2))

    np.testing.assert_allclose(model.inverse(s), direct, rtol=2e-13, atol=2e-13)
    assert np.iscomplexobj(model.inverse(s)) is np.iscomplexobj(s)


def test_woodbury_inverse_remains_finite_at_the_representable_pole_neighbor():
    """A large rank-one K term must not hide its finite analytic inverse."""
    model = _constant_background_model()
    energy = float(np.nextafter(model.mass, np.inf))
    s = energy**2
    delta = model.mass**2 - s
    assert delta != 0.0

    # The explicitly formed K is so unbalanced that NumPy's default numerical
    # rank threshold classifies it as singular.  Its semantic inverse remains
    # finite and is given by the independent Sherman--Morrison expression.
    k_value = model.k(s)
    assert np.linalg.matrix_rank(k_value) < len(model.couplings)
    background = model.coefficients[0]
    background_inverse = np.linalg.inv(background)
    direction = background_inverse @ model.couplings
    denominator = delta + float(model.couplings @ direction)
    expected = background_inverse - np.outer(direction, direction) / denominator

    actual = model.inverse(s)
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)


def test_bare_pole_and_true_k_zero_remain_fail_closed():
    """The analytic inverse fixes conditioning, not genuine singular points."""
    model = _constant_background_model()
    with pytest.raises(ValueError, match="exact bare K pole"):
        model.k(model.mass**2)
    with pytest.raises(ValueError, match="exact bare K pole"):
        model.inverse(model.mass**2)

    k_zeros = [point for point in model.s_breakpoints() if point != model.mass**2]
    assert len(k_zeros) == 1
    with pytest.raises(ValueError, match="singular K has no inverse"):
        model.inverse(float(k_zeros[0]))
