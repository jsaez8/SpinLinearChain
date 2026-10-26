import numpy as np
import pytest

from spin_linear_chain.couplings import alternating_couplings


def test_heisenberg_couplings_alternate_between_even_and_odd_bonds():
    exchange_x, exchange_y, exchange_z = alternating_couplings(
        J=10,
        delta=0.2,
        n_sites=4,
        model="Heisenberg",
    )

    expected = np.array([12, 8, 12])
    np.testing.assert_array_equal(exchange_x, expected)
    np.testing.assert_array_equal(exchange_y, expected)
    np.testing.assert_array_equal(exchange_z, expected)


def test_xy_model_has_no_z_exchange():
    exchange_x, exchange_y, exchange_z = alternating_couplings(
        J=10,
        delta=0.2,
        n_sites=4,
        model="XY",
    )

    np.testing.assert_array_equal(exchange_x, [12, 8, 12])
    np.testing.assert_array_equal(exchange_y, [12, 8, 12])
    np.testing.assert_array_equal(exchange_z, [0, 0, 0])


def test_disorder_scales_each_bond_coupling():
    disorder = np.array([1.0, 0.5, 1.0])

    exchange_x, exchange_y, exchange_z = alternating_couplings(
        J=10,
        delta=0.2,
        n_sites=4,
        disorder=disorder,
        model="Heisenberg",
    )

    expected = np.array([12, 4, 12])
    np.testing.assert_array_equal(exchange_x, expected)
    np.testing.assert_array_equal(exchange_y, expected)
    np.testing.assert_array_equal(exchange_z, expected)


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError):
        alternating_couplings(
            J=10,
            delta=0.2,
            n_sites=4,
            model="unknown",
        )