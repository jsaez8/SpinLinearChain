import numpy as np
from qutip import qeye, sigmax, sigmay, sigmaz, tensor

from spin_linear_chain.operators import build_spin_operators

def test_number_operators():

    n_sites = 2
    spin_ops = build_spin_operators(n_sites)

    assert len(spin_ops.sx) == n_sites
    assert len(spin_ops.sy) == n_sites
    assert len(spin_ops.sz) == n_sites

def test_operator_shapes():

    n_sites = 2
    spin_ops = build_spin_operators(n_sites)

    for sx, sy, sz in zip(spin_ops.sx, spin_ops.sy, spin_ops.sz):
        assert sx.shape == (2**n_sites, 2**n_sites)
        assert sy.shape == (2**n_sites, 2**n_sites)
        assert sz.shape == (2**n_sites, 2**n_sites)


def test_sx_operator_is_applied_to_the_correct_site():
    operators = build_spin_operators(n_sites=2)

    expected_site_0 = tensor(sigmax(), qeye(2))
    expected_site_1 = tensor(qeye(2), sigmax())

    ### X operator
    np.testing.assert_allclose(
        operators.sx[0].full(),
        expected_site_0.full(),
    )
    np.testing.assert_allclose(
        operators.sx[1].full(),
        expected_site_1.full(),
    )

    ### Y operator
    expected_site_0 = tensor(sigmay(), qeye(2))
    expected_site_1 = tensor(qeye(2), sigmay())

    np.testing.assert_allclose(
        operators.sy[0].full(),
        expected_site_0.full(),
    )
    np.testing.assert_allclose(
        operators.sy[1].full(),
        expected_site_1.full(),
    )

    ### Z operator
    expected_site_0 = tensor(sigmaz(), qeye(2))
    expected_site_1 = tensor(qeye(2), sigmaz())

    np.testing.assert_allclose(
        operators.sz[0].full(),
        expected_site_0.full(),
    )
    np.testing.assert_allclose(
        operators.sz[1].full(),
        expected_site_1.full(),
    )