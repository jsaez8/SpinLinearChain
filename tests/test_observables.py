import numpy as np

from spin_linear_chain.observables import compute_total_magnetization

from qutip import basis, tensor

def test_total_magnetization_upup():
    n_sites = 2

    basis_up = basis(2, 0)
    state_upup = tensor([basis_up] * n_sites)

    magnetization = compute_total_magnetization([state_upup], n_sites)

    assert np.isclose(magnetization[0], n_sites / 2)

def test_total_magnetization_downdown():
    n_sites = 2

    basis_down = basis(2, 1)
    state_downdown = tensor([basis_down] * n_sites)

    magnetization = compute_total_magnetization([state_downdown], n_sites)

    assert np.isclose(magnetization[0], -n_sites / 2)

def test_total_magnetization_updown():
    n_sites = 2

    basis_up = basis(2, 0)
    basis_down = basis(2, 1)
    state_updown = tensor([basis_up, basis_down])

    magnetization = compute_total_magnetization([state_updown], n_sites)

    assert np.isclose(magnetization[0], 0)