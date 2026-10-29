import numpy as np

from spin_linear_chain.observables import compute_total_magnetization

from qutip import basis, tensor

def test_total_magnetization_updown():
    n_sites = 2

    basis_up = basis(2, 0)
    basis_down = basis(2, 1)

    state_upup = tensor([basis_up] * n_sites)
    state_downdown = tensor([basis_down] * n_sites)
    state_updown = tensor([basis_up, basis_down])

    magnetization = compute_total_magnetization([state_upup, state_downdown, state_updown], n_sites)

    assert np.isclose(magnetization[0], 1)
    assert np.isclose(magnetization[1], -1)
    assert np.isclose(magnetization[2], 0)