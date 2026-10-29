import numpy as np

from spin_linear_chain.observables import compute_total_magnetization, compute_local_magnetization

from qutip import basis, tensor

def test_total_magnetization():
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

def test_local_magnetization():
    n_sites = 2

    basis_up = basis(2, 0)
    basis_down = basis(2, 1)

    state_upup = tensor([basis_up] * n_sites)
    state_downdown = tensor([basis_down] * n_sites)
    state_updown = tensor([basis_up, basis_down])

    local_magnetization_upup = compute_local_magnetization([state_upup], n_sites)
    assert np.allclose(local_magnetization_upup[:, 0], [0.5, 0.5])

    local_magnetization_downdown = compute_local_magnetization([state_downdown], n_sites)
    assert np.allclose(local_magnetization_downdown[:, 0], [-0.5, -0.5])

    local_magnetization_updown = compute_local_magnetization([state_updown], n_sites)
    assert np.allclose(local_magnetization_updown[:, 0], [0.5, -0.5])
    