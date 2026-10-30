import numpy as np
from qutip import basis, tensor

from spin_linear_chain.observables import (
    compute_chain_concurrence,
    compute_concurrence,
    compute_local_magnetization,
    compute_total_magnetization,
)


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

def test_local_magnetization_shape():
    n_sites = 3

    basis_up = basis(2, 0)
    basis_down = basis(2, 1)

    state_upupup = tensor([basis_up] * n_sites)
    state_downdowndown = tensor([basis_down] * n_sites)

    local_magnetization = compute_local_magnetization([state_upupup, state_downdowndown], n_sites)

    assert local_magnetization.shape == (n_sites, 2)
    assert local_magnetization[:, 0].shape[0] == n_sites

def test_concurrence():
    product_state = tensor(basis(2, 0), basis(2, 0))
    entangled_state = (tensor(basis(2, 0), basis(2, 1)) + tensor(basis(2, 1), basis(2, 0))).unit()
    partially_entangled_state = (0.2*tensor(basis(2, 0), basis(2, 1)) + tensor(basis(2, 1), basis(2, 0))).unit()
    

    concurrence_product = compute_concurrence(product_state, 0, 1)
    concurrence_entangled = compute_concurrence(entangled_state, 0, 1)
    concurrence_partially_entangled = compute_concurrence(partially_entangled_state, 0, 1)

    expected_partial = 2 * 0.2 / (1 + 0.2**2)
    
    assert np.isclose(concurrence_product, 0.0)
    assert np.isclose(concurrence_entangled, 1.0)
    assert np.isclose(concurrence_partially_entangled, expected_partial, atol=0.03)

def test_concurrence_chain():

    product_state = tensor(basis(2, 0), basis(2, 0))
    entangled_state = (tensor(basis(2, 0), basis(2, 1)) + tensor(basis(2, 1), basis(2, 0))).unit()
    partially_entangled_state = (0.2*tensor(basis(2, 0), basis(2, 1)) + tensor(basis(2, 1), basis(2, 0))).unit()

    ground_states = [product_state, entangled_state, partially_entangled_state]
    spin_a = 0
    spin_b = 1

    concurrences = compute_chain_concurrence(ground_states, spin_a, spin_b,)

    expected_partial = 2 * 0.2 / (1 + 0.2**2)

    np.testing.assert_allclose(
        concurrences,
        [0.0, 1.0, expected_partial],
        atol=1e-12,
    )