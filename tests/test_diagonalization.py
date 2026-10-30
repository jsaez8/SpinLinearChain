import numpy as np

from spin_linear_chain.diagonalization import diagonalize_hamiltonian
from spin_linear_chain.hamiltonians import build_spin_chain
from spin_linear_chain.operators import build_spin_operators
from spin_linear_chain.parameters import SpinChainParameters


def test_two_site_heisenberg_energies_are_sorted_and_correct():
    params = SpinChainParameters(
        n_sites=2,
        J=4.0,
        delta=0.0,
        magnetic_field=0.0,
        model="Heisenberg",
    )
    operators = build_spin_operators(params.n_sites)
    hamiltonian = build_spin_chain(params, operators)

    energies, _ = diagonalize_hamiltonian(hamiltonian)

    np.testing.assert_allclose(energies, [-3.0, 1.0, 1.0, 1.0], atol=1e-12)
    assert np.all(np.diff(energies) >= 0)


def test_eigenstates_satisfy_eigenvalue_equation():
    params = SpinChainParameters(
        n_sites=2,
        J=4.0,
        delta=0.0,
        magnetic_field=0.0,
        model="Heisenberg",
    )
    operators = build_spin_operators(params.n_sites)
    hamiltonian = build_spin_chain(params, operators)

    energies, states = diagonalize_hamiltonian(hamiltonian)

    for energy, state in zip(energies, states):
        residual = hamiltonian * state - energy * state
        assert residual.norm() < 1e-12
