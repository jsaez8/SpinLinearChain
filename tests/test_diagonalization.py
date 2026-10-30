import numpy as np

from spin_linear_chain.diagonalization import (
    diagonalize_hamiltonian,
    sweep_delta_diagonalization,
)
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

def test_diagonalization_with_deltas():
    params = SpinChainParameters(
        n_sites=2,
        J=4.0,
        delta=0.0,
        magnetic_field=0.0,
        model="Heisenberg",
    )
    deltas = np.linspace(-1, 1, 5)
    eigvals = 4

    eigenvalues, eigenstates = sweep_delta_diagonalization(
        params=params,
        deltas=deltas,
        eigvals=eigvals,
        sparse=False,
    )

    assert eigenvalues.shape == (eigvals, len(deltas))
    assert eigenstates.shape == (eigvals, len(deltas))

def test_diagonalization_with_deltas_eigenvalue_equation():
    params = SpinChainParameters(
        n_sites=2,
        J=4.0,
        delta=0.0,
        magnetic_field=0.0,
        model="Heisenberg",
    )
    deltas = np.linspace(-1, 1, 5)
    eigvals = 4

    eigenvalues, eigenstates = sweep_delta_diagonalization(
        params=params,
        deltas=deltas,
        eigvals=eigvals,
        sparse=False,
    )

    for i, delta in enumerate(deltas):
        params.delta = delta
        operators = build_spin_operators(params.n_sites)
        hamiltonian = build_spin_chain(params, operators)

        for j in range(eigvals):
            energy = eigenvalues[j, i]
            state = eigenstates[j, i]
            residual = hamiltonian * state - energy * state
            assert residual.norm() < 1e-12


def test_delta_sweep_does_not_modify_input_parameters():
    initial_delta = 0.25
    params = SpinChainParameters(
        n_sites=2,
        J=4.0,
        delta=initial_delta,
        magnetic_field=0.0,
        model="Heisenberg",
    )

    sweep_delta_diagonalization(
        params=params,
        deltas=np.array([-0.5, 0.0, 0.5]),
        eigvals=2,
        sparse=False,
    )

    assert params.delta == initial_delta

def test_diagonalization_with_deltas_sparse():
    params = SpinChainParameters(
        n_sites=2,
        J=4.0,
        delta=0.0,
        magnetic_field=0.0,
        model="Heisenberg",
    )
    deltas = np.linspace(-1, 1, 5)
    eigvals = 4

    eigenvalues, eigenstates = sweep_delta_diagonalization(
        params=params,
        deltas=deltas,
        eigvals=eigvals,
        sparse=True,
    )

    assert eigenvalues.shape == (eigvals, len(deltas))
    assert eigenstates.shape == (eigvals, len(deltas))

def test_diagonalization_with_deltas_sparse_eigenvalue_equation():
    params = SpinChainParameters(
        n_sites=2,
        J=4.0,
        delta=0.0,
        magnetic_field=0.0,
        model="Heisenberg",
    )
    deltas = np.linspace(-1, 1, 5)
    eigvals = 4

    eigenvalues, eigenstates = sweep_delta_diagonalization(
        params=params,
        deltas=deltas,
        eigvals=eigvals,
        sparse=True,
    )

    for i, delta in enumerate(deltas):
        params.delta = delta
        operators = build_spin_operators(params.n_sites)
        hamiltonian = build_spin_chain(params, operators)

        for j in range(eigvals):
            energy = eigenvalues[j, i]
            state = eigenstates[j, i]
            residual = hamiltonian * state - energy * state
            assert residual.norm() < 1e-12