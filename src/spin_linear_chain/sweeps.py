from dataclasses import replace

import numpy as np

from .diagonalization import diagonalize_hamiltonian
from .hamiltonians import build_spin_chain
from .observables import compute_local_magnetization, compute_total_magnetization
from .operators import build_spin_operators
from .parameters import SpinChainParameters


def run_delta_sweep(
    params: SpinChainParameters,
    deltas: np.ndarray | list[float],
    eigvals: int = 1,
    sparse: bool = False,
) -> list[dict[str, float]]:
    """Run a sweep over a parameter and return structured observables.

    The function is intentionally a thin experiment-layer wrapper around the
    stable core routines: it builds a Hamiltonian for each delta, diagonalizes
    it, extracts the ground state, and reports scalar observables useful for
    plots or summary tables.
    """

    delta_values = np.asarray(deltas, dtype=float)
    operators = build_spin_operators(params.n_sites)
    results: list[dict[str, float]] = []

    for delta in delta_values:
        sweep_params = replace(params, delta=float(delta))
        hamiltonian = build_spin_chain(sweep_params, operators)

        energies, states = diagonalize_hamiltonian(
            hamiltonian,
            eigvals=eigvals,
            sparse=sparse,
        )

        ground_state = states[0]
        total_magnetization = compute_total_magnetization(
            [ground_state],
            sweep_params.n_sites,
        )[0]

        local_magnetizations = compute_local_magnetization(
            [ground_state],
            sweep_params.n_sites,
        )

        results.append(
            {
                "delta": float(delta),
                "ground_energy": float(energies[0]),
                "total_magnetization": float(total_magnetization),
                "local_magnetizations": local_magnetizations,
            }
        )

    return results

def run_magnetic_field_sweep(
    params: SpinChainParameters,
    magnetic_fields: np.ndarray | list[float],
    eigvals: int = 1,
    sparse: bool = False,
) -> list[dict[str, float]]:
    """Run a sweep over a parameter and return structured observables.

    The function is intentionally a thin experiment-layer wrapper around the
    stable core routines: it builds a Hamiltonian for each delta, diagonalizes
    it, extracts the ground state, and reports scalar observables useful for
    plots or summary tables.
    """

    magnetic_field_values = np.asarray(magnetic_fields, dtype=float)
    operators = build_spin_operators(params.n_sites)
    results: list[dict[str, float]] = []

    for magnetic_field in magnetic_field_values:
        sweep_params = replace(params, magnetic_field=float(magnetic_field))
        hamiltonian = build_spin_chain(sweep_params, operators)

        energies, states = diagonalize_hamiltonian(
            hamiltonian,
            eigvals=eigvals,
            sparse=sparse,
        )

        ground_state = states[0]
        total_magnetization = compute_total_magnetization(
            [ground_state],
            sweep_params.n_sites,
        )[0]

        local_magnetizations = compute_local_magnetization(
            [ground_state],
            sweep_params.n_sites,
        )

        results.append(
            {
                "magnetic_field": float(magnetic_field),
                "ground_energy": float(energies[0]),
                "total_magnetization": float(total_magnetization),
                "local_magnetizations": local_magnetizations,
            }
        )

    return results