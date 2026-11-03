from dataclasses import dataclass, replace

import numpy as np

from .diagonalization import diagonalize_hamiltonian
from .hamiltonians import build_spin_chain
from .observables import compute_local_magnetization, compute_total_magnetization
from .operators import build_spin_operators
from .parameters import SpinChainParameters


@dataclass
class SweepResults:
    """Numerical results from a one-parameter sweep.

    The first axis of each array corresponds to ``parameter_values``.
    """

    parameter_name: str
    parameter_values: np.ndarray
    eigenvalues: np.ndarray
    total_magnetizations: np.ndarray
    local_magnetizations: np.ndarray


def run_delta_sweep(
    params: SpinChainParameters,
    deltas: np.ndarray | list[float],
    eigvals: int = 0,
    sparse: bool = False,
) -> SweepResults:
    """Run a delta sweep and return structured observables.

    The function is intentionally a thin experiment-layer wrapper around the
    stable core routines: it builds a Hamiltonian for each delta, diagonalizes
    it, extracts the ground state, and reports scalar observables useful for
    plots or summary tables.
    """

    delta_values = np.asarray(deltas, dtype=float)
    operators = build_spin_operators(params.n_sites)
    eigenvalue_rows = []
    total_magnetizations = []
    local_magnetization_rows = []

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

        eigenvalue_rows.append(energies)
        total_magnetizations.append(total_magnetization)
        local_magnetization_rows.append(local_magnetizations[:, 0])

    return SweepResults(
        parameter_name="delta",
        parameter_values=delta_values,
        eigenvalues=np.asarray(eigenvalue_rows),
        total_magnetizations=np.asarray(total_magnetizations),
        local_magnetizations=np.asarray(local_magnetization_rows),
    )

def run_magnetic_field_sweep(
    params: SpinChainParameters,
    magnetic_fields: np.ndarray | list[float],
    eigvals: int = 0,
    sparse: bool = False,
) -> SweepResults:
    """Run a magnetic-field sweep and return structured observables.

    The function is intentionally a thin experiment-layer wrapper around the
    stable core routines: it builds a Hamiltonian for each delta, diagonalizes
    it, extracts the ground state, and reports scalar observables useful for
    plots or summary tables.
    """

    magnetic_field_values = np.asarray(magnetic_fields, dtype=float)
    operators = build_spin_operators(params.n_sites)
    eigenvalue_rows = []
    total_magnetizations = []
    local_magnetization_rows = []

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

        eigenvalue_rows.append(energies)
        total_magnetizations.append(total_magnetization)
        local_magnetization_rows.append(local_magnetizations[:, 0])

    return SweepResults(
        parameter_name="magnetic_field",
        parameter_values=magnetic_field_values,
        eigenvalues=np.asarray(eigenvalue_rows),
        total_magnetizations=np.asarray(total_magnetizations),
        local_magnetizations=np.asarray(local_magnetization_rows),
    )