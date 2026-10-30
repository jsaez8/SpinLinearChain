import numpy as np
from tqdm import tqdm

from .hamiltonians import build_spin_chain
from .operators import build_spin_operators
from .parameters import SpinChainParameters


def diagonalize_hamiltonian(
    hamiltonian,
    eigvals=0,
    sparse=False,
):
    """
    Diagonalize a single Hamiltonian.

    Parameters
    ----------
    hamiltonian : qutip.Qobj
        Hamiltonian to diagonalize. Internally stored as a sparse
        (CSR) matrix by qutip.

    eigvals : int, optional
        Number of low-lying eigenvalues/eigenstates to compute.
        0 (default) computes the full spectrum. Only used when
        `sparse=True` — requesting a small `eigvals` there is what
        makes large chains (N >~ 10-12) tractable, since it runs a
        sparse Lanczos solver (scipy ARPACK) instead of a dense
        diagonalization of the full 2**N x 2**N matrix.

    sparse : bool, optional
        If True, use qutip's sparse eigensolver (recommended for
        large `n_sites` combined with a small `eigvals`, e.g. when
        you only need the ground state and a few excited states).
        If False (default), use the dense solver, which computes the
        full spectrum regardless of `eigvals` and is only practical
        for small chains (roughly N <~ 12-14).

    Returns
    -------
    energies : ndarray
        Eigenvalues, sorted ascending.

    states : list
        Eigenstates, same order as `energies`.
    """

    energies, states = hamiltonian.eigenstates(
        sparse=sparse,
        eigvals=eigvals,
        sort="low",
    )

    # Hamiltonian is Hermitian: eigenvalues are real. The sparse solver
    # returns them as complex128 with a numerically negligible imaginary
    # part, so drop it explicitly rather than let numpy warn/truncate.
    energies = np.real(np.asarray(energies))

    return energies, states



def sweep_delta_diagonalization(
    params: SpinChainParameters,
    deltas: np.ndarray,
    eigvals=0,
    sparse=False,
):
    """
    Diagonalize the spin chain for a sweep of delta values.

    Parameters
    ----------
    params : SpinChainParameters
        Spin-chain parameters.

    deltas : ndarray
        Values of the dimerization parameter.

    eigvals : int, optional
        Number of low-lying eigenvalues/eigenstates to compute per
        delta. 0 (default) computes the full spectrum. See
        diagonalize_hamiltonian for when to combine this with
        `sparse=True`.

    sparse : bool, optional
        Use the sparse Lanczos eigensolver instead of dense
        diagonalization. Strongly recommended together with a small
        `eigvals` for large chains, where a full dense diagonalization
        at every delta is prohibitively slow.

    Returns
    -------
    eigenvalues : ndarray
        Shape (eigvals or 2**N, len(deltas)).

    eigenstates : ndarray
        Shape (eigvals or 2**N, len(deltas)).
    """

    # Build operators only once
    operators = build_spin_operators(
        params.n_sites
    )

    dimension = 2 ** params.n_sites

    n_levels = eigvals if eigvals > 0 else dimension

    eigenvalues = np.zeros(
        (n_levels, len(deltas))
    )

    eigenstates = np.empty(
        (n_levels, len(deltas)),
        dtype=object,
    )


    for idx, delta in enumerate(
        tqdm(deltas, desc="Diagonalizing")
    ):

        # Update parameter
        params.delta = delta

        # Build Hamiltonian
        hamiltonian = build_spin_chain(
            params,
            operators,
        )

        # Diagonalize
        energies, states = diagonalize_hamiltonian(
            hamiltonian,
            eigvals=eigvals,
            sparse=sparse,
        )

        eigenvalues[:, idx] = energies
        eigenstates[:, idx] = states


    return eigenvalues, eigenstates