import numpy as np

from .couplings import alternating_couplings

import scipy.constants
from scipy.constants import physical_constants, eV


MU_B = physical_constants["Bohr magneton"][0] / eV
PLANCK = scipy.constants.h / eV


def build_spin_chain(
    params,
    operators,
):
    """
    Construct the spin-chain Hamiltonian.
    """

    exchange_x, exchange_y, exchange_z = alternating_couplings(
        params.J,
        params.delta,
        params.n_sites,
        params.exchange_disorder,
        params.model,
    )

    # Zeeman energies
    zeeman_energy = (
        0.5
        * MU_B
        / PLANCK
        * np.asarray(params.g_factors)
        * params.magnetic_field
        * 1e-6
    )

    if np.ndim(zeeman_energy) == 0:
        zeeman_energy = np.full(
            params.n_sites,
            zeeman_energy,
        )

    hamiltonian = 0

    # Zeeman term
    for site in range(params.n_sites):

        hamiltonian += (
            -zeeman_energy[site]
            * operators.sz[site]
        )

    # Exchange interaction
    for bond in range(params.n_sites - 1):

        hamiltonian += (
            0.25
            * exchange_x[bond]
            * operators.sx[bond]
            * operators.sx[bond + 1]
        )

        hamiltonian += (
            0.25
            * exchange_y[bond]
            * operators.sy[bond]
            * operators.sy[bond + 1]
        )

        hamiltonian += (
            0.25
            * exchange_z[bond]
            * operators.sz[bond]
            * operators.sz[bond + 1]
        )

    return hamiltonian