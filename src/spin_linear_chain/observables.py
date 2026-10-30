"""
Calculation of physical observables for spin chains.
"""

import numpy as np
from qutip import sigmax, sigmay, sigmaz, tensor, qeye, expect
import qutip

# ============================================================
# Magnetization
# ============================================================

def compute_total_magnetization(
    ground_states,
    n_sites,
):
    """
    Compute total magnetization along z for a sweep of states.

    Parameters
    ----------
    ground_states : array of qutip.Qobj
        Ground state for each value of the swept parameter.

    n_sites : int
        Number of spins in the chain.

    Returns
    -------
    magnetization : np.ndarray
        Total magnetization <sum_i Sz_i> for each parameter value.
    """

    # --------------------------------------------------------
    # Build total Sz operator
    # --------------------------------------------------------

    total_sz = 0

    for i in range(n_sites):

        op_list = [
            qeye(2)
        ] * n_sites

        op_list[i] = 0.5 * sigmaz()

        total_sz += tensor(op_list)


    # --------------------------------------------------------
    # Compute expectation values
    # --------------------------------------------------------

    magnetization = np.zeros(
        len(ground_states)
    )

    for idx, state in enumerate(ground_states):

        magnetization[idx] = np.real(
            expect(
                total_sz,
                state,
            )
        )

    return magnetization

def compute_local_magnetization(
    ground_states,
    n_sites,
):
    """
    Compute local magnetization <Sz_i> for every site
    and every value of the swept parameter.

    Parameters
    ----------
    ground_states : array of qutip.Qobj
        Ground state for each value of the swept parameter.

        Shape:
            (n_points,)

    n_sites : int
        Number of spins in the chain.

    Returns
    -------
    local_magnetization : ndarray

        Shape:
            (n_sites, n_points)

        local_magnetization[i, j]
        corresponds to <Sz_i> at parameter point j.
    """

    n_points = len(ground_states)

    local_magnetization = np.zeros(
        (n_sites, n_points)
    )


    # --------------------------------------------------------
    # Calculate <Sz_i> for every site
    # --------------------------------------------------------

    for site in range(n_sites):

        # Build local Sz operator
        op_list = [
            qeye(2)
        ] * n_sites

        op_list[site] = 0.5 * sigmaz()

        sz_operator = tensor(
            op_list
        )


        # Expectation value along sweep
        for idx, state in enumerate(ground_states):

            local_magnetization[site, idx] = np.real(
                expect(
                    sz_operator,
                    state,
                )
            )


    return local_magnetization


# ============================================================
# Concurrence
# ============================================================

def compute_concurrence(
    state,
    spin_a,
    spin_b,
):
    """
    Compute concurrence between two spins.
    """

    rho = state.proj()

    rho_ab = rho.ptrace(
        [spin_a, spin_b]
    )

    sy = sigmay()

    Y = tensor(
        sy,
        sy,
    )

    R = (
        rho_ab
        * Y
        * rho_ab.conj()
        * Y
    )

    eigenvalues = np.real(
        R.eigenenergies()
    )

    eigenvalues = np.maximum(
        eigenvalues,
        0,
    )

    lambdas = np.sqrt(
        eigenvalues
    )

    lambdas = np.sort(
        lambdas
    )[::-1]

    concurrence = max(
        0,
        lambdas[0]
        - lambdas[1]
        - lambdas[2]
        - lambdas[3],
    )

    return float(concurrence)


def compute_chain_concurrence(
    ground_states,
    spin_a,
    spin_b,
):
    """
    Compute concurrence for all parameter values.
    """

    concurrence = np.zeros(
        len(ground_states)
    )

    for i, state in enumerate(
        ground_states
    ):

        concurrence[i] = compute_concurrence(
            state,
            spin_a,
            spin_b,
        )

    return concurrence


# # ============================================================
# # Spin-spin correlation
# # ============================================================

# def compute_spin_correlation(
#     state,
#     spin_a,
#     spin_b,
# ):
#     """
#     Compute <S_a . S_b> between two spins in a many-body state.
#     """

#     rho = state.proj()

#     rho_ab = rho.ptrace(
#         [spin_a, spin_b]
#     )

#     correlation_operator = 0.25 * (
#         tensor(sigmax(), sigmax())
#         +
#         tensor(sigmay(), sigmay())
#         +
#         tensor(sigmaz(), sigmaz())
#     )

#     return float(
#         np.real(
#             (rho_ab * correlation_operator).tr()
#         )
#     )


# def compute_chain_spin_correlation(
#     ground_states,
#     spin_a,
#     spin_b,
# ):
#     """
#     Compute <S_a . S_b> for all values of the swept parameter.
#     """

#     correlation = np.zeros(
#         len(ground_states)
#     )

#     for i, state in enumerate(
#         ground_states
#     ):

#         correlation[i] = compute_spin_correlation(
#             state,
#             spin_a,
#             spin_b,
#         )

#     return correlation

# # ============================================================
# # Basis-state probabilities
# # ============================================================

# def compute_state_amplitudes(
#     state,
# ):
#     """
#     Extract computational-basis amplitudes <n|state> of a many-body ket.

#     Parameters
#     ----------
#     state : qutip.Qobj
#         Many-body ket.

#     Returns
#     -------
#     amplitudes : ndarray of complex
#         Amplitude of each computational basis state, ordered to
#         match qutip's tensor-product basis index.
#     """

#     return state.full().flatten()


# def compute_basis_probabilities(
#     state,
# ):
#     """
#     Compute computational-basis probabilities |<n|state>|^2.

#     Parameters
#     ----------
#     state : qutip.Qobj
#         Many-body ket.

#     Returns
#     -------
#     probabilities : ndarray
#         Probability of each computational basis state, ordered
#         to match qutip's tensor-product basis index.
#     """

#     return np.abs(compute_state_amplitudes(state)) ** 2


# def basis_state_labels(
#     n_sites,
# ):
#     """
#     LaTeX ket labels for the computational basis, e.g. "|\\uparrow\\downarrow\\rangle".

#     Labels are ordered to match qutip's tensor-product basis index,
#     where site 0 is the most significant bit ('0' -> up, '1' -> down).
#     They are raw LaTeX (no surrounding '$') so they can be used either
#     inside an existing math environment (e.g. reporting.format_ket_expansion)
#     or wrapped in '$...$' for matplotlib mathtext tick labels.

#     Parameters
#     ----------
#     n_sites : int
#         Number of spins in the chain.

#     Returns
#     -------
#     labels : list of str
#         One LaTeX ket label per basis state.
#     """

#     labels = []

#     for index in range(2 ** n_sites):

#         bits = format(index, f"0{n_sites}b")

#         spins = "".join(
#             r"\uparrow" if bit == "0" else r"\downarrow"
#             for bit in bits
#         )

#         labels.append(
#             rf"|{spins}\rangle"
#         )

#     return labels


# # ============================================================
# # Triplet probabilities
# # ============================================================

# def _bond_site_labels(
#     n_sites,
#     add_ancillas,
# ):
#     """
#     Site labels for bond naming: real sites are numbered 1..n_sites,
#     ancillas (when present) are named A1/A2 at the two chain ends.
#     """

#     labels = [
#         str(site + 1)
#         for site in range(n_sites)
#     ]

#     if add_ancillas:
#         labels = ["A1"] + labels + ["A2"]

#     return labels


# def compute_triplet_weights(
#     state,
#     add_ancillas=False,
# ):
#     """
#     Compute triplet probability for every neighbouring pair.

#     Parameters
#     ----------
#     state : qutip.Qobj
#         Many-body state.

#     add_ancillas : bool
#         If True, adds |up> ancillas at both ends.

#     Returns
#     -------
#     triplet_weights : ndarray
#         Triplet probability for every bond.

#     bond_labels : list
#         Labels of the measured pairs.
#     """

#     N = int(np.log2(state.shape[0]))

#     site_labels = _bond_site_labels(
#         N,
#         add_ancillas,
#     )


#     up = qutip.basis(
#         2,
#         0,
#     )

#     down = qutip.basis(
#         2,
#         1,
#     )


#     # Add ancillas
#     if add_ancillas:

#         state = qutip.tensor(
#             up,
#             state,
#             up,
#         )

#         N += 2


#     # Define singlet/triplet states

#     triplet_p = qutip.tensor(
#         up,
#         up,
#     )

#     triplet_0 = (
#         qutip.tensor(up, down)
#         +
#         qutip.tensor(down, up)
#     ).unit()

#     triplet_m = qutip.tensor(
#         down,
#         down,
#     )


#     P_triplet = (
#         triplet_p * triplet_p.dag()
#         +
#         triplet_0 * triplet_0.dag()
#         +
#         triplet_m * triplet_m.dag()
#     )


#     rho = qutip.ket2dm(
#         state
#     )


#     triplet_weights = []

#     bond_labels = []


#     for i in range(N - 1):

#         rho_pair = rho.ptrace(
#             [
#                 i,
#                 i + 1
#             ]
#         )

#         weight = (
#             rho_pair
#             *
#             P_triplet
#         ).tr().real


#         triplet_weights.append(
#             weight
#         )

#         bond_labels.append(
#             f"{site_labels[i]}-{site_labels[i+1]}"
#         )


#     return (
#         np.array(triplet_weights),
#         bond_labels,
#     )



# def compute_chain_triplet_weights(
#     ground_states,
#     add_ancillas=False,
# ):
#     """
#     Compute triplet weights for all values of the swept parameter.

#     Parameters
#     ----------
#     ground_states : list
#         Ground state for every delta.

#     Returns
#     -------
#     weights : ndarray
#         Shape:
#         (n_delta, n_bonds)

#     labels : list
#         Bond labels.
#     """

#     weights = []

#     labels = None


#     for state in ground_states:

#         w, labels = compute_triplet_weights(
#             state,
#             add_ancillas=add_ancillas,
#         )

#         weights.append(
#             w
#         )


#     return (
#         np.array(weights),
#         labels,
#     )


# # ============================================================
# # Singlet probabilities
# # ============================================================

# def compute_singlet_weights(
#     state,
#     add_ancillas=False,
# ):
#     """
#     Compute singlet probability for every neighbouring pair.

#     Parameters
#     ----------
#     state : qutip.Qobj
#         Many-body state.

#     add_ancillas : bool
#         If True, adds |up> ancillas at both ends.

#     Returns
#     -------
#     singlet_weights : ndarray
#         Singlet probability for every bond.

#     bond_labels : list
#         Labels of the measured pairs.
#     """

#     N = int(np.log2(state.shape[0]))

#     site_labels = _bond_site_labels(
#         N,
#         add_ancillas,
#     )


#     up = qutip.basis(
#         2,
#         0,
#     )

#     down = qutip.basis(
#         2,
#         1,
#     )


#     # Add ancillas
#     if add_ancillas:

#         state = qutip.tensor(
#             up,
#             state,
#             up,
#         )

#         N += 2


#     # Define singlet state

#     singlet = (
#         qutip.tensor(up, down)
#         -
#         qutip.tensor(down, up)
#     ).unit()


#     P_singlet = (
#         singlet * singlet.dag()
#     )


#     rho = qutip.ket2dm(
#         state
#     )


#     singlet_weights = []

#     bond_labels = []


#     for i in range(N - 1):

#         rho_pair = rho.ptrace(
#             [
#                 i,
#                 i + 1
#             ]
#         )

#         weight = (
#             rho_pair
#             *
#             P_singlet
#         ).tr().real


#         singlet_weights.append(
#             weight
#         )

#         bond_labels.append(
#             f"{site_labels[i]}-{site_labels[i+1]}"
#         )


#     return (
#         np.array(singlet_weights),
#         bond_labels,
#     )



# def compute_chain_singlet_weights(
#     ground_states,
#     add_ancillas=False,
# ):
#     """
#     Compute singlet weights for all values of the swept parameter.

#     Parameters
#     ----------
#     ground_states : list
#         Ground state for every delta.

#     Returns
#     -------
#     weights : ndarray
#         Shape:
#         (n_delta, n_bonds)

#     labels : list
#         Bond labels.
#     """

#     weights = []

#     labels = None


#     for state in ground_states:

#         w, labels = compute_singlet_weights(
#             state,
#             add_ancillas=add_ancillas,
#         )

#         weights.append(
#             w
#         )


#     return (
#         np.array(weights),
#         labels,
#     )


# # ============================================================
# # Bond order parameter
# # ============================================================

# def compute_bond_order_parameter(
#     singlet_weights,
#     bond_labels,
# ):
#     """
#     Compute the local dimerization (bond order parameter) from
#     singlet weights.

#     Defined as the difference between consecutive nearest-neighbour
#     correlators:

#         D_i = <S_i . S_{i+1}> - <S_{i+1} . S_{i+2}>

#     using <S_i . S_j> = 1/4 - P_singlet(i, j) for spin-1/2 pairs.

#     Parameters
#     ----------
#     singlet_weights : ndarray
#         Singlet probability for every bond, for every value of the
#         swept parameter. Shape (n_points, n_bonds), as returned by
#         compute_chain_singlet_weights.

#     bond_labels : list
#         Label of each bond (same length as the last axis of
#         `singlet_weights`), as returned by compute_chain_singlet_weights.

#     Returns
#     -------
#     order_parameter : ndarray
#         Shape (n_points, n_bonds - 1).

#     order_parameter_labels : list
#         Label of each pair of adjacent bonds compared.
#     """

#     correlations = 0.25 - singlet_weights

#     order_parameter = (
#         correlations[:, :-1]
#         - correlations[:, 1:]
#     )

#     order_parameter_labels = [
#         f"{bond_labels[i]} vs {bond_labels[i+1]}"
#         for i in range(len(bond_labels) - 1)
#     ]

#     return (
#         order_parameter,
#         order_parameter_labels,
#     )

