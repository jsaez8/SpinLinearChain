from dataclasses import dataclass

from qutip import qeye, sigmax, sigmay, sigmaz, tensor


@dataclass
class SpinOperators:
    sx: list
    sy: list
    sz: list


def build_spin_operators(n_sites: int) -> SpinOperators:
    """
    Construct all single-spin Pauli operators.
    """

    sx = []
    sy = []
    sz = []

    for site in range(n_sites):

        operator_list = [qeye(2)] * n_sites

        operator_list[site] = sigmax()
        sx.append(tensor(operator_list))

        operator_list[site] = sigmay()
        sy.append(tensor(operator_list))

        operator_list[site] = sigmaz()
        sz.append(tensor(operator_list))

    return SpinOperators(
        sx=sx,
        sy=sy,
        sz=sz,
    )