import numpy as np


def alternating_couplings(
    J: float,
    delta: float,
    n_sites: int,
    disorder=None,
    model="Heisenberg",
):
    """
    Compute nearest-neighbour exchange couplings.
    """

    if model == "Heisenberg":
        z_factor = 1.0

    elif model == "XY":
        z_factor = 0.0

    else:
        raise ValueError(model)

    J_even = J * (1 + delta)
    J_odd = J * (1 - delta)

    exchange_x = np.zeros(n_sites - 1)
    exchange_y = np.zeros(n_sites - 1)
    exchange_z = np.zeros(n_sites - 1)

    exchange_x[::2] = J_even
    exchange_y[::2] = J_even
    exchange_z[::2] = z_factor * J_even

    exchange_x[1::2] = J_odd
    exchange_y[1::2] = J_odd
    exchange_z[1::2] = z_factor * J_odd

    if disorder is not None:

        exchange_x *= disorder
        exchange_y *= disorder
        exchange_z *= disorder

    return exchange_x, exchange_y, exchange_z