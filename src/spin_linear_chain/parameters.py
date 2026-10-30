from dataclasses import dataclass

import numpy as np


@dataclass
class SpinChainParameters:
    """Parameters defining an alternating spin chain."""

    n_sites: int = 6

    # Exchange interaction (MHz)
    J: float = 60.0
    delta: float = 0.0

    # Magnetic field (Tesla)
    magnetic_field: float = 0.005

    # Spin-orbit coupling (MHz/T)
    delta_STmin: float = 0.0

    # g-factor
    g_factors: np.ndarray | float = 0.2
    
    # Optional multiplicative disorder
    exchange_disorder: np.ndarray | None = None

    # Spin model
    model: str = "Heisenberg"


def default_spin_chain_params(
    n_sites: int = 6,
    J: float = 30.0,
    delta: float = 0.0,
    magnetic_field: float = 0.005,
    delta_STmin: float = 0.0,
    model: str = "Heisenberg",
) -> SpinChainParameters:
    """Build SpinChainParameters with uniform g-factors."""

    return SpinChainParameters(
        n_sites=n_sites,
        J=J,
        delta=delta,
        magnetic_field=magnetic_field,
        delta_STmin=delta_STmin,
        g_factors=np.full(n_sites, 0.2),
        exchange_disorder=None,
        model=model,
    )