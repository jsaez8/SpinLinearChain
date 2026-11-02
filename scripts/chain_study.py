import numpy as np

from spin_linear_chain.parameters import SpinChainParameters
from spin_linear_chain.sweeps import run_delta_sweep

spin_chain_params = SpinChainParameters(
    n_sites=4,
    delta=0.1,
    magnetic_field=0.0
)

results = run_delta_sweep(
    params=spin_chain_params,
    deltas=np.linspace(-1.0, 1.0, 11)
)