# %%
import numpy as np

from spin_linear_chain.parameters import SpinChainParameters
from spin_linear_chain.plotting import (
    plot_energy_spectrum,
    plot_local_magnetization,
    plot_local_magnetization_heatmap,
    plot_total_magnetization,
)
from spin_linear_chain.sweeps import run_delta_sweep, run_magnetic_field_sweep

spin_chain_params = SpinChainParameters(
    n_sites=8,
    delta=0.1,
    J=10.0,
    magnetic_field=0.005
)
# %%
results = run_delta_sweep(
    params=spin_chain_params,
    deltas=np.linspace(-1.0, 1.0, 30)
)
plot_energy_spectrum(results)
plot_total_magnetization(results)
plot_local_magnetization(results)
plot_local_magnetization_heatmap(results)
# %%
results = run_magnetic_field_sweep(
        params=spin_chain_params,
        magnetic_fields=np.linspace(-0.05, 0.05, 30),
        eigvals=0,
    )
# %%
plot_energy_spectrum(results)
plot_total_magnetization(results)
plot_local_magnetization(results)
plot_local_magnetization_heatmap(results)
# %%
