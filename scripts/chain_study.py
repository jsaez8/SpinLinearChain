# %%
from pathlib import Path

import numpy as np

from spin_linear_chain.parameters import SpinChainParameters
from spin_linear_chain.plotting import (
    plot_energy_spectrum,
    plot_local_magnetization,
    plot_total_magnetization,
)
from spin_linear_chain.sweeps import (
    run_delta_sweep,
    run_magnetic_field_sweep,
)

spin_chain_params = SpinChainParameters(
    n_sites=8,
    delta=0.1,
    J=10.0,
    magnetic_field=0.005
)

def format_filename_value(value: float) -> str:
    """Format a float value for use in a filename.

    This function replaces the decimal point with 'p' to avoid issues with
    file systems that may not handle periods well in filenames.
    """
    return str(value).replace(".", "p")

run_name = (
    f"{spin_chain_params.model}"
    f"_N{spin_chain_params.n_sites}"
    f"_J{format_filename_value(spin_chain_params.J)}"
    f"_delta{format_filename_value(spin_chain_params.delta)}"
    f"_B{format_filename_value(spin_chain_params.magnetic_field)}"
)
output_dir = Path(__file__).resolve().parents[1] / "outputs" / run_name
output_dir.mkdir(parents=True, exist_ok=True)

# %%
results = run_delta_sweep(
    params=spin_chain_params,
    deltas=np.linspace(-1.0, 1.0, 30)
)
fig_spectrum, _ = plot_energy_spectrum(results)
fig_total_magnetization, _ = plot_total_magnetization(results)
fig_local_magnetization, _ = plot_local_magnetization(results)
# %%
results = run_magnetic_field_sweep(
        params=spin_chain_params,
        magnetic_fields=np.linspace(-0.05, 0.05, 30),
        eigvals=0,
    )
# %%
fig_spectrum, _ = plot_energy_spectrum(results)
fig_total_magnetization, _ = plot_total_magnetization(results)
fig_local_magnetization, _ = plot_local_magnetization(results)
# %%
