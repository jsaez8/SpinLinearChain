import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure

from .sweeps import SweepResults


def plot_energy_spectrum(results: SweepResults) -> tuple[Figure, Axes]:
    """
    Create a plot of the spin-chain energy spectrum.

    Returns the figure and axes so the caller can display, save, or customize
    the plot.
    """

    figure, axes = plt.subplots(figsize=(8, 6))
    axes.plot(results.parameter_values, results.eigenvalues)
    axes.set_xlabel(results.parameter_name)
    axes.set_ylabel("Energy/h (MHz)")
    return figure, axes

def plot_total_magnetization(results: SweepResults) -> tuple[Figure, Axes]:
    """
    Create a plot of the spin-chain total magnetization.

    Returns the figure and axes so the caller can display, save, or customize
    the plot.
    """

    figure, axes = plt.subplots(figsize=(8, 6))
    axes.plot(results.parameter_values, results.total_magnetizations)
    axes.set_xlabel(results.parameter_name)
    axes.set_ylabel("Total Magnetization")
    return figure, axes

def plot_local_magnetization(
    results: SweepResults,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """
    Plot local magnetization as site curves and as a heatmap.

    Returns one figure with two axes: the line plot and heatmap.
    """

    magnetizations = results.local_magnetizations
    site_numbers = np.arange(1, magnetizations.shape[1] + 1)

    figure, (line_axes, heatmap_axes) = plt.subplots(
        ncols=2,
        figsize=(14, 6),
        constrained_layout=True,
    )
    for site_index, site_number in enumerate(site_numbers):
        line_axes.plot(
            results.parameter_values,
            magnetizations[:, site_index],
            label=f"Site {site_number}",
        )
    line_axes.set_xlabel(results.parameter_name)
    line_axes.set_ylabel("Local Magnetization")
    line_axes.legend()

    color_limit = np.max(np.abs(magnetizations))
    if color_limit == 0:
        color_limit = 1.0

    mesh = heatmap_axes.pcolormesh(
        site_numbers,
        results.parameter_values,
        magnetizations,
        shading="auto",
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-color_limit, vcenter=0, vmax=color_limit),
    )
    heatmap_axes.set_xticks(site_numbers)
    heatmap_axes.set_xlabel("Site")
    heatmap_axes.set_ylabel(results.parameter_name)
    figure.colorbar(mesh, ax=heatmap_axes, label="Local Magnetization")
    return figure, (line_axes, heatmap_axes)