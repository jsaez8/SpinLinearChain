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

def plot_local_magnetization(results: SweepResults) -> tuple[Figure, Axes]:
    """
    Create a plot of the spin-chain local magnetization.

    Returns the figure and axes so the caller can display, save, or customize
    the plot.
    """

    figure, axes = plt.subplots(figsize=(8, 6))
    for site in range(results.local_magnetizations.shape[1]):
        axes.plot(
            results.parameter_values,
            results.local_magnetizations[:, site],
            label=f"Site {site}",
        )
    axes.set_xlabel(results.parameter_name)
    axes.set_ylabel("Local Magnetization")
    axes.legend()
    return figure, axes

def plot_local_magnetization_heatmap(results: SweepResults) -> tuple[Figure, Axes]:
    """
    Create a heatmap of the spin-chain local magnetization.

    Returns the figure and axes so the caller can display, save, or customize
    the plot.
    """

    magnetizations = results.local_magnetizations
    color_limit = np.max(np.abs(magnetizations))
    if color_limit == 0:
        color_limit = 1.0

    site_numbers = np.arange(1, magnetizations.shape[1] + 1)

    figure, axes = plt.subplots(figsize=(8, 6))
    axes.pcolormesh(
        site_numbers,
        results.parameter_values,
        magnetizations,
        shading="auto",
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-color_limit, vcenter=0, vmax=color_limit),
    )

    axes.set_xticks(site_numbers)
    axes.set_xlabel("Site")
    axes.set_ylabel(results.parameter_name)
    figure.colorbar(axes.collections[0], ax=axes, label="Local Magnetization")
    return figure, axes