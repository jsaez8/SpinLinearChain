from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def format_filename_value(value: float) -> str:
    """Format a number compactly and safely for a filename."""
    return f"{value:g}".replace("-", "m").replace(".", "p")


def save_figures(
    figures: Mapping[str, Figure],
    sweep_name: str,
    output_dir: Path,
) -> None:
    """Save named figures in an output directory and close them."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, figure in figures.items():
        path = output_dir / f"{sweep_name}_{name}.png"
        figure.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(figure)
