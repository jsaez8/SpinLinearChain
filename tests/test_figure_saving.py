import matplotlib.pyplot as plt

from spin_linear_chain.figure_saving import format_filename_value, save_figures


def test_format_filename_value_is_readable_and_path_safe():
    assert format_filename_value(0.005) == "0p005"
    assert format_filename_value(-1.25) == "m1p25"


def test_save_figures_writes_named_png_and_closes_figure(tmp_path):
    figure, _ = plt.subplots()

    save_figures({"spectrum": figure}, "delta", tmp_path)

    assert (tmp_path / "delta_spectrum.png").is_file()
    assert not plt.fignum_exists(figure.number)
