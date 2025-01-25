import os
import sys

# Add 'libs' path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

from libs.utils.logger_config import get_logger
from libs.utils.plot_config import PlotConfig
from libs.utils.config_variables import MOTION_COLORS

module = "plotter"
logger = get_logger(module)

PlotConfig.setup_matplotlib()

def save_picture(fname, type_output, ax, **kwargs):
    """
    Save the current matplotlib figure to a file.

    Parameters
    ----------
    fname : str
        Full path to save the picture.
    type_output : str
        File format (e.g., 'png', 'svg').
    ax : matplotlib.axes.Axes
        The axes object to adjust the aspect ratio.
    **kwargs
        Additional keyword arguments passed to plt.savefig.
    """
    logger.info(f"Saving figure to {fname}")
    plt.savefig(fname, format=type_output, **kwargs)


def plot_time_series(
    y,
    x,
    title_x,
    title_y,
    unit,
    note_by_plot,
    folder,
    name,
    type_output,
    name_serie,
    figsize=(10, 8),
    pad=2.0,
):
    """
    Plot time series data for motion components.

    Parameters
    ----------
    y : list of np.ndarray
        List of y-values for each motion component.
    x : np.ndarray
        x-values (time).
    title_x : str
        Title for the x-axis.
    title_y : str
        Title for the y-axis.
    unit : str
        Unit of the y-values.
    note_by_plot : list of str
        Notes to display on each plot.
    folder : str
        Directory to save the plot.
    name : str
        Base name for the plot file.
    type_output : str
        File format for the plot (e.g., 'png', 'svg').
    name_serie : list of str
        Names for each series.
    figsize : tuple
        Size of the figure.
    """
    logger.info(f"Plotting time series for {name}")

    fig, axes = plt.subplots(3, 1, figsize=figsize)

    for i, ax in enumerate(axes):
        ax.text(
            0.01,
            0.95,
            name_serie[i],
            verticalalignment="top",
            horizontalalignment="left",
            transform=ax.transAxes,
            color="k",
            fontsize=10,
        )
        ax.text(
            0.81,
            0.95,
            f"{note_by_plot[i]}",
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes,
        )
        ax.plot(x, y[i], MOTION_COLORS[i], linewidth=0.4)
        ax.grid()
        if i == 1:
            ax.set_ylabel(f"{title_y} ({unit})")
        if i != 2:
            ax.set_xticklabels([])  # Remove x-axis labels for the first two plots
        else:
            ax.set_xlabel(title_x)

    os.makedirs(folder, exist_ok=True)
    save_picture(os.path.join(folder, f"{name}.{type_output}"), type_output, ax)


def plot_spectral_series(
    y,
    x,
    title_y,
    title_x,
    folder,
    name,
    type_output,
    name_serie,
    figsize=(10, 8),
    xlog=True,
    ylog=True,
):
    """
    Plot spectral series data for motion components.

    Parameters
    ----------
    y : list of np.ndarray
        List of y-values for each motion component.
    x : np.ndarray
        x-values (frequency).
    title_y : str
        Title for the y-axis.
    title_x : str
        Title for the x-axis.
    ylog : bool
        Whether to use a logarithmic scale for the y-axis.
    folder : str
        Directory to save the plot.
    name : str
        Base name for the plot file.
    type_output : str
        File format for the plot (e.g., 'png', 'svg').
    name_serie : list of str
        Names for each series.
    figsize : tuple
        Size of the figure.
    xlog : bool, optional
        Whether to use a logarithmic scale for the x-axis (default is True).
    """
    logger.info(f"Plotting spectral series for {name}")

    fig, ax = plt.subplots(figsize=figsize)

    for i, color in enumerate(MOTION_COLORS):
        ax.text(
            0.015,
            0.98 - i * 0.03,
            name_serie[i],
            verticalalignment="top",
            horizontalalignment="left",
            transform=ax.transAxes,
            color=color,
            fontsize=10,
        )
        ax.plot(x, y[i], color, linewidth=1)

    if xlog:
        ax.set(xscale="log")
    if ylog:
        ax.set(yscale="log")

    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_major_locator(
        ticker.LogLocator(base=10.0, subs=[], numticks=10)
    )  # Ensure x-axis shows only 10^n values

    ax.set_ylabel(title_y)
    ax.set_xlabel(title_x)
    ax.grid()

    os.makedirs(folder, exist_ok=True)
    save_picture(os.path.join(folder, f"{name}.{type_output}"), type_output, ax)


# Example usage
if __name__ == "__main__":
    from libs.utils.tools import name_orientation

    quake = "sample"
    folder = os.path.join(".", "outputs", module, quake)
    type_output = "svg"
    name_serie = name_orientation("sensor")

    # Example data
    y = [np.random.randn(100) for _ in range(3)]
    x = np.linspace(0, 10, 100)
    title_x = "Tiempo (s)"
    title_y = "Aceleración"
    unit = "cm/s²"
    name = "example_ACC"
    note_by_plot = [
        f'PGA:{format(round(np.max(np.abs(arr)), 2), ".2f").replace(".", ",")} {unit}'
        for arr in y
    ]

    # Plot time series
    plot_time_series(
        y,
        x,
        title_x,
        title_y,
        unit,
        note_by_plot,
        folder,
        name,
        type_output,
        name_serie,
        figsize=(8, 5),
        pad=2.0,
    )

    # Example data for spectral series
    y = [np.abs(np.fft.fft(arr)) for arr in y]
    x = np.fft.fftfreq(len(x), d=(x[1] - x[0]))
    title_y = "Amplitud de Fourier (cm/s²·s)"
    title_x = "Frecuencia (Hz)"
    ylog = True
    name = "example_FOU"

    # Plot spectral series
    plot_spectral_series(
        y,
        x,
        title_y,
        title_x,
        folder,
        name,
        type_output,
        name_serie,
        figsize=(8, 5),
        ylog=ylog,
    )
