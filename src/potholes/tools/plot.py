import pathlib
from typing import Tuple

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt, colors

import matplotlib.ticker as mticker

from potholes.tools.generator import generate_sample, infer_fs
from potholes.tools.signal_tools import transform_to_db

axis_names = [
    "x_accel", "y_accel", "z_accel",
    "x_gyro", "y_gyro", "z_gyro"
]

acoussense_values = np.array([
    [24, 0, 124],
    [30, 0, 174],
    [41, 0, 250],
    [51, 26, 250],
    [62, 81, 250],
    [76, 127, 251],
    [90, 167, 251],
    [106, 210, 252],
    [98, 203, 192],
    [87, 187, 139],
    [82, 181, 102],
    [85, 190, 67],
    [95, 214, 57],
    [110, 232, 39],
    [156, 222, 51],
    [192, 235, 42],
    [225, 248, 44],
    [247, 255, 73],
    [224, 223, 37],
    [226, 185, 32],
    [219, 129, 26],
    [220, 68, 48]
])


def reconstruct_axes(Sxx, fs, nperseg, noverlap):
    n_freqs, n_times = Sxx.shape

    # Frequencies
    freqs = np.linspace(0, fs / 2, n_freqs)

    # Time bins (centered windows)
    hop = nperseg - noverlap
    times = (np.arange(n_times) * hop + nperseg / 2) / fs

    return freqs, times


def plot_axis(signal: np.ndarray, freqs: np.ndarray, t_spec: np.ndarray,
              axis_name: str, min_db: float, max_db: float, ax: plt.Axes, flip: bool = True, yaxis_values: np.ndarray = None)\
        -> Tuple[plt.Figure, plt.Axes]:
    signal_t = signal.T

    signal_db = transform_to_db(signal_t)

    acoussense_cmap = colors.ListedColormap(acoussense_values / 255.0, 'acoussense')

    pm = ax.imshow(signal_db,
                   origin='lower',
                   extent=(
                       freqs.min(), freqs.max(),
                       t_spec.max(), t_spec.min()
                   ),
                   cmap=acoussense_cmap,
                   vmin=min_db,
                   vmax=max_db,
                   interpolation='nearest',
                   aspect='auto')

    if flip:
        ax.invert_yaxis()

    if yaxis_values is not None:
        yaxis_values_rel = np.round(yaxis_values - yaxis_values.min())
        yaxis_desired_values = np.searchsorted(yaxis_values_rel, np.arange(1, t_spec.max()), side='left')
        ax.set_yticks(np.arange(1, t_spec.max()))
        ax.set_yticklabels([f"{t:.2f}s" for t in yaxis_values[yaxis_desired_values]], fontsize=12)
    else:
        ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))

    ax.tick_params(axis='y', labelsize=12)

    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
    ax.tick_params(axis='x', labelsize=10, rotation=0)

    return pm


def plot_sample(spectrogram_data: np.ndarray,
                spectrogram_params: dict,
                max_db: float = None,
                min_db: float = None):
    d = spectrogram_data[0]
    freqs, times = reconstruct_axes(Sxx=d,
                                    fs=spectrogram_params["freq"],
                                    nperseg=spectrogram_params["nperseg"],
                                    noverlap=spectrogram_params["noverlap"])

    if min_db is None or max_db is None:
        db_min = np.inf
        db_max = -np.inf
        for i in range(6):
            db = transform_to_db(spectrogram_data[i].T)
            db_min = min(db_min, np.nanmin(db))
            db_max = max(db_max, np.nanmax(db))
        if min_db is None:
            min_db = db_min
        if max_db is None:
            max_db = db_max

    nrows, ncols = 2, 3

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(14, 10),
                             constrained_layout=False
                             )

    fig.subplots_adjust(left=0.085, right=0.9, bottom=0.10, top=0.92, wspace=0.05, hspace=0.1)

    axes = axes.flatten()
    pms = []

    for i in range(6):
        r, c = divmod(i, ncols)

        pm = plot_axis(signal=spectrogram_data[i],
                       freqs=freqs,
                       t_spec=times,
                       max_db=max_db,
                       min_db=min_db,
                       axis_name=axis_names[i],
                       ax=axes[i],
                       )

        axes[i].set_title(axis_names[i])

        if c != 0:
            axes[i].tick_params(axis="y", labelleft=False)
            axes[i].set_ylabel("")
        else:
            axes[i].set_ylabel("Time [sec]", fontsize=14)

            # Only bottom row shows X tick labels + X axis label
        if r != nrows - 1:
            axes[i].tick_params(axis="x", labelbottom=False)
            axes[i].set_xlabel("")
        else:
            axes[i].set_xlabel("Frequency [Hz]", fontsize=14)

        pms.append(pm)

    cax = fig.add_axes([0.92, 0.15, 0.02, 0.70])
    cb = fig.colorbar(pms[0], cax=cax)
    cb.set_label('dB', fontsize=14)

    return fig


def plot_session_sample(session_data: pd.DataFrame,
                        start: int,
                        window_size: int,
                        ):
    rows_per_sec = infer_fs(session_data)
    rows_per_window = window_size * rows_per_sec

    start_idx = start * rows_per_sec
    end_idx = int(start_idx + rows_per_window)
    window_data = session_data.iloc[start_idx:end_idx]

    spectrogram_params = {
        "freq": rows_per_sec,
        "nperseg": 64,
        "noverlap": 48,
        "nfft": 64
    }

    sample = generate_sample(window_data,
                             freq=rows_per_sec,
                             nperseg=spectrogram_params["nperseg"],
                             noverlap=spectrogram_params["noverlap"],
                             nfft=spectrogram_params["nfft"])

    fig = plot_sample(sample, spectrogram_params)
    return fig


if __name__ == '__main__':
    import os
    import click

    def output_option(func):
        return click.option(
            "-o", "--output",
            type=click.Path(path_type=pathlib.Path, dir_okay=False, writable=True),
            default=None,
            help="Write the plot to this file (e.g. out.png). If omitted, show interactively.",
        )(func)

    @click.group()
    def cli(ctx, output: click.File):
        pass

    @cli.command(name="plot-sample")
    @click.argument("sample_file", type=click.File("rb"))
    @output_option
    def plot_file(sample_file: pathlib.Path, output: pathlib.Path | None):
        sample_path = os.path.dirname(sample_file.name)
        spec_params_file = os.path.join(sample_path, R"spectrogram_params.yaml")
        spec_params = yaml.safe_load(open(spec_params_file))
        data = np.load(sample_file.name)
        fig = plot_sample(data, spec_params)

        if output is not None:
            fig.savefig(output)
        else:
            fig.show()

    @cli.command(name="plot-session-sample")
    @click.argument("session_file", type=click.File("rb"))
    @output_option
    def plot_session_sample(session_file: pathlib.Path, output: pathlib.Path | None):
        session_data = pd.read_csv(session_file.name, parse_dates=["timestamp"])
        fig = plot_session_sample(session_data, start=0, window_size=20)

        if output is not None:
            fig.savefig(output)
        else:
            fig.show()

    cli()