from datetime import datetime
import json
import pathlib
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt, colors

import matplotlib.ticker as mticker
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from potholes.tools.generator import generate_sample, infer_fs, generate_window_sample
from potholes.tools.signal_tools import transform_to_db

LABEL_COLORS = {
    "pothole": "red",
    "other": "blue",
    "manhole": "purple",
    "speed_bump": "green",
}

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
        -> AxesImage:
    signal_t = signal.T

    signal_db = transform_to_db(signal_t)

    acoussense_cmap = colors.ListedColormap(acoussense_values / 255.0, 'acoussense')

    pm = ax.imshow(signal_db,
                   origin='lower',
                   extent=(
                       freqs.min(), freqs.max(),
                       t_spec.min(), t_spec.max()
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


def plot_title(fig: plt.Figure, metadata: dict):
    sensor_name: str = metadata["sensor_name"]

    start_dt = datetime.fromisoformat(metadata["start_timestamp"])
    end_dt = datetime.fromisoformat(metadata["end_timestamp"])

    date_str = start_dt.strftime("%Y-%m-%d")
    start_time = start_dt.strftime("%H:%M:%S.%f")[:-3]
    end_time = end_dt.strftime("%H:%M:%S.%f")[:-3]

    title = (
        f"{sensor_name.replace(' ', '')} | "
        f"{date_str} {start_time}–{end_time}"
    )

    labels = { x['label']: LABEL_COLORS[x['label']] for x in metadata['labels'] }
    add_label_legend(fig, labels)

    fig.text(0.5, 0.97, title,
             ha="center",
             va="center",
             fontsize=16)

    return fig


def add_label_legend(fig, label_colors: dict,
                     loc="center",
                     bbox_to_anchor=(0.5, 0.94),
                     fontsize=12):

    if len(label_colors) == 0:
        return

    handles = [
        Line2D(
            [0], [0],
            color=color,
            lw=4,
            label=label
        )
        for label, color in label_colors.items()
    ]

    fig.legend(
        handles=handles,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        ncol=len(label_colors),
        frameon=False,
        fontsize=fontsize
    )


def extract_label_seconds(metadata: dict) -> list[tuple[float, str]]:
    start_dt = datetime.fromisoformat(metadata["start_timestamp"])

    out: list[tuple[float, str]] = []
    for item in metadata["labels"]:
        lbl = item["label"]
        t_dt = datetime.fromisoformat(item["timestamp_label"])
        t0 = (t_dt - start_dt).total_seconds()

        out.append((t0, lbl))

    return out


def add_label_boxes(axes: list[plt.Axes],
                    freqs: np.ndarray,
                    t_spec: np.ndarray,
                    label_events: list[tuple[float, str]],
                    box_len_s: float = 1.0,
                    linewidth: float = 1.0):
    """
    Draw a full-width rectangle for each label, 1 second tall, on every axis.
    """
    if not label_events:
        return

    x0 = float(freqs.min())
    w = float(freqs.max() - freqs.min())

    t_min = float(min(t_spec.min(), t_spec.max()))
    t_max = float(max(t_spec.min(), t_spec.max()))

    for t0, lbl in label_events:
        color = LABEL_COLORS.get(lbl, "white")

        y0 = t0 - box_len_s / 2
        y0 = max(t_min, min(y0, t_max - box_len_s))

        for ax in axes:
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    w,
                    box_len_s,
                    fill=False,
                    edgecolor=color,
                    linewidth=linewidth,
                    zorder=10,
                )
            )


def plot_sample(spectrogram_data: np.ndarray,
                spectrogram_params: dict,
                label_events: list[tuple[float, str]] = None,
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

    fig.subplots_adjust(left=0.085, right=0.915, bottom=0.08, top=0.90, wspace=0.05, hspace=0.1)

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

    if label_events:
        add_label_boxes(
            axes=list(axes),
            freqs=freqs,
            t_spec=times,
            label_events=label_events,
            box_len_s=1.0
        )

    cax = fig.add_axes((0.93, 0.15, 0.02, 0.70))
    cb = fig.colorbar(pms[0], cax=cax)
    cb.set_label("")  # remove side label
    cax.set_title("dB", fontsize=14, pad=8)

    return fig


def plot_session(session_file: pathlib.Path,
                        start: int,
                        window_size: int):
    sensor_name = session_file.name.split("_")[1]
    session_data = pd.read_csv(session_file, parse_dates=["timestamp"])

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

    sample, metadata = generate_window_sample(window_data,
                                          start_idx=start_idx,
                                          fs=rows_per_sec,
                                          window_size=window_size,
                                          step=0,
                                          spectrogram_params=spectrogram_params,
                                          )
    metadata["sensor_name"] = sensor_name
    label_events = extract_label_seconds(metadata)
    fig = plot_sample(sample, spectrogram_params, label_events=label_events)
    fig = plot_title(fig, metadata=metadata)
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
    def cli():
        pass

    @cli.command(name="plot-sample")
    @click.argument("sample_file",
                    type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True, path_type=pathlib.Path))
    @output_option
    def plot_file(sample_file: pathlib.Path, output: Optional[pathlib.Path]):
        spec_params_file = os.path.join(sample_file.parent, R"spectrogram_params.yaml")
        spec_params = yaml.safe_load(open(spec_params_file))
        d = np.load(sample_file, allow_pickle=False)
        data = d["sample"]
        metadata = json.loads(d["meta"].item())

        label_events = extract_label_seconds(metadata)
        fig = plot_sample(data, spec_params, label_events=label_events)
        fig = plot_title(fig, metadata=metadata)

        if output is not None:
            os.makedirs(output.parent, exist_ok=True)
            fig.savefig(output)
        else:
            plt.show()

    @cli.command(name="plot-session-sample")
    @click.argument("session_file",
                    type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True, path_type=pathlib.Path))
    @click.argument("start_second", type=int, required=True)
    @click.option("-w", "--window-size", type=int, required=False, default=20,
                  help="Window size")
    @output_option
    def plot_session_sample(session_file: pathlib.Path, start_second: int, window_size: int, output: Optional[pathlib.Path]):
        fig = plot_session(session_file, start=start_second, window_size=window_size)

        if output is not None:
            os.makedirs(output.parent, exist_ok=True)
            fig.savefig(output)
        else:
            plt.show()

    cli()
