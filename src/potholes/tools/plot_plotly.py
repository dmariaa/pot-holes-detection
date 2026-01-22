from typing import Optional

import numpy as np
import yaml
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from potholes.tools.signal_tools import transform_to_db


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


def acoussense_colorscale(vals_rgb: np.ndarray):
    vals = vals_rgb / 255.0
    n = len(vals)
    cs = []
    for i, (r, g, b) in enumerate(vals):
        t = i / (n - 1)
        cs.append([t, f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"])
    return cs


def reconstruct_axes(Sxx, fs, nperseg, noverlap):
    n_freqs, n_times = Sxx.shape

    # Frequencies
    freqs = np.linspace(0, fs / 2, n_freqs)

    # Time bins (centered windows)
    hop = nperseg - noverlap
    times = (np.arange(n_times) * hop + nperseg / 2) / fs

    return freqs, times


def plot_sample(spectrogram_data: np.ndarray,
                spectrogram_params: dict,
                min_db: Optional[float] = None,
                max_db: Optional[float] = None,
                flip: bool = True):

    axis_names = [
        "x_accel", "y_accel", "z_accel",
        "x_gyro", "y_gyro", "z_gyro"
    ]

    colorscale = acoussense_colorscale(acoussense_values)

    freqs, t_spec = reconstruct_axes(Sxx=spectrogram_data[0],
                                    fs=spectrogram_params["freq"],
                                    nperseg=spectrogram_params["nperseg"],
                                    noverlap=spectrogram_params["noverlap"])

    db_list = []
    db_min = np.inf
    db_max = -np.inf
    for i in range(6):
        # spectrogram_window[i] is (n_freq, n_time); transpose to (n_time, n_freq)
        db = transform_to_db(spectrogram_data[i].T)
        db_list.append(db)
        db_min = min(db_min, float(np.nanmin(db)))
        db_max = max(db_max, float(np.nanmax(db)))

    if min_db is None:
        min_db = db_min
    if max_db is None:
        max_db = db_max

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=axis_names,
        horizontal_spacing=0.02,
        vertical_spacing=0.05
    )

    positions = [
        (1, 1), (1, 2), (1, 3),
        (2, 1), (2, 2), (2, 3),
    ]

    for i, (r, c) in enumerate(positions):
        show_scale = (i == 2)
        colorbar = dict(title="dB", len=0.85, x=1.02, y=0.5) if show_scale else None

        fig.add_trace(
            go.Heatmap(
                z=db_list[i],  # (n_time, n_freq)
                x=freqs,
                y=t_spec,
                colorscale=colorscale,
                zmin=min_db,
                zmax=max_db,
                showscale=show_scale,
                colorbar=colorbar,
                hovertemplate="f=%{x:.1f}Hz<br>t=%{y:.2f}s<br>dB=%{z:.2f}<extra></extra>",
            ),
            row=r, col=c
        )

    if not flip:
        # time increases top -> bottom
        for r in (1, 2):
            for c in (1, 2, 3):
                fig.update_yaxes(autorange="reversed", row=r, col=c)

    # Axis visibility rules
    for r in (1, 2):
        for c in (1, 2, 3):
            if c != 1:
                fig.update_yaxes(showticklabels=False, title_text="", row=r, col=c)
            else:
                fig.update_yaxes(title_text="Time [sec]",
                                 ticks="outside",
                                 ticklen=6,
                                 tickwidth=1,
                                 tickcolor="black",
                                 row=r,
                                 col=c)

            if r != 2:
                fig.update_xaxes(showticklabels=False, title_text="", row=r, col=c)
            else:
                fig.update_xaxes(title_text="Frequency [Hz]",
                                 ticks="outside",
                                 ticklen=6,
                                 tickwidth=1,
                                 tickcolor="black",
                                 row=r,
                                 col=c)

            fig.update_xaxes(nticks=6,
                             showline=True,
                             linecolor="black",
                             linewidth=1,
                             mirror=True,
                             row=r,
                             col=c)

            fig.update_yaxes(nticks=20,
                             showline=True,
                             linecolor="black",
                             linewidth=1,
                             mirror=True,
                             row=r,
                             col=c)

    fig.update_layout(
        width=1100,
        height=650,
        margin=dict(l=60, r=100, t=60, b=50),
    )

    fig.show()
    return fig

if __name__ == '__main__':
    import os

    file = "data_1950_20_1_speed_bump.npy"
    sample_file = os.path.join(R"data/Data_20251226/session_STBPRO3@71E957_20251226_114336/", file)
    spec_params_file = R"data/Data_20251226/session_STBPRO3@71E957_20251226_114336/spectrogram_params.yaml"
    spec_params = yaml.safe_load(open(spec_params_file))
    data = np.load(sample_file)
    plot_sample(data, spec_params)