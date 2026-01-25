from datetime import datetime
from typing import Any, List, Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors, animation, gridspec
from matplotlib.patches import Rectangle
from pandas import DataFrame
from scipy.signal import stft
from tqdm import tqdm
import pyproj
import contextily as cx

from potholes.tools.generator import infer_fs
from potholes.tools.gps_tools import filter_by_distance
from potholes.tools.plot import axis_names, acoussense_values, LABEL_COLORS
from potholes.tools.signal_tools import transform_to_db


def setup_figure_with_map():
    fig = plt.figure(figsize=(26, 10))
    gs = gridspec.GridSpec(
        2, 5,                   # 2 rows, 4 cols
        width_ratios=[1.24, 0.12, 1, 1, 1],  # left map wider
        wspace=0.04, hspace=0.10,
        left=0.012, right=0.945, bottom=0.06, top=0.92
    )

    ax_map = fig.add_subplot(gs[:, 0])  # span both rows on left

    ax_spec = []
    ax_spec.append(fig.add_subplot(gs[0, 2]))
    ax_spec.append(fig.add_subplot(gs[0, 3]))
    ax_spec.append(fig.add_subplot(gs[0, 4]))
    ax_spec.append(fig.add_subplot(gs[1, 2]))
    ax_spec.append(fig.add_subplot(gs[1, 3]))
    ax_spec.append(fig.add_subplot(gs[1, 4]))

    return fig, ax_map, ax_spec


def add_colorbar(fig: plt.Figure, ims: List[plt.Axes]):
    cax = fig.add_axes([0.955, 0.15, 0.018, 0.70])
    cb = fig.colorbar(ims[0], cax=cax)
    # cb.set_label("dB", fontsize=14)
    cb.set_label("")  # remove side label
    cax.set_title("dB", fontsize=14, pad=8)


def prepare_gps_for_plot(df: pd.DataFrame) -> tuple[DataFrame, Any]:
    gps = (
        df.dropna(subset=["latitude", "longitude"])
          .sort_values("timestamp")
          .copy()
    )

    gps_unique = gps.loc[
        (gps["latitude"].diff().ne(0)) |
        (gps["longitude"].diff().ne(0))
    ].copy()

    gps_unique = filter_by_distance(gps_unique, min_meters=3)

    return gps, gps_unique


def set_equal_aspect_fill(ax, x, y, pad_frac=0.0):
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))

    dx0 = xmax - xmin
    dy0 = ymax - ymin
    if dx0 <= 0 or dy0 <= 0:
        return

    # optional padding as fraction of span
    x0, x1 = xmin - pad_frac * dx0, xmax + pad_frac * dx0
    y0, y1 = ymin - pad_frac * dy0, ymax + pad_frac * dy0

    dx = x1 - x0
    dy = y1 - y0

    ax.figure.canvas.draw()
    bbox = ax.get_window_extent()
    box_aspect = bbox.height / bbox.width  # ✅ y/x

    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)

    # Make dy/dx match box_aspect with MINIMUM expansion
    if (dy / dx) < box_aspect:
        # too "wide" in data terms -> expand y
        new_dy = box_aspect * dx
        y0, y1 = cy - 0.5 * new_dy, cy + 0.5 * new_dy
    else:
        # too "tall" -> expand x
        new_dx = dy / box_aspect
        x0, x1 = cx - 0.5 * new_dx, cx + 0.5 * new_dx

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("auto")


def add_route_axes(session_data: pd.DataFrame, ax_map: plt.Axes):
    """
    Draws route line + labeled points + basemap tiles, and returns an updater for the marker.
    Uses Web Mercator (EPSG:3857) so tiles align correctly.
    """
    gps, gps_unique = prepare_gps_for_plot(session_data)

    session_e0 = session_data["elapsed"].iloc[0]
    t_gps = (gps_unique["elapsed"].to_numpy(np.int64) - np.int64(session_e0)) / 1e9

    ax_map.set_title("GPS route", fontsize=14)

    # ---- Project WGS84 lon/lat -> Web Mercator meters (EPSG:3857) ----
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    lon_u = gps_unique["longitude"].to_numpy(dtype=float)
    lat_u = gps_unique["latitude"].to_numpy(dtype=float)
    x_u, y_u = transformer.transform(lon_u, lat_u)  # meters

    # Route polyline (above tiles)
    route_line, = ax_map.plot(x_u, y_u, linewidth=2, zorder=5)

    # Labeled points (also projected)
    labeled = gps[gps["label"].notna()].copy()
    label_artists = []
    if not labeled.empty:
        # Project labeled points
        lon_l = labeled["longitude"].to_numpy(dtype=float)
        lat_l = labeled["latitude"].to_numpy(dtype=float)
        x_l, y_l = transformer.transform(lon_l, lat_l)

        # We’ll scatter per label for legend colors, but need to select by mask
        labels_str = labeled["label"].astype(str).to_numpy()

        for label, color in LABEL_COLORS.items():
            m = labels_str == label
            if not np.any(m):
                continue
            sc = ax_map.scatter(
                np.asarray(x_l)[m],
                np.asarray(y_l)[m],
                s=24,
                c=color,
                label=label,
                zorder=6
            )
            label_artists.append(sc)

        # Unknown labels -> gray
        known = set(LABEL_COLORS.keys())
        m_other = ~np.isin(labels_str, list(known))
        if np.any(m_other):
            sc = ax_map.scatter(
                np.asarray(x_l)[m_other],
                np.asarray(y_l)[m_other],
                s=18,
                c="gray",
                label="unknown",
                zorder=6
            )
            label_artists.append(sc)

        ax_map.legend(loc="lower left", fontsize=10, framealpha=0.9)

    # Current position marker (projected) - updated each frame
    current_pt, = ax_map.plot([x_u[0]], [y_u[0]], marker="o", markersize=8, zorder=7)

    # Make it look “map-like”
    ax_map.grid(False)

    ax_map.tick_params(axis="y", labelleft=False)
    ax_map.set_ylabel("")
    ax_map.tick_params(axis="x", labelbottom=False)
    ax_map.set_xlabel("")

    #cset_equal_aspect_fill(ax_map, x_u, y_u, pad_frac=0)
    xmin, xmax = float(np.min(x_u)), float(np.max(x_u))
    ymin, ymax = float(np.min(y_u)), float(np.max(y_u))

    pad_frac = 0.0  # or 0.0 if you want tight
    dx = xmax - xmin
    dy = ymax - ymin

    ax_map.set_xlim(xmin - pad_frac * dx, xmax + pad_frac * dx)
    ax_map.set_ylim(ymin - pad_frac * dy, ymax + pad_frac * dy)

    ax_map.set_aspect("auto")  # <-- key line

    # ---- Add basemap tiles underneath ----
    # NOTE: this fetches tiles from the internet unless cached.
    # Add tiles AFTER limits/aspect so it uses the right extent.
    cx.add_basemap(
        ax_map,
        source=cx.providers.OpenStreetMap.Mapnik,
        crs="EPSG:3857",
        reset_extent=False,
        zorder=0
    )

    def update_route_marker(t0: float, t1: float):
        """
        Updates the current point at the mid time of [t0, t1]
        """
        t_mid = 0.5 * (t0 + t1)
        j = int(np.searchsorted(t_gps, t_mid, side="left"))
        j = int(np.clip(j, 0, len(t_gps) - 1))
        current_pt.set_data([x_u[j]], [y_u[j]])

    return update_route_marker


def compute_session_stft(session_data: pd.DataFrame):
    freq = infer_fs(session_data)

    spectrogram_params = {
        "freq": freq,
        "nperseg": 64,
        "noverlap": 48,
        "nfft": 64
    }

    Z_list = []
    f_out = None
    t_out = None

    for name in axis_names:
        x = session_data[name].to_numpy(dtype=float)

        f, t, Z = stft(
            x,
            fs=freq,
            window="hann",
            nperseg=spectrogram_params["nperseg"],
            noverlap=spectrogram_params["noverlap"],
            nfft=spectrogram_params["nfft"],
            boundary=None,
            padded=False,
            detrend=False
        )

        mag = np.abs(Z)
        Z_list.append(mag)

        if f_out is None:
            f_out, t_out = f, t

    S = np.stack(Z_list, axis=0)
    return f_out, t_out, S

# TODO: This code is almost duplicated in plot.py,
#  refactor to reuse
def extract_label_seconds(df: pd.DataFrame) -> list[tuple[float, str]]:
    start_dt = pd.to_datetime(df['timestamp'].iloc[0])
    labels_data = df.loc[df['label'].notna()]
    out: list[tuple[float, str]] = []
    for item in labels_data.to_dict(orient="records"):
        lbl = item["label"]
        t_dt = datetime.fromisoformat(item["timestamp_label"])
        t0 = (t_dt - start_dt).total_seconds()
        out.append((t0, lbl))

    return out


# TODO: This code is almost duplicated in plot.py,
#  refactor to reuse
def add_label_rectangles(
    axes: Iterable[plt.Axes],
    freqs: np.ndarray,
    events: list[tuple[float, str]],
    *,
    label_colors: dict[str, str],
    box_len_s: float = 1.0,
    linewidth: float = 1.0,
):
    """
    Adds rectangles to all axes. Returns a structure you can use later (video).
    Returns: list of (t_center, [rect_per_axis])
    """
    if not events:
        return []

    x0 = float(freqs.min())
    w = float(freqs.max() - freqs.min())

    out = []
    for ev in events:
        color = label_colors.get(ev[1], "white")
        y0 = float(ev[0] - box_len_s / 2)

        rects = []
        for ax in axes:
            rect = Rectangle(
                (x0, y0), w, box_len_s,
                fill=False, edgecolor=color,
                linewidth=linewidth, zorder=10
            )
            ax.add_patch(rect)
            rects.append(rect)

        out.append((ev[0], rects))

    return out


def add_spectrogram_axes(session_data: pd.DataFrame,
                         axes: List[plt.Axes],
                         window_size: int,
                         hop_seconds: float,
                         min_db: float,
                         max_db: float):
    freqs, t_stft, S = compute_session_stft(session_data)
    S_db = transform_to_db(S)

    if min_db is None:
        min_db = float(np.nanmin(S_db))
    if max_db is None:
        max_db = float(np.nanmax(S_db))

    win = float(window_size)
    hop = float(hop_seconds)

    dt_frame = float(np.median(np.diff(t_stft)))
    win_bins = max(1, int(round(win / dt_frame)))
    hop_bins = max(1, int(round(hop / dt_frame)))
    start_bins = list(range(0, max(1, t_stft.size - win_bins + 1), hop_bins))

    acoussense_cmap = colors.ListedColormap(acoussense_values / 255.0, 'acoussense')
    s0 = start_bins[0]
    s1 = min(t_stft.size, s0 + win_bins)

    # time increasing upwards (matches your current plot style)
    def extent_for_slice(a, b):
        # y goes from later -> earlier, and then we invert the axis
        return (freqs.min(), freqs.max(), t_stft[a], t_stft[b - 1])

    ncols = 3
    nrows = 2
    ims = []
    for i in range(6):
        ax = axes[i]
        sl = S_db[i, :, s0:s1]  # [freq, time]
        # imshow expects [rows, cols] -> we want time on Y, so transpose to [time, freq]
        img = ax.imshow(
            sl.T,
            origin="lower",
            extent=extent_for_slice(s0, s1),
            cmap=acoussense_cmap,
            vmin=min_db,
            vmax=max_db,
            interpolation="nearest",
            aspect="auto"
        )
        ax.invert_yaxis()
        ax.set_title(axis_names[i])

        # labels like your style
        r, c = divmod(i, ncols)
        if c != 0:
            ax.tick_params(axis="y", labelleft=False)
            ax.set_ylabel("")
        else:
            ax.set_ylabel("Time [sec]", fontsize=14)

        if r != nrows - 1:
            ax.tick_params(axis="x", labelbottom=False)
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Frequency [Hz]", fontsize=14)

        ims.append(img)

    label_events = extract_label_seconds(df=session_data)
    label_patches = add_label_rectangles(
        axes=axes, freqs=freqs, events=label_events,
        label_colors=LABEL_COLORS, box_len_s=1.0
    )

    def update_spectrogram_axes(frame_idx: int):
        s0 = start_bins[frame_idx]
        s1 = min(t_stft.size, s0 + win_bins)

        for i in range(6):
            sl = S_db[i, :, s0:s1]  # [freq, time]
            ims[i].set_data(sl.T)  # [time, freq]
            ims[i].set_extent(extent_for_slice(s0, s1))

        t0 = float(t_stft[s0])
        t1 = float(t_stft[s1 - 1])

        half = 0.5
        for t_center, rects in label_patches:
            visible = not ((t_center + half) < t0 or (t_center - half) > t1)
            for r in rects:
                r.set_visible(visible)

        return t0, t1

    return ims, len(start_bins), update_spectrogram_axes


def generate_session_video(session_data: pd.DataFrame,
                           out_mp4: str,
                           window_size: int,
                           hop_seconds: float = 0.25,
                           min_db: float | None = None,
                           max_db: float | None = None,
                           title_prefix: str = "Session Spectrogram"):
    freq = infer_fs(session_data)
    fig, ax_map, axes = setup_figure_with_map()

    # Spectrogram plotting
    ims, total_frames, update_spectrogram_axes = add_spectrogram_axes(session_data,
                               window_size=window_size,
                               hop_seconds=hop_seconds,
                               axes=axes,
                               min_db=min_db,
                               max_db=max_db)

    # GPS plotting
    update_route_marker = add_route_axes(session_data, ax_map=ax_map)

    add_colorbar(fig, ims=ims)

    # add time text overlay
    time_text = fig.text(0.5, 0.97, "",
                         ha="center",
                         va="center",
                         fontsize=16)

    def update(frame_idx: int):
        t0, t1 = update_spectrogram_axes(frame_idx)
        update_route_marker(t0, t1)
        time_text.set_text(f"{title_prefix}  |  t = {t0:.2f}s → {t1:.2f}s  |  fs≈{freq}Hz")
        return ims + [time_text]


    writer = animation.FFMpegWriter(
        fps=freq // 2,
        codec="libx264",
        bitrate=2500
    )

    with writer.saving(fig, out_mp4, dpi=100):
        for i in tqdm(range(total_frames), desc="Rendering video"):
            update(i)  # your update(frame_idx)
            writer.grab_frame()

    plt.close(fig)
    return out_mp4


def save_debug_frame(session_data: pd.DataFrame,
                     out_png: str,
                     frame_idx: int,
                     window_size: int,
                     hop_seconds: float = 0.25,
                     min_db: float | None = None,
                     max_db: float | None = None,
                     title_prefix: str = "Session Spectrogram"):
    freq = infer_fs(session_data)
    fig, ax_map, axes = setup_figure_with_map()

    # Spectrogram
    ims, total_frames, update_spectrogram_axes = add_spectrogram_axes(
        session_data,
        window_size=window_size,
        hop_seconds=hop_seconds,
        axes=axes,
        min_db=min_db,
        max_db=max_db
    )

    # GPS
    update_route_marker = add_route_axes(session_data, ax_map=ax_map)

    add_colorbar(fig, ims=ims)

    # Time overlay
    time_text = fig.text(0.5, 0.97, "",
                         ha="center",
                         va="center",
                         fontsize=16)

    # Clamp frame index
    frame_idx = int(np.clip(frame_idx, 0, total_frames - 1))

    # ---- render ONE frame ----
    t0, t1 = update_spectrogram_axes(frame_idx)
    update_route_marker(t0, t1)
    time_text.set_text(
        f"{title_prefix}  |  t = {t0:.2f}s → {t1:.2f}s  |  fs≈{freq}Hz"
    )

    # Save PNG
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    import click
    import pathlib
    import os

    @click.command()
    @click.argument("session_file", type=click.Path(exists=True))
    @click.option("-o", "--output", type=click.Path(path_type=pathlib.Path, dir_okay=False, writable=True),
            required=True,
            help="Write the video to this file (extension not needed, .mp4 will be added).")
    @click.option("-d", "--debug", is_flag=True, default=False,
            help="Debug mode, renders only a single frame as a png.")
    def make_video(session_file: str, output: pathlib.Path, debug: bool):
        os.makedirs(output.parent, exist_ok=True)
        output_file = output.with_suffix(".mp4") if not debug else output.with_suffix(".png")

        file_name = os.path.basename(session_file)
        title = os.path.splitext(file_name)[0].removeprefix("session_")
        session_data = pd.read_csv(session_file, parse_dates=["timestamp"])
        if not debug:
            generate_session_video(session_data, window_size=20, out_mp4=str(output_file), title_prefix=title)
        else:
            save_debug_frame(session_data, out_png=str(output_file), window_size=20, frame_idx=1, title_prefix=title)

    make_video()