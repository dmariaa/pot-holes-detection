import os

import tqdm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib.ticker as mticker
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Rectangle

from potholes.old.generator import generate_spectrogram, get_anomaly_intervals
from potholes.old.loader import sensor_cols, load_processed_data
from potholes.tools import transform_to_db

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

def flipper(x: np.array, flip: bool):
    if flip:
        return np.flip(x, axis=0)
    else:
        return x



def plot_axis(signal: np.ndarray, freqs: np.ndarray, t_spec: np.ndarray,
              axis_name: str, min_db: float, max_db: float, flip: bool = True, yaxis_values: np.ndarray = None):
    signal = signal.T

    signal_db = transform_to_db(signal)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10), tight_layout={
        'rect': [0, 0, 1, 0.99]
    })

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

    x_axis_labels = 20
    ax.xaxis.set_major_locator(mticker.MultipleLocator((freqs.max() - freqs.min()) / x_axis_labels))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
    ax.tick_params(axis='x', labelsize=12)

    cb = fig.colorbar(pm, ax=ax, fraction=0.05, pad=0.02)

    cb.set_label(f'dB - {axis_name}', fontsize=14)
    ax.set_xlabel('Frequency [Hz]', fontsize=14)
    ax.set_ylabel('Time [sec]', fontsize=14)

    return fig, ax


def plot_anomaly(ax: plt.axis, interval: dict):
    x0 = 0.0
    x1 = -0.02
    from matplotlib.transforms import blended_transform_factory

    segment_data = interval['data']
    segment_start = interval['start']
    anomaly_time = (segment_data.loc[segment_start]['rel_timestamp'] -
                    segment_data.loc[segment_data.index[0]]['rel_timestamp'])

    ax.hlines(
        y=anomaly_time,
        xmin=x0, xmax=x1,
        transform = blended_transform_factory(ax.transAxes, ax.transData),
        color='red',
        linewidth=3,
        clip_on=False,
    )


def plot_interval(ax: plt.axis, interval: dict, block_time: int):
    segment_data = interval['data']
    segment_start = interval['start']
    segment_end = interval['end']

    interval_rel_timestamp = segment_data.loc[segment_start]['rel_timestamp'] - segment_data.loc[segment_data.index[0]]['rel_timestamp']
    half_block = block_time / 2.0
    interval_start = interval_rel_timestamp - half_block
    interval_end = interval_rel_timestamp + half_block

    rect = Rectangle(
        (0, interval_start),
        ax.get_xlim()[1],
        interval_end - interval_start,
        fill=False,
        linewidth=2,
        edgecolor='black'
    )

    ax.add_patch(rect)
    build_legend(interval, ax)


def build_legend(anomaly_data: dict, ax: plt.axis):
    label = anomaly_data['label']
    data_start = anomaly_data['start']
    data_end = anomaly_data['end']
    segment_data = anomaly_data['data']

    vis_start = segment_data.iloc[0]['rel_timestamp']
    start = segment_data.loc[data_start]['rel_timestamp']
    end = segment_data.loc[data_end]['rel_timestamp']

    legend = (
        f"Label: {label}\n",
        f"Start: {start:.4f}s\n",
        f"End: {end:.4f}s\n",
        f"Visualization start: {vis_start:.4f}s\n",
    )

    # create an AnchoredText box in the upper right
    at = AnchoredText(
        "".join(legend),
        loc='upper right',
        prop=dict(size=10),
        pad=0.1,
        frameon=True,
        bbox_to_anchor=(0.99, 0.99),
        bbox_transform=ax.transAxes
    )
    # style the box: rounded corners, white fill, black edge
    at.patch.set_boxstyle("round,pad=0.5")
    at.patch.set_facecolor("white")
    at.patch.set_edgecolor("black")
    at.patch.set_linewidth(1)
    ax.add_artist(at)


def plot_anomalies(data: pd.DataFrame,
                   max_db: float,
                   min_db: float,
                   window_time: int = 20,
                   output_folder_base: str = "output/plots"):
    intervals = get_anomaly_intervals(data, window_time=window_time)

    with tqdm.tqdm(total=len(sensor_cols) * len(intervals)) as pbar:
        for selected_interval in intervals:
            interval_data = selected_interval['data']
            interval_start_time = interval_data['rel_timestamp'].iloc[0]
            interval_end_time = interval_data['rel_timestamp'].iloc[-1]
            anomaly_start = interval_data.loc[selected_interval['start'], ['rel_timestamp']].values[0]
            anomaly_end = interval_data.loc[selected_interval['end'], ['rel_timestamp']].values[0]

            interval_name = f"{selected_interval['label']}-{anomaly_start*1000:.0f}"

            interval_folder = os.path.join(output_folder_base, f'{window_time}-secs/{interval_name}')
            os.makedirs(interval_folder, exist_ok=True)
            for axis in sensor_cols:
                segment_data = selected_interval['data']
                f, t_spec, Sxx = generate_spectrogram(segment_data, magnitude=axis)

                fig, ax = plot_axis(Sxx, f, t_spec, axis, max_db=max_db, min_db=min_db)
                plot_interval(ax, selected_interval, 2)
                plot_anomaly(ax, selected_interval)

                # fig.show()
                fig.savefig(os.path.join(interval_folder, f"{axis}.png"))
                plt.close(fig)
                pbar.update(1)


if __name__ == "__main__":
    window_time = 10
    data = load_processed_data("data/merged_data.csv")

    min_db = float('inf')
    max_db = -float('inf')

    for col in sensor_cols:
        f, t_spec, Sxx = generate_spectrogram(data, magnitude=col)
        Sxx = transform_to_db(Sxx)
        min_db = min(min_db, Sxx.min())
        max_db = max(max_db, Sxx.max())

    # Anomalies plotting
    plot_anomalies(data,
                   max_db=max_db,
                   min_db=min_db,
                   window_time=window_time)

    # # Clean blocks plotting
    # clean_blocks = get_clean_blocks(data, window_time=window_time)
    #
    # with tqdm.tqdm(total=len(clean_blocks)) as pbar:
    #     for clean_block in clean_blocks:
    #         clean_block_data = clean_block['data']
    #         with tqdm.tqdm(total=(len(clean_block_data) // window_time) + len(sensor_cols)) as pbar2:
    #             for i in range(len(clean_block_data) // window_time):
    #                 clean_block_start_time = clean_block_data['rel_timestamp'].iloc[0] + (window_time * i)
    #                 clean_block_end_time = clean_block_start_time + window_time
    #                 data = clean_block_data[(clean_block_data['rel_timestamp'] >= clean_block_start_time) &
    #                                         (clean_block_data['rel_timestamp'] <= clean_block_end_time)]
    #
    #                 output_folder_base = os.path.join(f'output/plots/{window_time}-secs/empty-{clean_block_start_time*1000:.0f}')
    #                 os.makedirs(output_folder_base, exist_ok=True)
    #                 for axis in sensor_cols:
    #                     f, t_spec, Sxx = generate_spectrogram(data, magnitude=axis)
    #                     fig, ax = plot_axis(Sxx, f, t_spec, axis, max_db=max_db, min_db=min_db)
    #
    #                     # fig.show()
    #                     fig.savefig(os.path.join(output_folder_base, f"{axis}.png"))
    #                     plt.close(fig)
    #
    #                     pbar2.update(1)
    #
    #         pbar.update(1)