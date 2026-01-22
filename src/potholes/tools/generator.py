from typing import Iterator, Tuple

import numpy as np
import pandas as pd

signal_cols = ['x_accel', 'y_accel', 'z_accel', 'x_gyro', 'y_gyro', 'z_gyro']

from potholes.tools.signal_tools import generate_spectrogram


def infer_fs(data: pd.DataFrame) -> int:
    delta_time = data['elapsed'].iloc[1]  # assumes sampling is uniform
    rows_per_sec = int(round(1e9 / delta_time))
    return rows_per_sec


def generate_sample(window_data: pd.DataFrame, freq: int, nperseg: int, noverlap: int, nfft: int) -> np.ndarray:
    channels = []

    for axis_idx, axis in enumerate(signal_cols):
        f, t_spec, Sxx = generate_spectrogram(window_data, axis,
                                              freq=freq,
                                              nperseg=nperseg,
                                              noverlap=noverlap,
                                              nfft=nfft)
        channels.append(Sxx)

    sample = np.stack(channels, axis=0)
    return sample


def generate_samples(data: pd.DataFrame, window_size: int, step: int) \
        -> Tuple[int, Iterator[Tuple[np.ndarray, str, int]], dict]:

    rows_per_sec = infer_fs(data)
    rows_per_window = window_size * rows_per_sec
    rows_per_step = step * rows_per_sec
    starts = np.arange(0, len(data) - rows_per_window, rows_per_step).astype(int)

    # TODO: for now define this parameters here, check wether it's better to parametrize the function
    spectrogram_params = {
        "freq": rows_per_sec,
        "nperseg": 64,
        "noverlap": 48,
        "nfft": 64
    }

    def _gen() -> Iterator[Tuple[np.ndarray, str, int]]:
        for start_idx in starts:
            end_idx = int(start_idx + rows_per_window)
            window_data = data.iloc[start_idx:end_idx]
            window_data.loc[window_data["label"].isna(), ["label"]] = "normal"
            labels = window_data.loc[window_data['label'] != "normal", "label"].unique().tolist()
            label =  "+".join(sorted(set(labels))) if len(labels) > 0 else "normal"

            sample = generate_sample(window_data,
                                     freq=rows_per_sec,
                                     nperseg=spectrogram_params["nperseg"],
                                     noverlap=spectrogram_params["noverlap"],
                                     nfft=spectrogram_params["nfft"])

            yield sample, label, start_idx

    return len(starts), _gen(), spectrogram_params