from typing import Iterator, Tuple

import numpy as np
import pandas as pd
from scipy.signal import spectrogram


def generate_spectrogram(data: pd.DataFrame, magnitude: str, freq: int, overlap: int = 45, nfft: int = 512):
    f, t_spec, Sxx = spectrogram(data[magnitude].values,
                                 fs=freq,
                                 window='hann',
                                 nperseg=freq,
                                 noverlap=overlap,
                                 nfft=nfft
                                 )
    return f, t_spec, Sxx


def generate_samples(data: pd.DataFrame, window_size: int, step: int) \
        -> Tuple[int, Iterator[Tuple[np.ndarray, str, int]]]:
    signal_cols = ['x_accel', 'y_accel', 'z_accel', 'x_gyro', 'y_gyro', 'z_gyro']

    delta_time = data['elapsed'].iloc[1]    # assumes sampling is uniform
    rows_per_sec = int(round(1e9 / delta_time))
    rows_per_window = window_size * rows_per_sec
    rows_per_step = step * rows_per_sec

    starts = np.arange(0, len(data) - rows_per_window, rows_per_step).astype(int)

    def _gen() -> Iterator[Tuple[np.ndarray, str, int]]:
        for start_idx in starts:
            end_idx = int(start_idx + rows_per_window)
            window_data = data.iloc[start_idx:end_idx]
            window_data.loc[window_data["label"].isna(), ["label"]] = "normal"
            labels = window_data.loc[window_data['label'] != "normal", "label"].unique().tolist()
            label =  "+".join(sorted(set(labels))) if len(labels) > 0 else "normal"

            channels = []
            for axis_idx, axis in enumerate(signal_cols):
                f, t_spec, Sxx = generate_spectrogram(window_data, axis, freq=rows_per_sec)
                channels.append(Sxx)

            sample = np.stack(channels, axis=0)
            yield sample, label, start_idx

    return len(starts), _gen()