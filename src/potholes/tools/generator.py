from typing import Iterator, Tuple, List

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


def generate_window_sample(
    window_data: pd.DataFrame,
    *,
    start_idx: int,
    fs: int,
    window_size: int,
    step: int,
    spectrogram_params: dict
) -> Tuple[np.ndarray, dict]:
    labels_data = window_data.loc[window_data['label'].notna()]

    meta = {
        "start_timestamp": window_data["timestamp"].iloc[0].isoformat(),
        "end_timestamp": window_data["timestamp"].iloc[-1].isoformat(),
        "start_idx": int(start_idx),
        "end_idx": int(start_idx + len(window_data)),
        "labels": labels_data.to_dict(orient="records"),
        "window_size": window_size,
        "step": step,
        "params": spectrogram_params,
    }

    sample = generate_sample(window_data,
                             freq=fs,
                             nperseg=spectrogram_params["nperseg"],
                             noverlap=spectrogram_params["noverlap"],
                             nfft=spectrogram_params["nfft"])

    return sample, meta


def generate_samples(data: pd.DataFrame, window_size: int, step: int) \
        -> Tuple[int, Iterator[Tuple[np.ndarray,dict]], dict]:

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

    def _gen() -> Iterator[Tuple[np.ndarray, dict]]:
        for start_idx in starts:
            end_idx = int(start_idx + rows_per_window)
            window_data = data.iloc[start_idx:end_idx]

            sample, meta = generate_window_sample(window_data,
                                                  start_idx=start_idx,
                                                  fs=rows_per_sec,
                                                  window_size=window_size,
                                                  step=step,
                                                  spectrogram_params=spectrogram_params)

            yield sample, meta

    return len(starts), _gen(), spectrogram_params