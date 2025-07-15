import numpy as np
import pandas as pd
from scipy.signal import spectrogram

from src.loader import load_sensor_data, sensor_cols, load_processed_data


def get_clean_blocks(data: pd.DataFrame, window_time: int = 30):
    is_normal = data['Label'] == 'Normal'
    changes = is_normal.astype(int).diff().fillna(0)

    data['rel_timestamp'] = data['timestamp'] - data['timestamp'].iloc[0]
    starts = data.loc[changes == 1, 'rel_timestamp'].index
    ends = data.loc[changes == -1, 'rel_timestamp'].index

    if is_normal.iloc[0]:
        starts = np.r_[data.index[0], starts]
    if is_normal.iloc[-1]:
        ends = np.r_[ends, data.index[-1]]

    t_max = data['rel_timestamp'].max()
    half_window_time = window_time / 2.0
    blocks = []
    for start_idx, end_idx in zip(starts, ends):
        block_start = data.loc[start_idx, 'rel_timestamp']
        block_end = data.loc[end_idx, 'rel_timestamp']
        ex_start = block_start + half_window_time
        ex_end = block_end - half_window_time
        selected_data = data[(data['rel_timestamp'] >= ex_start) & (data['rel_timestamp'] <= ex_end)]

        if selected_data.empty:
            continue

        if selected_data['rel_timestamp'].iloc[-1] - selected_data['rel_timestamp'].iloc[0] < window_time:
            continue

        blocks.append({
            'data': selected_data
        })

    return blocks


def get_anomaly_intervals(data: pd.DataFrame, window_time: int = 30):
    is_anom = data['Label'] != 'Normal'

    changes = is_anom.astype(int).diff().fillna(0)

    data['rel_timestamp'] = data['timestamp'] - data['timestamp'].iloc[0]
    starts = data.loc[changes == 1, 'rel_timestamp'].index
    ends = data.loc[changes == -1, 'rel_timestamp'].index

    if is_anom.iloc[0]:
        starts = np.r_[data.index[0], starts]
    if is_anom.iloc[-1]:
        ends = np.r_[ends, data.index[-1]]

    t_max = data['rel_timestamp'].max()
    half_window_time = window_time / 2.0
    segments = []

    for start_idx, end_idx in zip(starts, ends):
        label = data.loc[start_idx, 'Label']
        anomaly_start = data.loc[start_idx, 'rel_timestamp']
        anomaly_end = data.loc[end_idx, 'rel_timestamp']
        center = (anomaly_start + anomaly_end) / 2.0
        ex_start = max(0, center - half_window_time)
        ex_end = min(t_max, center + half_window_time)
        seg_df = data[(data['rel_timestamp'] >= ex_start) &
                      (data['rel_timestamp'] <= ex_end)].copy()

        segments.append({
            'label': label,
            'start': start_idx,
            'end': end_idx,
            'data': seg_df
        })

    return segments


def resample(data: pd.DataFrame, sample_rate: int):
    return data.resample(f"{sample_rate}ms").mean().interpolate(
        method="linear", limit_direction="both"
    )


def generate_spectrogram(data: pd.DataFrame, magnitude: str):
    fs = np.round(float(1.0 / data['delta_time'].iloc[1]))

    f, t_spec, Sxx = spectrogram(data[magnitude].values,
                                 fs=fs,
                                 window='hann',
                                 nperseg=50,
                                 noverlap=45,
                                 nfft=512
                                 )
    return f, t_spec, Sxx


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    magnitude = sensor_cols[5]
    data = load_processed_data("data/merged_data.csv")
    f, t_spec, Sxx = generate_spectrogram(data, magnitude=magnitude)

    Sxx_dB = 10 * np.log10(Sxx + 1e-20)

    plt.figure(figsize=(6, 8))
    # transpose Sxx so rows=times, cols=freqs
    plt.pcolormesh(f, t_spec, Sxx_dB.T, shading='gouraud')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Time [sec]')
    plt.title(f'Spectrogram of {magnitude} over time')
    plt.colorbar(label='Power spectral density')
    plt.tight_layout()
    plt.show()
