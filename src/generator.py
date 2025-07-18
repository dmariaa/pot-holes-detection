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


def generate_samples(data: pd.DataFrame, sample_name: str, window_size: int = 20, step: int = 1):
    delta_time = data['delta_time'].iloc[1]    # assumes sampling is uniform
    rows_per_sec = 1 / delta_time
    rows_per_window = window_size * rows_per_sec
    rows_per_step = step * rows_per_sec

    starts = np.arange(0, len(data) - rows_per_window, rows_per_step).astype(int)
    pass

    import tqdm
    with tqdm.tqdm(total=len(starts)) as pbar:
        for start_idx in starts:
            end_idx = int(start_idx + rows_per_window)
            window_data = data.iloc[start_idx:end_idx]

            labs = window_data['Label'].unique().tolist()
            anoms = [lab.lower().replace(' ', '_')
                     for lab in labs if lab != 'Normal']
            if anoms:
                label = '+'.join(sorted(set(anoms)))
            else:
                label = 'normal'

            channels = []
            with tqdm.tqdm(total=len(sensor_cols)) as pbar2:
                for axis_idx, axis in enumerate(sensor_cols):
                    f, t_spec, Sxx = generate_spectrogram(window_data, axis)
                    channels.append(Sxx)
                    pbar2.update(1)

            sample = np.stack(channels, axis=0)
            np.save(f"data/samples/{sample_name}_{start_idx}_{window_size}_{step}_{label}.npy", sample)
            pbar.update(1)

    data.to_csv(f"data/samples/{sample_name}.csv")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    data = load_processed_data("data/merged_data.csv")
    generate_samples(data, "stb1", window_size=10, step=1)


