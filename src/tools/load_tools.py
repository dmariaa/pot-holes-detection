import numpy as np
import pandas as pd
from pandas import DataFrame


class DataLoadException(Exception):
    pass

def load_labels_file(filepath: str):
    labels = pd.read_csv(
        filepath,
        low_memory=False,
        skipinitialspace=True,
    )

    labels["timestamp"] = pd.to_datetime(labels["timestamp"], unit="ms")
    labels = labels.sort_values(by=["timestamp"])

    return labels

def load_data_file(filepath: str, silent: bool = False) -> pd.DataFrame:
    data = pd.read_csv(
        filepath,
        low_memory=False,
        skipinitialspace=True,
    )

    data = data.sort_values(by="sensor_timestamp")

    if not silent:
        print(f"Loading {filepath}")
        print(f"The file contains {data['sensor_type'].iloc[0]} data, version { 2 if 'elapsed' in data.columns else 1 }.")
        print(f"Total number of rows: {len(data)}")
        print(f"Duplicated  timestamps: {data['timestamp'].duplicated().sum()}")
        print("")

    # raise error if timestamp is not sorted
    if (data["timestamp"].diff() < 0).any():
        raise DataLoadException("Timestamp is not monotonic")

    if 'elapsed' not in data.columns:
        data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms")

        # Remove duplicated timestamps if needed
        ts = data["timestamp"]
        base_ns = ts.astype("int64")
        within = ts.groupby(ts).cumcount().astype("int64")
        data["timestamp"] = pd.to_datetime(base_ns + within * 100, unit="ns")

        # Generate elapsed column for old files
        t0 = ts.iloc[0]
        data['elapsed'] = (ts - t0).dt.total_seconds().mul(1e9).astype("int64")
    else:
        t0_ns = data["timestamp"].iloc[0]
        e0 = data["elapsed"].iloc[0]

        data["timestamp"] = pd.to_datetime(
            t0_ns + (data["elapsed"] - e0),
            unit="ns"
        )

    data['lat'] = pd.to_numeric(data['lat'], errors='coerce')
    data['lon'] = pd.to_numeric(data['lon'], errors='coerce')

    return data

def merge_data(a_data: pd.DataFrame, g_data: pd.DataFrame) -> pd.DataFrame:
    columns = {
        'timestamp_accel': 'timestamp',
        'lat_accel': 'latitude',
        'lon_accel': 'longitude',
        'elapsed_accel': 'elapsed',
        'sensor_timestamp': 'sensor_timestamp',
        'x_accel': 'x_accel',
        'y_accel': 'y_accel',
        'z_accel': 'z_accel',
        'x_gyro': 'x_gyro',
        'y_gyro': 'y_gyro',
        'z_gyro': 'z_gyro',
        'raw_data_accel': 'raw_data_accel'
    }

    data = a_data.merge(
        g_data,
        on="sensor_timestamp",
        suffixes=("_accel", "_gyro"),
        how="outer",
        indicator=True
    )

    data = data[columns.keys()]
    data = data.rename(columns=columns)
    return data[~data['latitude'].isna()]


def merge_labels(labels: pd.DataFrame, data: pd.DataFrame) -> tuple[DataFrame, DataFrame]:
    lb = labels.sort_values(by=["timestamp"]).copy()
    lb = lb.rename(columns={
        "timestamp": "timestamp_label",
        "lat": "lat_label",
        "lon": "lon_label"
    })

    md = data.reset_index(drop=True).copy()
    md["sensor_row_id"] = md.index
    md = md.sort_values(by=["timestamp"])

    label_to_sensor = pd.merge_asof(
        lb,
        md[["timestamp", "sensor_row_id"]],
        left_on="timestamp_label",
        right_on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("1s")
    )

    matched = label_to_sensor.dropna(subset=['sensor_row_id']).copy()
    matched['sensor_row_id'] = matched['sensor_row_id'].astype(int)

    labels_cols = ["timestamp_label", "lat_label", "lon_label", "label"]
    merged = md.copy()
    merged["timestamp_label"] = pd.NaT
    merged["lat_label"] = np.nan
    merged["lon_label"] = np.nan
    merged["label"] = pd.Series(pd.NA, index=merged.index, dtype="string")

    merged.loc[matched["sensor_row_id"], labels_cols] = matched[labels_cols].values

    return merged, labels


def resample_data(data: pd.DataFrame, fs: int) -> pd.DataFrame:
    df = data.copy().sort_values(by=["timestamp"]).set_index("timestamp")

    signal_cols = ['x_accel', 'y_accel', 'z_accel', 'x_gyro', 'y_gyro', 'z_gyro']
    gps_cols = ['latitude', 'longitude']

    period_ns = int(round(1e9 / fs))
    sample_rate = f"{period_ns}ns"

    df_u = df[signal_cols].resample(sample_rate).mean()
    is_interpolated = df_u.isna().all(axis=1)

    df_u[signal_cols] = df_u[signal_cols].interpolate(limit=3)
    df_u[gps_cols] = df[gps_cols].resample(sample_rate).ffill()
    df_u["sensor_timestamp"] = df["sensor_timestamp"].resample(sample_rate).last().astype("Int64")

    # Reconstruct elapsed field, redundant, but we do it
    t0 = df_u.index[0]
    df_u["elapsed"] = (df_u.index - t0).view("int64")

    df_u["is_interpolated"] = is_interpolated

    return df_u.reset_index()

