from datetime import datetime

import numpy as np
import pandas as pd

sensor_cols = ["gyroX[mdps]", "gyroY[mdps]", "gyroZ[mdps]",
               "accX[mg]", "accY[mg]", "accZ[mg]"]


def load_processed_data(filepath: str) -> pd.DataFrame:
    data = pd.read_csv(filepath, index_col=0)
    data.rename(columns={"Timestamp": "timestamp"}, inplace=True)
    data = resample_uniform(data)
    return data


def resample_uniform(data: pd.DataFrame):
    t = data['timestamp'].values
    median_dt = np.median(np.diff(t))
    fs = np.round(float(1.0 / median_dt))
    t_uniform = np.arange(t[0], t[-1], 1 / fs)

    df = pd.DataFrame({'timestamp': t_uniform})

    for col in sensor_cols:
        acc_x_sig = data[col].values
        df[col] = np.interp(t_uniform, t, acc_x_sig)

    df = pd.merge_asof(
        df,
        data[['timestamp', 'Label']],
        on='timestamp',
        direction='nearest'
    )

    df = df[data.columns]
    df['delta_time'] = np.concatenate([[0.0],
                                       np.diff(df['timestamp']).round(2)])

    return df


def load_sensor_data(filepath: str) -> pd.DataFrame:
    """
    Load sensor data from CSV, drop incomplete rows/columns, merge date and time, and convert to proper types.
    """
    data = pd.read_csv(filepath)
    data = data.dropna(axis=1, how="all").dropna().reset_index(drop=True)
    # Merge the date and time columns (assumes columns "dd/mm/yyyy" and "hh:mm:ss.ms")
    data['dateTime'] = data["dd/mm/yyyy"].astype(str).str.strip() + " " + data["hh:mm:ss.ms"].astype(str).str.strip()
    data.drop(columns=["dd/mm/yyyy", "hh:mm:ss.ms"], inplace=True)

    # Convert sensor columns to float
    data[sensor_cols] = data[sensor_cols].astype(np.float64)

    # Convert the merged DateTime to a UNIX timestamp
    data["timestamp"] = data["dateTime"].apply(convert_date_to_timestamp)
    data.drop(columns=["dateTime"], inplace=True)

    # Add delta time
    data["delta_time"] = data["timestamp"].diff().round(2)
    data.loc[0, "delta_time"] = 0.0
    return data


def convert_date_to_timestamp(date_str: str) -> float:
    """
    Convert a date string to a UNIX timestamp.
    Assumes date_str format: "dd/mm/yyyy HH:MM:SS.ms"
    """
    dt = datetime.strptime(date_str.strip(), "%d/%m/%Y %H:%M:%S.%f")
    return round(dt.timestamp(), 3)


if __name__ == "__main__":
    load_sensor_data("data/merged_data.csv")
