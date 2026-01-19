import glob
import os
from pathlib import Path

import click
import pandas as pd
import yaml

from tools.load_tools import load_data_file, merge_data, resample_data, merge_labels, load_labels_file, \
    DataLoadException


def get_session_stats(session: dict):
    accel_data = load_data_file(session["accel_file"], silent=True)
    labels_data = load_labels_file(session["labels_file"])

    d = accel_data.sort_values("timestamp").copy()
    ts = d["timestamp"]
    records = len(d)
    time = (ts.iloc[-1] - ts.iloc[0])

    return {
        'frames': records,
        'time': time,
        'labels': labels_data.groupby('label').size().to_dict()
    }


def find_sessions(data_folder: str):
    metadata_files = glob.glob(os.path.join(data_folder, "**/metadata*.yaml"), recursive=True)
    sessions = []
    for metadata_file in metadata_files:
        metadata: dict = yaml.safe_load(open(metadata_file))

        metadata_key = metadata["session_start_time"].strftime("%Y%m%d_%H%M%S")
        metadata_path = os.path.dirname(metadata_file)
        metadata['session_path'] = metadata_path
        metadata['session_key'] = metadata_key

        session_accel_file = os.path.join(metadata_path, f"accel_{metadata_key}.csv")
        session_gyro_file = os.path.join(metadata_path, f"gyro_{metadata_key}.csv")
        session_labels_file = os.path.join(metadata_path, f"labels_{metadata_key}.csv")

        if not os.path.exists(session_accel_file):
            raise FileNotFoundError(f"{session_accel_file} does not exist")
        if not os.path.exists(session_gyro_file):
            raise FileNotFoundError(f"{session_gyro_file} does not exist")
        if not os.path.exists(session_labels_file):
            raise FileNotFoundError(f"{session_labels_file} does not exist")

        metadata.update({
            'accel_file': session_accel_file,
            'gyro_file': session_gyro_file,
            'labels_file': session_labels_file,
            "metadata_file": metadata_file
        })

        accel_stats = get_session_stats(metadata)

        metadata.update({
            'stats': accel_stats,
        })

        sessions.append(metadata)

    return sessions


def load_session(session: dict, sample_rate: int, verbose: bool = False) -> pd.DataFrame:
    if verbose:
        print(f"Loading session: {session['sensor_name']} {session['session_start_time']}")

    # 1) Load data
    accel_data = load_data_file(session["accel_file"], silent=(not verbose))
    gyro_data = load_data_file(session["gyro_file"], silent=(not verbose))

    # 2) Merge data
    merged_data = merge_data(accel_data, gyro_data)

    # 3) Resample sensor data to uniform rate 50hz
    resampled_data = resample_data(merged_data, fs=sample_rate)

    # 4) Merge sensor data with labels
    labels_data = load_labels_file(session['labels_file'])
    final_data, labels = merge_labels(labels_data, resampled_data)

    return final_data


def delete_session(session: dict):
    files = {
        "accel": session.get("accel_file"),
        "gyro": session.get("gyro_file"),
        "labels": session.get("labels_file"),
        "metadata": session.get("metadata_file")
    }

    for name, file in files.items():
        path = Path(file)
        path.unlink()
        click.echo(f"Deleted file: {file}")


def process_session(session: dict, resample_freq: int, verbose: bool = False):
    data = load_session(session=session, sample_rate=resample_freq, verbose=verbose)

    if output_folder is None:
        o_folder = session["session_path"]
    else:
        o_folder = output_folder

    session_file = os.path.join(o_folder,
                                f"session_{session['sensor_name']}_{session['session_key']}.csv".replace(" ", ""))
    data.to_csv(session_file, index=False)
    data.to_csv(session_file, index=False)
    click.echo(f"Session saved to {session_file}", nl=True)