import os

import click
import humanfriendly
import pandas as pd
import yaml


def split_session(session_folder: str, session_key: str, dry_run: bool = False):
    split_folder = os.path.join(session_folder, "split")
    if dry_run:
        print(f"Dry run: split files would be written to {split_folder}")
    else:
        os.makedirs(split_folder, exist_ok=True)

    session_accelerator_file = os.path.join(session_folder, f"accel_{session_key}.csv")
    session_gyroscope_file = os.path.join(session_folder, f"gyro_{session_key}.csv")
    session_label_file = os.path.join(session_folder, f"labels_{session_key}.csv")
    session_metadata_file = os.path.join(session_folder, f"metadata__{session_key}.yaml")

    data_accel = pd.read_csv(session_accelerator_file, low_memory=False, skipinitialspace=True)
    gyro_accel = pd.read_csv(session_gyroscope_file, low_memory=False, skipinitialspace=True)
    label_data = pd.read_csv(session_label_file, low_memory=False, skipinitialspace=True)
    metadata = yaml.safe_load(open(session_metadata_file))

    d = data_accel["sensor_timestamp"].diff()
    gaps_mask = (d > 3)

    starts = [0] + data_accel[gaps_mask].index.to_list()
    ends = [x-1 for x in starts[1:]] + [data_accel.index.to_list()[-1]]

    for start, end in zip(starts, ends):
        a = data_accel.loc[start:end]
        g = gyro_accel.loc[start:end]

        timestamps = pd.to_datetime(a["timestamp"], unit="ms")
        label_timestamps = label_data[["timestamp"]].copy()
        label_timestamps["timestamp"] = pd.to_datetime(label_timestamps["timestamp"], unit="ms")

        l = label_data[(label_timestamps["timestamp"] >= timestamps.iloc[0]) & (label_timestamps["timestamp"] <= timestamps.iloc[-1])]

        split_session_key = timestamps.loc[start].strftime("%Y%m%d_%H%M%S")
        a_file = os.path.join(split_folder, f"accel_{split_session_key}.csv")
        g_file = os.path.join(split_folder, f"gyro_{split_session_key}.csv")
        l_file = os.path.join(split_folder, f"labels_{split_session_key}.csv")

        m_file = os.path.join(split_folder, f"metadata__{split_session_key}.yaml")
        session_metadata = metadata.copy()
        session_metadata["session_start_time"] = timestamps.loc[start].replace(microsecond=0).to_pydatetime()

        if dry_run:
            print(f"Would write {a_file}")
            print(f"Would write {g_file}")
            print(f"Would write {l_file}")
            print(f"Would write {m_file}")
        else:
            a.to_csv(a_file, index=False)
            g.to_csv(g_file, index=False)
            l.to_csv(l_file, index=False)

            with open(m_file, "w") as f:
                yaml.safe_dump(session_metadata, f)

        print(f"Generated session {split_session_key} with {len(a)} frames. Total time {humanfriendly.format_timespan(timestamps.iloc[-1] - timestamps.iloc[0])}")


if __name__ == "__main__":
    @click.command()
    @click.argument(
        "session_folder",
        type=click.Path(exists=True, file_okay=False),
        metavar="SESSION_FOLDER",
    )
    @click.argument("session_key", metavar="SESSION_KEY")
    @click.option("--dry-run", is_flag=True, default=False, help="Show what would be written without writing files.")
    def cli(session_folder: str, session_key: str, dry_run: bool):
        """
        Split a raw session into smaller raw sessions.

        SESSION_FOLDER is the folder containing accel, gyro, labels, and metadata files.
        SESSION_KEY is the timestamp key used in the session filenames, for example 20260210_122003.
        """
        split_session(session_folder=session_folder, session_key=session_key, dry_run=dry_run)

    cli()
