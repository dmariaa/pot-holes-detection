import os
from datetime import datetime

import humanfriendly
import pandas as pd
import yaml

session_folder = R"data/Data_20251226"
session_key = "20251226_124317"
split_folder = os.path.join(session_folder, "split")
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

    split_session = timestamps.loc[start].strftime("%Y%m%d_%H%M%S")
    a_file = os.path.join(split_folder, f"accel_{split_session}.csv")
    g_file = os.path.join(split_folder, f"gyro_{split_session}.csv")
    l_file = os.path.join(split_folder, f"labels_{split_session}.csv")

    a.to_csv(a_file, index=False)
    g.to_csv(g_file, index=False)
    l.to_csv(l_file, index=False)

    m_file = os.path.join(split_folder, f"metadata__{split_session}.yaml")
    metadata["session_start_time"] = timestamps.loc[start].replace(microsecond=0).to_pydatetime()

    with open(m_file, "w") as f:
        yaml.safe_dump(metadata, f)

    print(f"Generated session {split_session} with {len(a_file)} frames. Total time {humanfriendly.format_timespan(timestamps.iloc[-1] - timestamps.iloc[0])}")
