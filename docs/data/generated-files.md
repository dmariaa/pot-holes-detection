# Generated Files

The loader produces a session CSV for each processed session using the loader `process` command. When
`--generate-samples` is enabled, it also produces spectrogram sample files in
`.npz` format.

## Session CSV (`session_*.csv`)

The session CSV contains merged, resampled sensor data plus label annotations.
It is the main input for plotting, sampling, and model training.

File naming:
`session_{sensor_name}_{session_date_time}.csv`

Field details:

| Field | Description |
| --- | --- |
| `timestamp` | ISO timestamp for each resampled row. |
| `x_accel` | X axis acceleration. |
| `y_accel` | Y axis acceleration. |
| `z_accel` | Z axis acceleration. |
| `x_gyro` | X axis gyroscope. |
| `y_gyro` | Y axis gyroscope. |
| `z_gyro` | Z axis gyroscope. |
| `latitude` | Latitude in decimal degrees. |
| `longitude` | Longitude in decimal degrees. |
| `sensor_timestamp` | Device monotonic time aligned to this row. |
| `elapsed` | Nanoseconds since the session start. |
| `is_interpolated` | True if the signal values were interpolated during resample. |
| `sensor_row_id` | Row id used when merging labels. |
| `timestamp_label` | Label timestamp (nearest match) or empty. |
| `lat_label` | Label latitude or empty. |
| `lon_label` | Label longitude or empty. |
| `label` | Label name or empty. |

Example records:
```csv
timestamp,x_accel,y_accel,z_accel,x_gyro,y_gyro,z_gyro,latitude,longitude,sensor_timestamp,elapsed,is_interpolated,sensor_row_id,timestamp_label,lat_label,lon_label,label
2025-12-26 11:43:41.280,38.0,-55.0,999.3333333333334,0.39999999999999997,-0.13333333333333333,0.0,,,30699,0,False,0,,,,
2025-12-26 11:43:41.300,36.5,-53.5,995.6666666666667,0.35,-0.11666666666666667,0.0,40.4076043,-3.6258028,,20000000,True,1,,,,
2025-12-26 11:44:33.260,-23.77777777777778,-134.55555555555554,1148.5555555555554,-1.5666666666666664,-0.7666666666666668,-0.1444444444444444,40.407912,-3.6259785,,51980000000,True,2599,2025-12-26 11:44:33.266,40.407912,-3.6259785,speed_bump
2025-12-26 11:44:39.880,40.5,-81.5,865.5,-1.95,0.9,-0.8500000000000001,40.407912,-3.6259785,38029.0,58600000000,False,2930,2025-12-26 11:44:39.880,40.407912,-3.6259785,speed_bump
```

## Sample NPZ (`data_*.npz`)

Sample files are generated with `--generate-samples` and contain the
spectrogram data for one sliding window. The filename encodes the start index,
window size, step size, and labels.

File naming:
`data_{start_idx}_{window_size}_{step}_{labels}.npz`

Keys:

| Key | Type | Description |
| --- | --- | --- |
| `sample` | numpy array | Shape `(6, n_freqs, n_times)` for axes `x_accel`, `y_accel`, `z_accel`, `x_gyro`, `y_gyro`, `z_gyro`. |
| `meta` | string | JSON string with the window metadata. |

Metadata fields:

| Field | Description | Example |
| --- | --- | --- |
| `sensor_name` | Sensor name from metadata. | `STBPRO3 @71E957` |
| `start_timestamp` | ISO start timestamp for the window. | `2025-12-26T11:43:41.280000` |
| `end_timestamp` | ISO end timestamp for the window. | `2025-12-26T11:44:01.260000` |
| `start_idx` | Start row index in the session CSV. | `0` |
| `end_idx` | End row index in the session CSV. | `1000` |
| `window_size` | Window size in seconds. | `20` |
| `step` | Step size in seconds. | `1` |
| `params` | Spectrogram parameters. | `{"freq": 50, "nperseg": 64, "noverlap": 48, "nfft": 64}` |
| `labels` | Labeled rows within the window (same schema as the session CSV). | `[]` |

Related file:
- `spectrogram_params.yaml` is saved alongside the `.npz` files with the same
  parameters used during generation.

## Loading an NPZ sample

```python
import json
import numpy as np

path = "data/Data_20251226/session_STBPRO3@71E957_20251226_114336/data_0_20_1_normal.npz"
with np.load(path, allow_pickle=False) as data:
    sample = data["sample"]
    meta = json.loads(data["meta"].item())

print(sample.shape)
print(meta["start_timestamp"], meta["end_timestamp"])
```
