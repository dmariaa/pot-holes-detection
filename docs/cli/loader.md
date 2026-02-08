# Loader CLI

The loader CLI lives in `potholes.loader`. Run commands from the repository root.

```powershell
python -m potholes.loader --help
```

## Commands 

### List sessions

Lists information about all the data collecting sessions stored in a folder and subfolders, 

Usage:
```shell-session
$ uv run -m potholes.loader list --help

Usage: python -m potholes.loader list [OPTIONS] FOLDER

  Lists all sessions found in a given folder

Options:
  --help  Show this message and exit.
```


Example:
```shell-session
$ uv run -m potholes.loader list data/Data_YYYYMMDD
                                                             Speed
#   Date       Time     Sensor               Frames    PotHole   Bump ManHole   Other Total time
------------------------------------------------------------------------------------------------------------------------
1   2026-01-08 16:18:34 PhoneA               60000          12      3       1       0 00:20:00
2   2026-01-08 17:05:12 PhoneB               45000           2      0       0       1 00:15:00
```

### Session statistics

Displays a detailed statistics report for a single session.

Usage:
```shell-session
$ uv run -m potholes.loader stats --help

Usage: python -m potholes.loader stats [OPTIONS] FOLDER SESSION_NUMBER

  Get several statistics for a session

Options:
  --help  Show this message and exit.
```

Example:
```shell-session
$ uv run -m potholes.loader stats data/Data_20251226 1
Session: STBPRO3 @71E957 (20251226_114336)
Start:   2025-12-26 11:43:36
Path:    data\Data_20251226

Frames:  17887
Duration: 5 minutes and 57.4 seconds
Overall rate: 50.05 Hz

Frame rate:
  min     5.49 Hz
  mean    158.44 Hz
  median  142.86 Hz
  max     1000.00 Hz
  std     160.34 Hz
  overall 50.05 Hz

Frame interval:
  min     0.0010 s
  mean    0.0200 s
  median  0.0070 s
  max     0.1820 s
  std     0.0186 s

GPS missing: 1.45% (260 rows)
Duplicate timestamps: 0

Labels:
  pothole      0
  speed_bump   2
  manhole      0
  other        1
```

### Process sessions

This command process a session and generates several outputs:

- a **`session_<sensor-name>_<session-datetime>.csv`** with all the session data merged into it and resampled to a new 
  sample rate.
- an html file showing the map of the route and all the labelled points in it.
- [optional] if _generate-samples_ paremeter is used, a set of **`data_<counter>_<window-size>_<step-size>_<labels>.npz`** 
  files containing the spectrogram of each window in the session, according to *`window-size`* and *`step`* parameters
- [optional] if samples are generated, a metadata file containing window and spectrogram parameters

Usage:
```shell-session
$ uv run -m potholes.loader --help

Usage: python -m potholes.loader process [OPTIONS] FOLDER SESSION_LIST

  Preprocess a session and generate a data file with all the session data.

  FOLDER Path to a folder containing raw sensor session files.

  SESSION_LIST Which sessions to process.

  Accepted formats:
    all            Process all detected sessions
    3              Process session 3 only
    1,2,5          Process multiple sessions
    2-6            Process a range of sessions
    1,3-5,8        Mixed list and ranges

Options:
  -o, --output-folder PATH  Output folder for the session file. Defaults to session folder.
  -sr, --sample-rate INTEGER  Resample rate in Hz (default 50)
  -g, --generate-samples  Generate sample files for the selected sessions
  -ws, --window-size INTEGER  Window size in seconds for the sample files (default 20)
  --step INTEGER  Seconds between consecutive windows for the sample files (default 1)
  -v, --verbose  Prints load and parse debug information
  --help  Show this message and exit.
```

Example:
```shell-session
$ uv run -m potholes.loader process data/Data_YYYYMMDD 1 -o output

Loading PhoneA              2026-01-08 16:18:34
Total frames: 60000
Session saved to output/session_PhoneA_20260108_161834/session_PhoneA_20260108_161834.csv
Session route map saved to output/session_PhoneA_20260108_161834/route_map.html
```

Example with samples:
```shell-session
$ uv run -m potholes.loader process data/Data_YYYYMMDD 1 -o output -g -ws 20 --step 1

Loading PhoneA              2026-01-08 16:18:34
Total frames: 60000
Session saved to output/session_PhoneA_20260108_161834/session_PhoneA_20260108_161834.csv
Session route map saved to output/session_PhoneA_20260108_161834/route_map.html
Generating data samples for session
Generated data_0_20_1_normal.npz:  10%|###                       | 30/300 [00:04<00:36,  7.50it/s]
```

### Delete a session

This command removes the raw `accel`, `gyro`, `labels`, and `metadata` files for the
selected session.

Help:
```shell-session
$ uv run -m potholes.loader delete --help

Usage: python -m potholes.loader delete [OPTIONS] FOLDER SESSION_NUMBER

  Deletes a session

Options:
  --help  Show this message and exit.
```

Example:
```shell-session
$ uv run -m potholes.loader delete data/Data_YYYYMMDD 1
                                                             Speed
    Date       Time     Sensor               Frames    PotHole   Bump ManHole   Other Total time
------------------------------------------------------------------------------------------------------------------------
1   2026-01-08 16:18:34 PhoneA               60000          12      3       1       0 00:20:00

Are you sure you want to delete this session? [y/N]: y

Deleted session 20260108_161834 in folder data\Data_YYYYMMDD
```
