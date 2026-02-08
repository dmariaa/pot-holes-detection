# Getting Started

## Requirements
- Python 3.12 or newer
- uv 0.9.6 or newer, or the python environment manager of your choice

All the command examples in this documentation use the `uv` commands, just adapt them to the environment 
of your choice.

## Setup
Clone this repository:

```shell-session
$ git clone https://github.com/dmariaa/pot-holes-detection
$ cd pot-holes-detection
```

From the repository root, run:

```shell-session
$ uv sync
```

## Quick check
Confirm the CLI is available:

```shell-session
$ uv run -m potholes.loader --help
```

## Typical flow
1) Place raw session data under `data/` (or any folder you prefer).
2) List sessions:

```shell-session
$ uv run -m potholes.loader list data/Data_YYYYMMDD
```

3) Process one session into `output/`:

```shell-session
$ uv run -m potholes.loader process data/Data_YYYYMMDD 1 -o output
```

4) Optionally generate spectrogram files:

```shell-session
$ uv run -m potholes.loader process data/Data_YYYYMMDD 1 -o output -g -ws 20 --step 1
```
