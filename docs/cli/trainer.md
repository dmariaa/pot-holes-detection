# Trainer CLI

The trainer CLI lives in `potholes.detection.trainer`. Run commands from the repository root.

```powershell
python -m potholes.detection.trainer --help
```

## Commands

### Train a model

Trains a transformer model from a config file or from built-in defaults.

Usage:
```shell-session
$ uv run -m potholes.detection.trainer train --help

Usage: python -m potholes.detection.trainer train [OPTIONS] [CONFIG_FILE]

Options:
  --use-defaults                  Allow running without a config file using
                                  built-in defaults.
  --training-log-folder DIRECTORY
                                  [default: (auto)]
  --batch-size INTEGER            [default: 8]
  --epochs INTEGER                [default: 100]
  --learning-rate FLOAT           [default: 0.005]
  --patience INTEGER              [default: 10]
  --data-folder DIRECTORY         [default: dataset]
  --data-version INTEGER RANGE    [default: 2; 1<=x<=2]
  --generate / --no-generate      [default: no-generate]
  --window-size INTEGER           [default: 10]
  --step INTEGER                  [default: 1]
  --verbose / --no-verbose        [default: verbose]
  --help                          Show this message and exit.
```

Example with a config file:
```shell-session
$ uv run -m potholes.detection.trainer train configs/train.yaml
```

Example with defaults:
```shell-session
$ uv run -m potholes.detection.trainer train --use-defaults --epochs 20 --batch-size 16
```

### Test a model

Evaluates a trained model from a training output folder.

Usage:
```shell-session
$ uv run -m potholes.detection.trainer test --help

Usage: python -m potholes.detection.trainer test [OPTIONS] MODEL_PATH

Options:
  --help  Show this message and exit.
```

Example:
```shell-session
$ uv run -m potholes.detection.trainer test output/training/training-001
```
