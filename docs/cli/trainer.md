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

### Weights & Biases logging

The trainer can log metrics and artifacts to Weights & Biases when a `wandb`
section is present in the training config.

Login from the project environment:
```shell-session
$ uv run wandb login
```

Example config:
```yaml
wandb:
  enabled: true
  project: potholes-detection
  entity:
  mode: online
  name: london-session-split-baseline
  tags:
    - baseline
    - ast
    - session-split
  log_split_artifact: true
  log_training_artifact: true
  log_model_artifact: true
  log_dataset_artifact: false
  log_batch_metrics: true
  log_batch_interval: 25
```

Logged metrics:
- `train/batch_loss`
- `train/batch_acc`
- `train/running_loss`
- `train/running_acc`
- `train/loss`
- `train/acc`
- `val/loss`
- `val/acc`
- `val/auc`
- `lr`
- `best/val_loss`
- `best/val_auc`

Batch metrics are logged every `log_batch_interval` optimizer steps when
`log_batch_metrics` is enabled. Validation metrics are logged once per epoch.

Logged artifacts:
- split artifact: `config.yaml`, `data_split.npz`, `split_summary.yaml`
- training artifact: `config.yaml`, `data_split.npz`, `split_summary.yaml`, `train_log.npz`
- model artifact: `best_model.pth`, `config.yaml`, `data_split.npz`, `split_summary.yaml`

Full dataset artifact logging is disabled by default because generated `.npz`
datasets can be large. Set `log_dataset_artifact: true` only when you want W&B
to upload the full configured `data.data_folder`.

### Generate a split file

Generates `train_idx`, `val_idx`, and `test_idx` arrays from a training config.
This supports both the default stratified split and explicit session splits.

Usage:
```shell-session
$ uv run -m potholes.detection.trainer split --help

Usage: python -m potholes.detection.trainer split [OPTIONS] CONFIG_FILE

Options:
  --output-file FILE  Where to save train_idx, val_idx, and test_idx.
  --help              Show this message and exit.
```

Example:
```shell-session
$ uv run python -m potholes.detection.trainer split configs/train_london_session_split.yaml --output-file output/splits/london_session_split.npz
```

Session splits can explicitly list all three splits:
```yaml
split:
  strategy: session
  shuffle: false
  train_sessions:
    - session_STBPRO1@824C26_20260210_122839
    - session_STBPRO1@824C26_20260210_141443
  val_sessions:
    - session_STBPRO1@824C26_20260211_094115
  test_sessions:
    - session_STBPRO3@71E957_20260211_094114
```

If `train_sessions` is omitted, every session not listed in `val_sessions` or
`test_sessions` is used for training. If `train_sessions` is present, sessions
not listed in any split are ignored.

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
