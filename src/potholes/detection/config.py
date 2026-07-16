import os
from datetime import datetime


def _is_blank(value) -> bool:
    return value is None or value == ""


def _default_run_name() -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"training-{timestamp}"


def _derive_run_name(config: dict) -> str:
    configured_run_name = config.get("run_name")
    if not _is_blank(configured_run_name):
        return str(configured_run_name)

    return _default_run_name()


def normalize_run_config(config: dict) -> dict:
    config = dict(config or {})
    wandb_config = dict(config.get("wandb") or {})

    run_name = _derive_run_name(config)
    config["run_name"] = run_name

    if "wandb" in config:
        wandb_config["name"] = run_name
        config["wandb"] = wandb_config

    return config


def prepare_config(base_config: dict) -> dict:
    config = dict(base_config or {})
    data_config = dict(config.get("data") or {})
    config["data"] = data_config

    train_defaults = {
        "batch_size": 8,
        "epochs": 100,
        "learning_rate": 0.005,
        "patience": 10,
    }
    data_defaults = {
        "version": 2,
        "data_folder": "dataset",
        "generate": False,
        "step": 1,
        "verbose": True,
        "window_size": 10,
    }

    for key, value in train_defaults.items():
        if config.get(key) is None:
            config[key] = value

    for key, value in data_defaults.items():
        if data_config.get(key) is None:
            data_config[key] = value

    return normalize_run_config(config)


def run_artifact_name(run_name: str, suffix: str) -> str:
    return f"{run_name}-{suffix}"


def run_output_path(base_folder: str, run_name: str) -> str:
    return os.path.join(base_folder, run_name)
