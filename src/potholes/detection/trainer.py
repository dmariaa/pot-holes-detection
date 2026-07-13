import os
from collections import Counter

import click
import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score, confusion_matrix
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import plotly.graph_objects as go

from potholes.detection.data import get_dataset
from potholes.detection.data.dataset import labels_to_id, stratified_split_indices, session_split_indices, RoadLabel
from potholes.detection.models.transformer import load_model

try:
    import wandb
except ImportError:
    wandb = None


def collate_fn(batch):
    samples, metas = zip(*batch)
    samples = torch.stack(samples)
    labels = [labels_to_id(m['labels']).value for m in metas]
    return samples, torch.tensor(labels, dtype=torch.long)


def build_split_indices(dataset, config: dict) -> tuple[list[int], list[int], list[int]]:
    split_config = config.get("split") or {}
    split_strategy = split_config.get("strategy", "stratified")
    if split_strategy == "session":
        return session_split_indices(
            dataset,
            train_sessions=split_config.get("train_sessions"),
            val_sessions=split_config.get("val_sessions", []),
            test_sessions=split_config.get("test_sessions", []),
            shuffle=split_config.get("shuffle", False),
            seed=split_config.get("seed"),
        )
    if split_strategy == "stratified":
        return stratified_split_indices(
            dataset,
            val_ratio=split_config.get("val_ratio", 0.2),
            test_ratio=split_config.get("test_ratio", 0.2),
            shuffle=split_config.get("shuffle", False),
        )

    raise ValueError(f"Unknown split strategy: {split_strategy}")


def metadata_at(dataset, idx: int) -> dict:
    if hasattr(dataset, "_data_"):
        return dataset._data_[idx][1]

    _, metadata = dataset[idx]
    return metadata


def split_distribution(dataset, indices: list[int] | np.ndarray) -> dict:
    labels = Counter()
    sessions = Counter()

    for idx in indices:
        metadata = metadata_at(dataset, int(idx))
        labels[labels_to_id(metadata.get("labels", [])).label] += 1
        sessions[metadata.get("session_id", "unknown")] += 1

    return {
        "size": len(indices),
        "labels": dict(sorted(labels.items())),
        "sessions": dict(sorted(sessions.items())),
    }


def build_split_summary(dataset, train_idx: list[int], val_idx: list[int], test_idx: list[int]) -> dict:
    return {
        "dataset_size": len(dataset),
        "splits": {
            "train": split_distribution(dataset, train_idx),
            "val": split_distribution(dataset, val_idx),
            "test": split_distribution(dataset, test_idx),
        },
    }


class Trainer:
    def __init__(self, config: dict):
        self.config = config
        self.test_mode = self.config.get('test_mode', False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_path = self.config.get('training_log_folder')
        self.wandb_run = None
        self.best_model_path = os.path.join(self.output_path, "best_model.pth")
        self.config_path = os.path.join(self.output_path, "config.yaml")
        self.split_path = os.path.join(self.output_path, "data_split.npz")
        self.split_summary_path = os.path.join(self.output_path, "split_summary.yaml")
        self.train_log_path = os.path.join(self.output_path, "train_log.npz")
        self.global_step = 0

        if not self.test_mode:
            # Save configuration in output folder
            os.makedirs(self.output_path, exist_ok=True)
            with open(self.config_path, 'w') as f:
                f.write(yaml.safe_dump(config))

    def __prepare_wandb__(self):
        wandb_config = self.config.get("wandb") or {}
        if not wandb_config.get("enabled", False):
            return

        if wandb is None:
            raise ImportError("wandb is enabled but not installed. Run: uv add wandb")

        self.wandb_run = wandb.init(
            project=wandb_config.get("project", "potholes-detection"),
            entity=wandb_config.get("entity"),
            name=wandb_config.get("name"),
            tags=wandb_config.get("tags"),
            group=wandb_config.get("group"),
            mode=wandb_config.get("mode", "online"),
            dir=wandb_config.get("dir", self.output_path),
            job_type=wandb_config.get("job_type", "train"),
            config=self.config,
            save_code=wandb_config.get("save_code", True),
        )

    def __log_wandb_artifact__(self, name: str, artifact_type: str, files: list[str], aliases: list[str] | None = None):
        if self.wandb_run is None:
            return

        artifact = wandb.Artifact(name=name, type=artifact_type)
        for file_path in files:
            if os.path.exists(file_path):
                artifact.add_file(file_path)

        self.wandb_run.log_artifact(artifact, aliases=aliases)

    def __log_wandb_data__(self):
        if self.wandb_run is None:
            return

        wandb_config = self.config.get("wandb") or {}
        self.wandb_run.summary["dataset_size"] = self.split_summary["dataset_size"]
        for split_name, split in self.split_summary["splits"].items():
            self.wandb_run.summary[f"{split_name}_size"] = split["size"]
            for label, count in split["labels"].items():
                self.wandb_run.summary[f"{split_name}_{label}_count"] = count

        if wandb_config.get("log_split_artifact", True):
            self.__log_wandb_artifact__(
                name=wandb_config.get("split_artifact_name", "potholes-session-split"),
                artifact_type="data-split",
                files=[self.config_path, self.split_path, self.split_summary_path],
                aliases=["latest"],
            )

        if wandb_config.get("log_dataset_artifact", False):
            artifact = wandb.Artifact(
                name=wandb_config.get("dataset_artifact_name", "potholes-dataset"),
                type="dataset",
            )
            artifact.add_dir(self.config.get("data", {}).get("data_folder", "dataset"))
            self.wandb_run.log_artifact(artifact, aliases=["latest"])

    def __prepare_data__(self):
        dataset = get_dataset(config=self.config.get('data'))

        train_idx, val_idx, test_idx = build_split_indices(dataset, self.config)

        print(f"Dataset size: {len(dataset)}")
        print(f"Train/val/test sizes: {len(train_idx)}, {len(val_idx)}, {len(test_idx)}")
        np.savez_compressed(self.split_path, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

        self.split_summary = build_split_summary(dataset, train_idx, val_idx, test_idx)
        with open(self.split_summary_path, "w") as f:
            yaml.safe_dump(self.split_summary, f, sort_keys=False)

        self.__log_wandb_data__()

        self.train_dataset = Subset(dataset, train_idx)
        self.val_dataset = Subset(dataset, val_idx)

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.config.get('batch_size'),
                shuffle=True, drop_last=False, pin_memory=True,
                collate_fn=collate_fn)

        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.config.get('batch_size'),
                shuffle=False, drop_last=False, pin_memory=True,
                collate_fn=collate_fn)

    def __prepare_model__(self):
        self.model = load_model(self.config.get("model")).to(self.device)
        if self.wandb_run is not None and hasattr(self.model, "channel_names"):
            self.wandb_run.summary["model_channels"] = self.model.channel_names

    def __prepare_loss__(self):
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def __prepare_optimizers__(self):
        optimizer_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(optimizer_params, lr=self.config.get('learning_rate'))
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.get('epochs'))
        if self.wandb_run is not None:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.wandb_run.summary["total_params"] = total_params
            self.wandb_run.summary["trainable_params"] = trainable_params

    def __train__(self):
        self.model.train()
        running_loss = 0.0
        total_train = 0
        train_correct = 0
        wandb_config = self.config.get("wandb") or {}
        log_batch_metrics = self.wandb_run is not None and wandb_config.get("log_batch_metrics", True)
        log_batch_interval = max(1, int(wandb_config.get("log_batch_interval", 25)))

        with tqdm(total=len(self.train_loader), position=1, leave=False) as self.batch_pbar:
            self.batch_pbar.set_description("Training")
            for i, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                preds = logits.argmax(dim=1)
                total_train += labels.size(0)
                batch_correct = (preds == labels).sum().item()
                train_correct += batch_correct
                running_loss += loss.item() * images.size(0)
                batch_acc = batch_correct / labels.size(0)
                running_avg_loss = running_loss / total_train
                running_avg_acc = train_correct / total_train
                self.global_step += 1

                should_log_batch = log_batch_metrics and (
                    self.global_step == 1
                    or self.global_step % log_batch_interval == 0
                    or i == len(self.train_loader) - 1
                )
                if should_log_batch:
                    self.wandb_run.log({
                        "epoch": self.epoch,
                        "train/batch_loss": loss.item(),
                        "train/batch_acc": batch_acc,
                        "train/running_loss": running_avg_loss,
                        "train/running_acc": running_avg_acc,
                        "lr": self.scheduler.get_last_lr()[0],
                    }, step=self.global_step)

                self.batch_pbar.set_postfix(loss=loss.item())
                self.batch_pbar.update(1)

        log = {
            'train_loss': running_loss / len(self.train_dataset),
            'train_acc': train_correct / total_train
        }

        return log

    def __validate__(self):
        self.model.eval()
        running_val_loss = 0.0
        total_val = 0
        val_correct = 0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            with tqdm(total=len(self.val_loader), position=1, leave=False) as self.batch_pbar:
                self.batch_pbar.set_description("Validating")
                for i, (images, labels) in enumerate(self.val_loader):
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                    running_val_loss += loss.item() * images.size(0)

                    probs = torch.softmax(logits, dim=1)
                    all_probs.append(probs.detach().cpu())
                    all_labels.append(labels.detach().cpu())

                    preds = logits.argmax(dim=1)
                    total_val += labels.size(0)
                    val_correct += (preds == labels).sum().item()

                    self.batch_pbar.set_postfix(val_loss=loss.item())
                    self.batch_pbar.update(1)

        probs_np = torch.cat(all_probs, dim=0).numpy()
        labels_np = torch.cat(all_labels, dim=0).numpy()
        auc = roc_auc_score(labels_np, probs_np, multi_class='ovr', average='macro')

        log = {
            'val_loss': running_val_loss / len(self.val_dataset),
            'val_acc': val_correct / total_val,
            'val_auc': auc
        }

        return log

    def __call__(self, *args, **kwargs):
        try:
            self.__prepare_wandb__()
            self.__prepare_data__()
            self.__prepare_model__()
            self.__prepare_loss__()
            self.__prepare_optimizers__()

            epochs = self.config.get('epochs')
            train_log_history = []
            val_log_history = []

            patience_counter = 0
            best_val_loss = float("inf")
            best_val_auc = 0.0

            with tqdm(total=epochs, position=0, leave=True) as self.epochs_pbar:
                self.epochs_pbar.set_description("Epochs")
                for self.epoch in range(epochs):
                    train_log = self.__train__()
                    train_log_history.append(train_log)

                    val_log = self.__validate__()
                    val_log_history.append(val_log)

                    improved = (val_log.get('val_loss') < best_val_loss) or (val_log.get('val_auc') > best_val_auc)
                    if improved:
                        best_val_loss = min(val_log.get('val_loss'), best_val_loss)
                        best_val_auc = max(val_log.get('val_auc'), best_val_auc)
                        patience_counter = 0
                        torch.save(self.model.state_dict(), self.best_model_path)
                    else:
                        patience_counter += 1

                    if self.wandb_run is not None:
                        self.wandb_run.log({
                            "epoch": self.epoch,
                            "train/loss": train_log.get("train_loss"),
                            "train/acc": train_log.get("train_acc"),
                            "val/loss": val_log.get("val_loss"),
                            "val/acc": val_log.get("val_acc"),
                            "val/auc": val_log.get("val_auc"),
                            "lr": self.scheduler.get_last_lr()[0],
                            "best/val_loss": best_val_loss,
                            "best/val_auc": best_val_auc,
                        }, step=self.global_step)

                    if patience_counter > self.config.get('patience'):
                        click.echo(f"Early stopping after {self.config.get('patience')} epochs with no improvement.")
                        break

                    self.scheduler.step()

                    self.epochs_pbar.set_postfix(train_loss=train_log.get('train_loss'),
                                                 train_acc=train_log.get('train_acc'),
                                                 val_loss=val_log.get('val_loss'),
                                                 val_acc=val_log.get('val_acc'),
                                                 val_auc=val_log.get('val_auc'))
                    self.epochs_pbar.update(1)

            np.savez_compressed(self.train_log_path,
                                train_log_history=train_log_history,
                                val_log_history=val_log_history)

            wandb_config = self.config.get("wandb") or {}
            if wandb_config.get("log_training_artifact", True):
                self.__log_wandb_artifact__(
                    name=wandb_config.get("training_artifact_name", "potholes-training-log"),
                    artifact_type="training-log",
                    files=[self.config_path, self.split_path, self.split_summary_path, self.train_log_path],
                    aliases=["latest"],
                )

            if wandb_config.get("log_model_artifact", True):
                if os.path.exists(self.best_model_path):
                    self.__log_wandb_artifact__(
                        name=wandb_config.get("model_artifact_name", "potholes-ast-classifier"),
                        artifact_type="model",
                        files=[self.best_model_path, self.config_path, self.split_path, self.split_summary_path],
                        aliases=["best"],
                    )
        finally:
            if self.wandb_run is not None:
                self.wandb_run.finish()


    @staticmethod
    def test(model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} does not exist")

        config_file = os.path.join(model_path, 'config.yaml')
        model_file = os.path.join(model_path, 'best_model.pth')
        split_file = os.path.join(model_path, 'data_split.npz')

        if not os.path.exists(config_file):
            raise FileNotFoundError(f"{config_file} does not exist")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"{model_file} does not exist")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"{split_file} does not exist")

        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        config['test_mode'] = True

        dataset = get_dataset(config=config.get("data", config))
        split = np.load(split_file)
        test_idx = split['test_idx']

        test_dataset = Subset(dataset, test_idx)
        test_loader = DataLoader(test_dataset, batch_size=config.get('batch_size'),
                                      shuffle=False, drop_last=False, collate_fn=collate_fn)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(config.get("model")).to(device)
        model.load_state_dict(torch.load(model_file, map_location=device))
        model.eval()

        all_probs = []
        all_labels = []
        all_preds = []
        test_correct = 0
        total_test = 0

        with torch.no_grad():
            with tqdm(total=len(test_loader), desc="Testing") as pbar:
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    logits = model(images)
                    probs = torch.softmax(logits, dim=1)
                    preds = logits.argmax(dim=1)

                    all_probs.append(probs.cpu())
                    all_labels.append(labels.cpu())
                    all_preds.append(preds.cpu())

                    total_test += labels.size(0)
                    test_correct += (preds == labels).sum().item()

                    pbar.update(1)

        probs_np = torch.cat(all_probs, dim=0).numpy()
        labels_np = torch.cat(all_labels, dim=0).numpy()
        preds_np = torch.cat(all_preds, dim=0).numpy()

        accuracy = 100.0 * test_correct / total_test
        auc = roc_auc_score(labels_np, probs_np, multi_class="ovr", average="macro")

        print(f"Test Accuracy: {accuracy:.2f}%")
        print(f"Test AUC (macro-ovr): {auc:.4f}")

        labels = [l.label for l in RoadLabel]
        cm = confusion_matrix(labels_np, preds_np)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig = go.Figure(
            data=go.Heatmap(
                z=cm_norm,
                x=labels,
                y=labels,
                colorscale="Blues",
                text=np.round(cm_norm, 2),
                texttemplate="%{text}",
                hovertemplate="True: %{y}<br>Pred: %{x}<br>Value: %{z:.3f}<extra></extra>"
            )
        )

        fig.update_layout(
            title="Normalized Confusion Matrix",
            xaxis_title="Predicted Label",
            yaxis_title="True Label"
        )

        fig.show()
        fig.write_image(os.path.join(model_path, 'confusion_matrix.png'))

if __name__=="__main__":
    import click
    import pathlib
    from datetime import datetime
    from click.core import ParameterSource

    @click.group()
    def cli():
        pass

    @cli.command()
    @click.argument("model_path", type=click.Path(exists=True, dir_okay=True, file_okay=False,
                                                  readable=True, path_type=pathlib.Path), required=True)
    def test(model_path: pathlib.Path):
        Trainer.test(str(model_path))


    def _default_training_log_folder() -> str:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return os.path.join("output", "training", f"training-{timestamp}")

    def _merge_config(base_config: dict, overrides: dict) -> dict:
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

        for key, value in overrides.get("train", {}).items():
            if value is not None:
                config[key] = value

        for key, value in overrides.get("data", {}).items():
            if value is not None:
                data_config[key] = value

        if not config.get("training_log_folder"):
            config["training_log_folder"] = _default_training_log_folder()

        return config

    @cli.command()
    @click.argument(
        "config_file",
        required=False,
        type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True),
    )
    @click.option(
        "--use-defaults",
        is_flag=True,
        default=False,
        help="Allow running without a config file using built-in defaults.",
        show_default=True,
    )
    @click.option("--training-log-folder", type=click.Path(dir_okay=True, file_okay=False), default=None, show_default="auto")
    @click.option("--batch-size", type=int, default=8, show_default=True)
    @click.option("--epochs", type=int, default=100, show_default=True)
    @click.option("--learning-rate", type=float, default=0.005, show_default=True)
    @click.option("--patience", type=int, default=10, show_default=True)
    @click.option("--data-folder", type=click.Path(exists=True, dir_okay=True, file_okay=False, readable=True), default="dataset", show_default=True)
    @click.option("--data-version", type=click.IntRange(1, 2), default=2, show_default=True)
    @click.option("--generate/--no-generate", default=False, show_default=True)
    @click.option("--window-size", type=int, default=10, show_default=True)
    @click.option("--step", type=int, default=1, show_default=True)
    @click.option("--verbose/--no-verbose", default=True, show_default=True)
    def train(
        config_file: str,
        use_defaults: bool,
        training_log_folder: str,
        batch_size: int,
        epochs: int,
        learning_rate: float,
        patience: int,
        data_folder: str,
        data_version: int,
        generate: bool,
        window_size: int,
        step: int,
        verbose: bool,
    ):
        ctx = click.get_current_context()

        def _is_provided(param_name: str) -> bool:
            return ctx.get_parameter_source(param_name) != ParameterSource.DEFAULT

        if config_file is None and not use_defaults:

            click.echo(
                click.style("\nError: Provide CONFIG_FILE or pass --use-defaults to run with built-in defaults.",
                           fg="red", bold=True), err=True, nl=True, color=True)
            click.echo(ctx.get_help())
            ctx.exit(2)

        config = {}
        if config_file:
            with open(config_file, "r") as f:
                config = yaml.safe_load(f) or {}

        overrides = {"train": {}, "data": {}}
        if _is_provided("training_log_folder"):
            overrides["train"]["training_log_folder"] = training_log_folder
        if _is_provided("batch_size"):
            overrides["train"]["batch_size"] = batch_size
        if _is_provided("epochs"):
            overrides["train"]["epochs"] = epochs
        if _is_provided("learning_rate"):
            overrides["train"]["learning_rate"] = learning_rate
        if _is_provided("patience"):
            overrides["train"]["patience"] = patience
        if _is_provided("data_folder"):
            overrides["data"]["data_folder"] = data_folder
        if _is_provided("data_version"):
            overrides["data"]["version"] = data_version
        if _is_provided("generate"):
            overrides["data"]["generate"] = generate
        if _is_provided("window_size"):
            overrides["data"]["window_size"] = window_size
        if _is_provided("step"):
            overrides["data"]["step"] = step
        if _is_provided("verbose"):
            overrides["data"]["verbose"] = verbose

        config = _merge_config(config, overrides)
        trainer = Trainer(config=config)
        trainer()

    def _print_split_summary(dataset, train_idx: list[int], val_idx: list[int], test_idx: list[int]):
        summary = build_split_summary(dataset, train_idx, val_idx, test_idx)
        click.echo(f"Dataset size: {summary['dataset_size']}")
        click.echo(f"Train/val/test sizes: {len(train_idx)}, {len(val_idx)}, {len(test_idx)}")

        for name, split in summary["splits"].items():
            click.echo(f"{name} labels: {split['labels']}")
            click.echo(f"{name} sessions: {split['sessions']}")

    @cli.command(name="split")
    @click.argument(
        "config_file",
        type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True),
    )
    @click.option(
        "--output-file",
        type=click.Path(dir_okay=False, file_okay=True, path_type=pathlib.Path),
        default=pathlib.Path("output/splits/data_split.npz"),
        show_default=True,
        help="Where to save train_idx, val_idx, and test_idx.",
    )
    def split_dataset(config_file: str, output_file: pathlib.Path):
        with open(config_file, "r") as f:
            config = yaml.safe_load(f) or {}

        config = _merge_config(config, {"train": {}, "data": {}})
        dataset = get_dataset(config=config.get("data"))
        train_idx, val_idx, test_idx = build_split_indices(dataset, config)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_file, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
        _print_split_summary(dataset, train_idx, val_idx, test_idx)
        click.echo(f"Saved split to {output_file}")

    cli()
