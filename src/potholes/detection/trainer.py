import os

import click
import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from potholes.detection.config import normalize_run_config, prepare_config, run_artifact_name, run_output_path
from potholes.detection.data import get_dataset
from potholes.detection.data.dataset import labels_to_id
from potholes.detection.data.split import build_split_indices, build_split_summary
from potholes.detection.models import load_model
from potholes.detection.plots import confusion_matrix_figure, write_confusion_matrix_image

try:
    import wandb
except ImportError:
    wandb = None


def collate_fn(batch):
    samples, metas = zip(*batch)
    samples = torch.stack(samples)
    labels = [labels_to_id(m['labels']).value for m in metas]
    return samples, torch.tensor(labels, dtype=torch.long)


class Trainer:
    def __init__(self, config: dict, training_log_base_folder: str = os.path.join("output", "training")):
        self.config = normalize_run_config(config)
        self.test_mode = self.config.get('test_mode', False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_path = run_output_path(training_log_base_folder, self.config["run_name"])
        self.wandb_run = None
        self.best_model_path = os.path.join(self.output_path, "best_model.pth")
        self.config_path = os.path.join(self.output_path, "config.yaml")
        self.split_path = os.path.join(self.output_path, "data_split.npz")
        self.split_summary_path = os.path.join(self.output_path, "split_summary.yaml")
        self.train_log_path = os.path.join(self.output_path, "train_log.npz")
        self.confusion_matrix_folder = os.path.join(self.output_path, "confusion_matrices")
        self.global_step = 0

        if not self.test_mode:
            # Save configuration in output folder
            os.makedirs(self.output_path, exist_ok=True)
            with open(self.config_path, 'w') as f:
                f.write(yaml.safe_dump(self.config))

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
            dir=self.output_path,
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
                name=run_artifact_name(self.config["run_name"], "split"),
                artifact_type="data-split",
                files=[self.config_path, self.split_path, self.split_summary_path],
                aliases=["latest"],
            )

        if wandb_config.get("log_dataset_artifact", False):
            artifact = wandb.Artifact(
                name=run_artifact_name(self.config["run_name"], "dataset"),
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
        if self.wandb_run is not None and hasattr(self.model, "freeze_encoder"):
            self.wandb_run.summary["freeze_encoder"] = self.model.freeze_encoder
        if self.wandb_run is not None and getattr(self.model, "encoder_checkpoint", None):
            self.wandb_run.summary["encoder_checkpoint"] = self.model.encoder_checkpoint

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
        all_preds = []

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
                    all_preds.append(preds.detach().cpu())
                    total_val += labels.size(0)
                    val_correct += (preds == labels).sum().item()

                    self.batch_pbar.set_postfix(val_loss=loss.item())
                    self.batch_pbar.update(1)

        probs_np = torch.cat(all_probs, dim=0).numpy()
        labels_np = torch.cat(all_labels, dim=0).numpy()
        preds_np = torch.cat(all_preds, dim=0).numpy()
        auc = roc_auc_score(labels_np, probs_np, multi_class='ovr', average='macro')

        log = {
            'val_loss': running_val_loss / len(self.val_dataset),
            'val_acc': val_correct / total_val,
            'val_auc': auc,
            'confusion_matrix': confusion_matrix_figure(
                labels_np,
                preds_np,
                title=f"Validation Confusion Matrix - Epoch {self.epoch}",
            ),
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
                    confusion_matrix_plot = val_log.pop("confusion_matrix", None)
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
                        wandb_log = {
                            "epoch": self.epoch,
                            "train/loss": train_log.get("train_loss"),
                            "train/acc": train_log.get("train_acc"),
                            "val/loss": val_log.get("val_loss"),
                            "val/acc": val_log.get("val_acc"),
                            "val/auc": val_log.get("val_auc"),
                            "lr": self.scheduler.get_last_lr()[0],
                            "best/val_loss": best_val_loss,
                            "best/val_auc": best_val_auc,
                        }
                        if confusion_matrix_plot is not None:
                            wandb_log["val/confusion_matrix"] = wandb.Plotly(confusion_matrix_plot)

                        self.wandb_run.log(wandb_log, step=self.global_step)

                    if confusion_matrix_plot is not None:
                        os.makedirs(self.confusion_matrix_folder, exist_ok=True)
                        confusion_matrix_plot.write_image(
                            os.path.join(
                                self.confusion_matrix_folder,
                                f"val_epoch_{self.epoch:04d}.png",
                            )
                        )

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
                    name=run_artifact_name(self.config["run_name"], "training-log"),
                    artifact_type="training-log",
                    files=[self.config_path, self.split_path, self.split_summary_path, self.train_log_path],
                    aliases=["latest"],
                )

            if wandb_config.get("log_model_artifact", True):
                if os.path.exists(self.best_model_path):
                    self.__log_wandb_artifact__(
                        name=run_artifact_name(self.config["run_name"], "model"),
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

        fig = write_confusion_matrix_image(
            labels_np,
            preds_np,
            os.path.join(model_path, 'confusion_matrix.png'),
        )
        fig.show()

if __name__=="__main__":
    import click
    import pathlib

    @click.group()
    def cli():
        pass

    @cli.command()
    @click.argument("model_path", type=click.Path(exists=True, dir_okay=True, file_okay=False,
                                                  readable=True, path_type=pathlib.Path), required=True)
    def test(model_path: pathlib.Path):
        Trainer.test(str(model_path))


    @cli.command()
    @click.argument(
        "config_file",
        required=True,
        type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True),
    )
    @click.option(
        "--training-log-folder",
        type=click.Path(dir_okay=True, file_okay=False),
        default=os.path.join("output", "training"),
        show_default=True,
        help="Base folder where run output folders are created.",
    )
    def train(
        config_file: str,
        training_log_folder: str,
    ):
        with open(config_file, "r") as f:
            config = yaml.safe_load(f) or {}

        config = prepare_config(config)
        trainer = Trainer(config=config, training_log_base_folder=training_log_folder)
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

        config = prepare_config(config)
        dataset = get_dataset(config=config.get("data"))
        train_idx, val_idx, test_idx = build_split_indices(dataset, config)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_file, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
        _print_split_summary(dataset, train_idx, val_idx, test_idx)
        click.echo(f"Saved split to {output_file}")

    cli()
