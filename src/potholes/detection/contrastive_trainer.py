import os

import click
import numpy as np
import torch
import yaml
from torch import optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from potholes.detection.config import normalize_run_config, prepare_config, run_artifact_name, run_output_path
from potholes.detection.data import get_dataset
from potholes.detection.data.dataset import labels_to_id
from potholes.detection.data.split import build_split_indices, build_split_summary, metadata_at
from potholes.detection.losses import SupervisedContrastiveLoss
from potholes.detection.models.contrastive import load_contrastive_model
from potholes.detection.samplers import BalancedLabelBatchSampler

try:
    import wandb
except ImportError:
    wandb = None


def contrastive_collate_fn(batch):
    samples, metas = zip(*batch)
    samples = torch.stack(samples)
    labels = [labels_to_id(m["labels"]).value for m in metas]
    return samples, torch.tensor(labels, dtype=torch.long)


def subset_labels(dataset: Subset) -> list[int]:
    labels = []
    for index in dataset.indices:
        metadata = metadata_at(dataset.dataset, int(index))
        labels.append(labels_to_id(metadata.get("labels", [])).value)
    return labels


def augment_batch(images: torch.Tensor, config: dict) -> torch.Tensor:
    config = config or {}
    x = images.clone()

    scale = float(config.get("amplitude_scale", 0.0))
    if scale > 0:
        factors = torch.empty(x.size(0), 1, 1, 1, device=x.device).uniform_(1.0 - scale, 1.0 + scale)
        x = x * factors

    noise_std = float(config.get("noise_std", 0.0))
    if noise_std > 0:
        x = x + torch.randn_like(x) * noise_std

    channel_dropout = float(config.get("channel_dropout", 0.0))
    if channel_dropout > 0 and x.size(1) > 1:
        keep = torch.rand(x.size(0), x.size(1), 1, 1, device=x.device) > channel_dropout
        empty = keep.flatten(1).sum(dim=1) == 0
        if empty.any():
            keep[empty, torch.randint(0, x.size(1), (int(empty.sum()),), device=x.device), 0, 0] = True
        x = x * keep

    freq_mask_fraction = float(config.get("freq_mask_fraction", 0.0))
    if freq_mask_fraction > 0:
        max_width = max(1, int(x.size(2) * freq_mask_fraction))
        for i in range(x.size(0)):
            width = int(torch.randint(1, max_width + 1, (1,), device=x.device).item())
            start = int(torch.randint(0, max(1, x.size(2) - width + 1), (1,), device=x.device).item())
            x[i, :, start:start + width, :] = 0

    time_mask_fraction = float(config.get("time_mask_fraction", 0.0))
    if time_mask_fraction > 0:
        max_width = max(1, int(x.size(3) * time_mask_fraction))
        for i in range(x.size(0)):
            width = int(torch.randint(1, max_width + 1, (1,), device=x.device).item())
            start = int(torch.randint(0, max(1, x.size(3) - width + 1), (1,), device=x.device).item())
            x[i, :, :, start:start + width] = 0

    return x


def projection_stats(projections: torch.Tensor, labels: torch.Tensor) -> dict:
    similarity = torch.matmul(projections, projections.T)
    labels = labels.view(-1, 1)
    same_label = torch.eq(labels, labels.T)
    eye = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    positive_mask = same_label & ~eye
    negative_mask = ~same_label

    stats = {}
    if positive_mask.any():
        stats["positive_similarity"] = similarity[positive_mask].mean().item()
    if negative_mask.any():
        stats["negative_similarity"] = similarity[negative_mask].mean().item()
    return stats


class ContrastivePretrainer:
    def __init__(self, config: dict, output_base_folder: str = os.path.join("output", "contrastive")):
        self.config = normalize_run_config(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_path = run_output_path(output_base_folder, self.config["run_name"])
        self.config_path = os.path.join(self.output_path, "config.yaml")
        self.split_path = os.path.join(self.output_path, "data_split.npz")
        self.split_summary_path = os.path.join(self.output_path, "split_summary.yaml")
        self.log_path = os.path.join(self.output_path, "contrastive_log.npz")
        self.model_path = os.path.join(self.output_path, "contrastive_model.pth")
        self.encoder_path = os.path.join(self.output_path, "encoder.pth")
        self.wandb_run = None
        self.global_step = 0

        os.makedirs(self.output_path, exist_ok=True)
        with open(self.config_path, "w") as f:
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
            job_type=wandb_config.get("job_type", "contrastive-pretrain"),
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

    def __prepare_data__(self):
        dataset = get_dataset(config=self.config.get("data"))
        train_idx, val_idx, test_idx = build_split_indices(dataset, self.config)
        np.savez_compressed(self.split_path, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

        self.split_summary = build_split_summary(dataset, train_idx, val_idx, test_idx)
        with open(self.split_summary_path, "w") as f:
            yaml.safe_dump(self.split_summary, f, sort_keys=False)

        self.train_dataset = Subset(dataset, train_idx)
        self.val_dataset = Subset(dataset, val_idx)

        contrastive_config = self.config.get("contrastive") or {}
        classes_per_batch = int(contrastive_config.get("classes_per_batch", 4))
        samples_per_class = int(contrastive_config.get("samples_per_class", 4))
        batch_size = classes_per_batch * samples_per_class
        train_batches = contrastive_config.get("batches_per_epoch")
        if train_batches is None:
            train_batches = max(1, len(self.train_dataset) // batch_size)
        val_batches = contrastive_config.get("val_batches")
        if val_batches is None:
            val_batches = max(1, min(50, len(self.val_dataset) // batch_size))

        seed = contrastive_config.get("seed")
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_sampler=BalancedLabelBatchSampler(
                subset_labels(self.train_dataset),
                classes_per_batch=classes_per_batch,
                samples_per_class=samples_per_class,
                batches_per_epoch=int(train_batches),
                seed=seed,
            ),
            collate_fn=contrastive_collate_fn,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_sampler=BalancedLabelBatchSampler(
                subset_labels(self.val_dataset),
                classes_per_batch=classes_per_batch,
                samples_per_class=samples_per_class,
                batches_per_epoch=int(val_batches),
                seed=None if seed is None else int(seed) + 1,
            ),
            collate_fn=contrastive_collate_fn,
            pin_memory=True,
        )

        print(f"Dataset size: {len(dataset)}")
        print(f"Train/val/test sizes: {len(train_idx)}, {len(val_idx)}, {len(test_idx)}")
        print(f"Contrastive batch size: {batch_size}")
        print(f"Train/val batches: {len(self.train_loader)}, {len(self.val_loader)}")

        if self.wandb_run is not None:
            self.wandb_run.summary["dataset_size"] = self.split_summary["dataset_size"]
            self.wandb_run.summary["contrastive_batch_size"] = batch_size
            for split_name, split in self.split_summary["splits"].items():
                self.wandb_run.summary[f"{split_name}_size"] = split["size"]
                for label, count in split["labels"].items():
                    self.wandb_run.summary[f"{split_name}_{label}_count"] = count

    def __prepare_model__(self):
        self.model = load_contrastive_model(self.config.get("model")).to(self.device)
        if self.wandb_run is not None:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.wandb_run.summary["total_params"] = total_params
            self.wandb_run.summary["trainable_params"] = trainable_params
            if hasattr(self.model, "channel_names"):
                self.wandb_run.summary["model_channels"] = self.model.channel_names

    def __prepare_loss_and_optimizer__(self):
        contrastive_config = self.config.get("contrastive") or {}
        self.criterion = SupervisedContrastiveLoss(
            temperature=float(contrastive_config.get("temperature", 0.1))
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.get("learning_rate"))
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.get("epochs"))

    def __train_epoch__(self):
        self.model.train()
        running_loss = 0.0
        running_pos_sim = 0.0
        running_neg_sim = 0.0
        stats_count = 0
        aug_config = (self.config.get("contrastive") or {}).get("augmentations") or {}
        wandb_config = self.config.get("wandb") or {}
        log_batch_interval = max(1, int(wandb_config.get("log_batch_interval", 25)))

        with tqdm(total=len(self.train_loader), position=1, leave=False) as pbar:
            pbar.set_description("Contrastive training")
            for i, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                view_1 = augment_batch(images, aug_config)
                view_2 = augment_batch(images, aug_config)
                views = torch.cat([view_1, view_2], dim=0)
                view_labels = torch.cat([labels, labels], dim=0)

                self.optimizer.zero_grad(set_to_none=True)
                _, projections = self.model(views)
                loss = self.criterion(projections, view_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                self.global_step += 1
                running_loss += loss.item()
                stats = projection_stats(projections.detach(), view_labels)
                if "positive_similarity" in stats and "negative_similarity" in stats:
                    running_pos_sim += stats["positive_similarity"]
                    running_neg_sim += stats["negative_similarity"]
                    stats_count += 1

                if self.wandb_run is not None and (
                        self.global_step == 1
                        or self.global_step % log_batch_interval == 0
                        or i == len(self.train_loader) - 1
                ):
                    log = {
                        "epoch": self.epoch,
                        "train/batch_loss": loss.item(),
                        "train/running_loss": running_loss / (i + 1),
                        "lr": self.scheduler.get_last_lr()[0],
                    }
                    log.update({f"train/{key}": value for key, value in stats.items()})
                    self.wandb_run.log(log, step=self.global_step)

                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

        log = {"train_loss": running_loss / len(self.train_loader)}
        if stats_count > 0:
            log["train_positive_similarity"] = running_pos_sim / stats_count
            log["train_negative_similarity"] = running_neg_sim / stats_count
        return log

    def __validate_epoch__(self):
        self.model.eval()
        running_loss = 0.0
        running_pos_sim = 0.0
        running_neg_sim = 0.0
        stats_count = 0

        with torch.no_grad():
            with tqdm(total=len(self.val_loader), position=1, leave=False) as pbar:
                pbar.set_description("Contrastive validation")
                for images, labels in self.val_loader:
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    _, projections = self.model(images)
                    loss = self.criterion(projections, labels)
                    running_loss += loss.item()
                    stats = projection_stats(projections, labels)
                    if "positive_similarity" in stats and "negative_similarity" in stats:
                        running_pos_sim += stats["positive_similarity"]
                        running_neg_sim += stats["negative_similarity"]
                        stats_count += 1

                    pbar.set_postfix(val_loss=loss.item())
                    pbar.update(1)

        log = {"val_loss": running_loss / len(self.val_loader)}
        if stats_count > 0:
            log["val_positive_similarity"] = running_pos_sim / stats_count
            log["val_negative_similarity"] = running_neg_sim / stats_count
        return log

    def __save_checkpoint__(self):
        torch.save(self.model.state_dict(), self.model_path)
        torch.save(self.model.encoder.state_dict(), self.encoder_path)

    def __call__(self):
        try:
            self.__prepare_wandb__()
            self.__prepare_data__()
            self.__prepare_model__()
            self.__prepare_loss_and_optimizer__()

            epochs = self.config.get("epochs")
            train_log_history = []
            val_log_history = []
            best_val_loss = float("inf")
            patience_counter = 0

            with tqdm(total=epochs, position=0, leave=True) as pbar:
                pbar.set_description("Contrastive epochs")
                for self.epoch in range(epochs):
                    train_log = self.__train_epoch__()
                    val_log = self.__validate_epoch__()
                    train_log_history.append(train_log)
                    val_log_history.append(val_log)

                    if val_log["val_loss"] < best_val_loss:
                        best_val_loss = val_log["val_loss"]
                        patience_counter = 0
                        self.__save_checkpoint__()
                    else:
                        patience_counter += 1

                    if self.wandb_run is not None:
                        log = {
                            "epoch": self.epoch,
                            "train/loss": train_log["train_loss"],
                            "val/loss": val_log["val_loss"],
                            "best/val_loss": best_val_loss,
                            "lr": self.scheduler.get_last_lr()[0],
                        }
                        for key, value in train_log.items():
                            if key != "train_loss":
                                log[f"train/{key.removeprefix('train_')}"] = value
                        for key, value in val_log.items():
                            if key != "val_loss":
                                log[f"val/{key.removeprefix('val_')}"] = value
                        self.wandb_run.log(log, step=self.global_step)

                    if patience_counter > self.config.get("patience"):
                        click.echo(f"Early stopping after {self.config.get('patience')} epochs with no improvement.")
                        break

                    self.scheduler.step()
                    pbar.set_postfix(train_loss=train_log["train_loss"], val_loss=val_log["val_loss"])
                    pbar.update(1)

            np.savez_compressed(
                self.log_path,
                train_log_history=train_log_history,
                val_log_history=val_log_history,
            )

            wandb_config = self.config.get("wandb") or {}
            if wandb_config.get("log_split_artifact", True):
                self.__log_wandb_artifact__(
                    name=run_artifact_name(self.config["run_name"], "contrastive-split"),
                    artifact_type="data-split",
                    files=[self.config_path, self.split_path, self.split_summary_path],
                    aliases=["latest"],
                )
            if wandb_config.get("log_model_artifact", True):
                self.__log_wandb_artifact__(
                    name=run_artifact_name(self.config["run_name"], "contrastive-encoder"),
                    artifact_type="model",
                    files=[self.encoder_path, self.model_path, self.config_path, self.split_path, self.split_summary_path],
                    aliases=["best"],
                )
        finally:
            if self.wandb_run is not None:
                self.wandb_run.finish()


if __name__ == "__main__":
    @click.group()
    def cli():
        pass

    @cli.command()
    @click.argument(
        "config_file",
        required=True,
        type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True),
    )
    @click.option(
        "--output-folder",
        type=click.Path(dir_okay=True, file_okay=False),
        default=os.path.join("output", "contrastive"),
        show_default=True,
        help="Base folder where contrastive run output folders are created.",
    )
    def train(config_file: str, output_folder: str):
        with open(config_file, "r") as f:
            config = yaml.safe_load(f) or {}

        config = prepare_config(config)
        pretrainer = ContrastivePretrainer(config=config, output_base_folder=output_folder)
        pretrainer()

    cli()
