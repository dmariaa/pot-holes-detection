import os

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
from potholes.detection.data.dataset import labels_to_id, stratified_split_indices, RoadLabel
from potholes.detection.models.transformer import load_model


def collate_fn(batch):
    samples, metas = zip(*batch)
    samples = torch.stack(samples)
    labels = [labels_to_id(m['labels']).value for m in metas]
    return samples, torch.tensor(labels, dtype=torch.long)


class Trainer:
    def __init__(self, config: dict):
        self.config = config
        self.test_mode = self.config.get('test_mode', False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_path = self.config.get('training_log_folder')

        if not self.test_mode:
            # Save configuration in output folder
            os.makedirs(self.output_path, exist_ok=True)
            with open(os.path.join(self.output_path, 'config.yaml'), 'w') as f:
                f.write(yaml.safe_dump(config))

    def __prepare_data__(self):
        dataset = get_dataset(config=self.config.get('data'))

        train_idx, val_idx, test_idx = stratified_split_indices(dataset, val_ratio=0.2, test_ratio=0.2, shuffle=False)

        print(f"Dataset size: {len(dataset)}")
        print(f"Train/val/test sizes: {len(train_idx)}, {len(val_idx)}, {len(test_idx)}")
        np.savez_compressed(os.path.join(self.output_path, 'data_split.npz'),
                            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

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
        self.model = load_model().to(self.device)

    def __prepare_loss__(self):
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def __prepare_optimizers__(self):
        optimizer_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(optimizer_params, lr=self.config.get('learning_rate'))
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.get('epochs'))

    def __train__(self):
        self.model.train()
        running_loss = 0.0
        total_train = 0
        train_correct = 0

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
                train_correct += (preds==labels).sum().item()
                running_loss += loss.item() * images.size(0)

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
                    torch.save(self.model.state_dict(), os.path.join(self.output_path, f"best_model.pth"))
                else:
                    patience_counter += 1

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

        np.savez_compressed(os.path.join(self.output_path, 'train_log.npz'),
                            train_log_history=train_log_history,
                            val_log_history=val_log_history)


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

        dataset = get_dataset(config=config)
        split = np.load(split_file)
        test_idx = split['test_idx']

        test_dataset = Subset(dataset, test_idx)
        test_loader = DataLoader(test_dataset, batch_size=config.get('batch_size'),
                                      shuffle=False, drop_last=False, collate_fn=collate_fn)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model().to(device)
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

if __name__=="__main__":
    training_session = "training-002-old-data"

    config_trainer = {
        'batch_size': 8,
        'epochs': 100,
        'learning_rate': 0.005,
        'patience': 10,
        'data': {
            'version': 1,
            'data_folder': 'data_old',
            'generate': False,
            'window_size': 10,
            'step': 1,
            'verbose': True,
        },
        'training_log_folder': os.path.join('output/training', training_session)
    }

    trainer = Trainer(config_trainer)
    trainer()

    # Trainer.test(os.path.join('output/training', training_session))