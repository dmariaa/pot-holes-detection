# from transformers import AutoFeatureExtractor, ASTForAudioClassification
# from datasets import load_dataset
# import torch
#
# dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
# dataset = dataset.sort("id")
# sampling_rate = dataset.features["audio"].sampling_rate
#
# feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
# model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
#
# # audio file is decoded on the fly
# inputs = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
#
# with torch.no_grad():
#     logits = model(**inputs).logits
import torch
import torch.nn as nn
from transformers import ASTConfig, ASTForAudioClassification
# from data_loader import CustomCSVDataset


import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModelForImageClassification
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
import seaborn as sns

import torch
from torchvision import transforms

from potholes.old.model.dataset import CustomCSVDataset

NUM_LABELS = 6

pretrained_model = "MIT/ast-finetuned-audioset-10-10-0.4593"

config = ASTConfig.from_pretrained(pretrained_model)
config.num_labels = NUM_LABELS

ast_backbone = ASTForAudioClassification.from_pretrained(
    pretrained_model,
    config=config,
    ignore_mismatched_sizes=True,
)

# Freeze everything
for p in ast_backbone.parameters():
    p.requires_grad = False

# Unfreeze classifier head only
for p in ast_backbone.classifier.parameters():
    p.requires_grad = True

N = 3
transformer_blocks = ast_backbone.audio_spectrogram_transformer.encoder.layer
for block in transformer_blocks[-N:]:
    for p in block.parameters():
        p.requires_grad = True

# Unfreeze global LayerNorm
for name, param in ast_backbone.named_parameters():
    if "audio_spectrogram_transformer.layernorm" in name.lower():
        param.requires_grad = True


class ASTWrapper(nn.Module):
    """
    Expects input: [B, 6, H, W]
    Converts to:   [B, 1, H, W] by direct reduction (no conv),
    then resizes to: [B, T=1024, F=128] for AST (time, num_mel_bins)
    """
    def __init__(self, ast_model: ASTForAudioClassification):
        super().__init__()
        self.ast = ast_model
        self.num_mel_bins = self.ast.config.num_mel_bins      # 128
        self.max_length   = self.ast.config.max_length        # 1024

    def forward(self, pixel_values, **kwargs):
        # pixel_values: [B, 6, H, W]

        # 1) 6 -> 1 channel WITHOUT conv (direct reduction)
        x = pixel_values.mean(dim=1, keepdim=True)    # [B, 1, H, W]
        # or: x = pixel_values[:, 0:1, :, :]

        # 2) Resize to AST's expected (freq, time) = (num_mel_bins, max_length)
        #    Shape: [B, 1, num_mel_bins=128, max_length=1024]
        x = F.interpolate(
            x,
            size=(self.num_mel_bins, self.max_length),
            mode="bilinear",
            align_corners=False,
        )  # [B, 1, 128, 1024]

        # 3) Convert to [B, time, num_mel_bins]
        x = x.squeeze(1)           # [B, 128, 1024]
        x = x.permute(0, 2, 1)     # [B, 1024, 128] = [B, T, F]

        # 4) Feed to AST
        out = self.ast(input_values=x, **kwargs)
        return out.logits          # [B, NUM_LABELS]

# 4) Wrap model with custom CNN stem (6 -> 3 channels)
# class ViTMSNWithStem(nn.Module):
#     def __init__(self, vit_model: ViTMSNForImageClassification):
#         super().__init__()
#         self.stem = nn.Sequential(
#             nn.Conv2d(6, 32, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(32),
#             nn.GELU(),
#             nn.Conv2d(32, 3, kernel_size=1, bias=True),
#         )
#         self.vit = vit_model
#
#     def forward(self, pixel_values, **kwargs):
#         # pixel_values: [B, 6, H, W]
#         x3 = self.stem(pixel_values)          # -> [B, 3, H, W]
#         return self.vit(pixel_values=x3, **kwargs)





# Build wrapped AST model with stem
model = ASTWrapper(ast_backbone)
trainable = [n for n, p in model.named_parameters() if p.requires_grad]
print("Trainable params:", trainable)



def count_parameters(m):
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable

total_params, trainable_params = count_parameters(model)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")


# 8) Example forward pass with 6-channel input
# x = torch.randn(2, 6, 224, 224)
# out = model(pixel_values=x)
# logits = out.logits
# print("Logits shape:", logits.shape)


def train(num_epochs, train_csv_file, val_csv_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 8
    patience = 10
    patience_counter = 0
    best_val_loss = float("inf")
    best_val_auc = 0.0


    # Standard ImageNet mean/std for RGB (3 channels)
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # Extend to 6 channels by repeating
    mean_6ch = imagenet_mean * 2
    std_6ch = imagenet_std * 2

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=mean_6ch, std=std_6ch),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=mean_6ch, std=std_6ch),
    ])

    # Datasets (make sure CustomCSVDataset returns 6-channel tensors)
    train_dataset = CustomCSVDataset(csv_file=train_csv_file,transform=train_transform)
    val_dataset   = CustomCSVDataset(csv_file=val_csv_file,transform=val_transform)

    # Plain DataLoaders (no DistributedSampler)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        drop_last=False,  pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        drop_last=False,  pin_memory=True
    )

    # Model (reuse the vit_model you built above)
    # Reuse the AST backbone defined above
    global ast_backbone
    model = ASTWrapper(ast_backbone).to(device)

    num_classes = 6
    counts = torch.zeros(num_classes)
    for _, y in train_loader:
        counts += torch.bincount(y, minlength=num_classes).cpu()

    weights = (counts.sum() / (num_classes * counts)).float()

    # Loss / Optim / Sched
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=0.005,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Histories
    train_loss_history, val_loss_history = [], []

    print("Training started")
    for epoch in range(num_epochs):
        # ---- Train ----
        model.train()
        running_loss = 0.0
        train_correct = 0
        total_train = 0
       # print("Epoch {}/{}".format(epoch + 1, num_epochs))

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            #print("before forward pass")
            logits = model(images)                # [B, 5]
            #print("after forward pass")
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            total_train += labels.size(0)
            train_correct += (preds == labels).sum().item()
            #print("This is the training loop number",i)
            if (i + 1) % 2 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Step [{i+1}/{len(train_loader)}] "
                      f"Train Loss: {loss.item():.4f}")

        average_train_loss = running_loss / len(train_dataset)
        train_accuracy = 100.0 * train_correct / total_train
        train_loss_history.append(average_train_loss)

        # ---- Validate ----
        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        total_val = 0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                logits = model(images)
                loss = criterion(logits, labels)
                running_val_loss += loss.item() * images.size(0)

                probs = torch.softmax(logits, dim=1)  # [B, 5]
                all_probs.append(probs.cpu())
                all_labels.append(labels.cpu())

                preds = logits.argmax(dim=1)
                total_val += labels.size(0)
                val_correct += (preds == labels).sum().item()

        import torch as _t
        probs_np = _t.cat(all_probs, dim=0).numpy()
        labels_np = _t.cat(all_labels, dim=0).numpy()

        # Multiclass AUC (macro OVR)
        try:
            auc = roc_auc_score(labels_np, probs_np, multi_class="ovr", average="macro")
        except ValueError:
            auc = float("nan")  # if some classes are missing in val set

        average_val_loss = running_val_loss / len(val_dataset)
        val_loss_history.append(average_val_loss)
        val_accuracy = 100.0 * val_correct / total_val

        log_message = (f"Epoch [{epoch + 1}/{num_epochs}] "
                       f"Train Loss: {average_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
                       f"Val Loss: {average_val_loss:.4f}, Val AUC (macro-ovr): {auc:.4f}, "
                       f"Val Acc: {val_accuracy:.2f}%")
        print(log_message)
        logging.info(log_message)

        # Early stopping (improve on loss or AUC)
        improved = (average_val_loss < best_val_loss) or (auc > best_val_auc)
        if improved:
            best_val_loss = min(best_val_loss, average_val_loss)
            if not (auc != auc):  # not NaN
                best_val_auc = max(best_val_auc, auc)
            patience_counter = 0
            # torch.save(model.state_dict(), "best_ast_direct_unfreeze3_6to1_6cls.pth") # freezing last 3 layers
            # torch.save(model.state_dict(), "best_ast_direct_6to1_6cls.pth")
            # torch.save(model.state_dict(), "best_ast_direct_train_balanced.pth")
            torch.save(model.state_dict(), "data_old/best_ast_direct_train_balanced_unfrozen_lr_005.pth")
        else:
            patience_counter += 1


        if patience_counter > patience:
            logging.info(f"Early stopping after {patience} epochs with no improvement.")
            break

        scheduler.step()

    # ---- Plot ----
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(val_loss_history,   label='Validation Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training and Validation Loss (CE only)')
    plt.legend()
    plt.savefig('training_validation_loss_ce_train_balanced_unfrozen_lr_005.png', dpi=150, bbox_inches='tight')
    plt.show()

def test(test_csv_file, model_path="best_ast_direct_train_balanced_unfrozen_lr_005.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Same normalization as training
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    mean_6ch = imagenet_mean * 2
    std_6ch = imagenet_std * 2

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=mean_6ch, std=std_6ch),
    ])

    # Load test dataset
    test_dataset = CustomCSVDataset(csv_file=test_csv_file, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, pin_memory=True)

    # Load model and weights
    global ast_backbone
    model = ASTWrapper(ast_backbone).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_probs = []
    all_labels = []
    all_preds = []
    test_correct = 0
    total_test = 0

    with torch.no_grad():
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

    # Concatenate results
    probs_np = torch.cat(all_probs, dim=0).numpy()
    labels_np = torch.cat(all_labels, dim=0).numpy()
    preds_np = torch.cat(all_preds, dim=0).numpy()

    # Compute metrics
    accuracy = 100.0 * test_correct / total_test
    try:
        auc = roc_auc_score(labels_np, probs_np, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Test AUC (macro-ovr): {auc:.4f}")

    # Confusion Matrix
    class_names = [f"Class {i}" for i in range(NUM_LABELS)]  # or use actual names
    cm = confusion_matrix(labels_np, preds_np)

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # normalize per class
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix with Accuracy per Class')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    tick_marks = np.arange(NUM_LABELS)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels([f'Class {i}' for i in range(NUM_LABELS)])
    ax.set_yticklabels([f'Class {i}' for i in range(NUM_LABELS)])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Annotate cells with values
    thresh = cm_norm.max() / 2.
    for i in range(NUM_LABELS):
        for j in range(NUM_LABELS):
            val = cm_norm[i, j]
            txt = f'{val * 100:.1f}%\n({cm[i, j]})'
            ax.text(
                j, i, txt,
                ha='center', va='center',
                color='white' if val > thresh else 'black',
                fontsize=8
            )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    num_epochs = 100
    # train_csv_file = "train.csv"
    # val_csv_file = "valid.csv"
    # test_csv_file = "test.csv"
    train_csv_file = "data_old/train_unbalanced.csv"
    val_csv_file = "data_old/valid_unbalanced.csv"
    test_csv_file = "data_old/test_unbalanced.csv"
    train(num_epochs, train_csv_file, val_csv_file)
    # test(test_csv_file, model_path="best_ast_direct_6to1_6cls.pth")
    # test(test_csv_file, model_path="best_ast_direct_unfreeze3_6to1_6cls.pth")
    # test(test_csv_file, model_path="best_ast_direct_unbalanced.pth")
    # test(test_csv_file, model_path="best_ast_direct_train_balanced.pth")
    # test(test_csv_file, model_path="best_ast_direct_train_balanced_frozen.pth")
    test(test_csv_file, model_path="data_old/best_ast_direct_train_balanced_unfrozen_lr_005.pth")

#Acc -> 83.11%