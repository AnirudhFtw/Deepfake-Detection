"""
Trains the spatial branch on processed/spatial/ (see pipeline/). Uses the
shared video-level split from pipeline/split_dataset.py instead of
re-deriving a random split, so results are comparable with the frequency
branch and, later, fusion.
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "pipeline"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config import SPATIAL_DIR, SPLITS_DIR
from splits import load_split_keys, indices_for_keys
from dataset import DeepfakeDataset
from utils import seed_everything, AverageMeter, calculate_accuracy, save_checkpoint
from model import DeepfakeResNet

# ===========================
# Configuration
# ===========================

BEST_SAVE_PATH = "../checkpoints/spatial_best.pth"
FINAL_SAVE_PATH = "../checkpoints/spatial_final.pth"

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed_everything(42)

# ===========================
# Dataset
# ===========================

dataset = DeepfakeDataset(SPATIAL_DIR)
dataset.summary()

video_indices = dataset.get_video_indices()

train_keys = load_split_keys(SPLITS_DIR, "train")
val_keys = load_split_keys(SPLITS_DIR, "val")

train_dataset = Subset(dataset, indices_for_keys(video_indices, train_keys))
val_dataset = Subset(dataset, indices_for_keys(video_indices, val_keys))

print(f"\nTraining Frames  : {len(train_dataset)}")
print(f"Validation Frames: {len(val_dataset)}")

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)

# ===========================
# Model / loss / optimizer
# ===========================

model = DeepfakeResNet(pretrained=True, freeze_backbone=False).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

best_accuracy = 0.0


def run_epoch(loader, train):
    model.train() if train else model.eval()

    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    progress = tqdm(loader)

    for images, labels in progress:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        with torch.set_grad_enabled(train):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        acc = calculate_accuracy(outputs, labels)
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc, images.size(0))

        mode = "Train" if train else "Val"
        progress.set_description(f"{mode} Loss {loss_meter.avg:.4f} | Acc {acc_meter.avg:.4f}")

    return loss_meter.avg, acc_meter.avg


# ===========================
# Training loop
# ===========================

for epoch in range(EPOCHS):
    print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")

    train_loss, train_acc = run_epoch(train_loader, train=True)

    with torch.no_grad():
        val_loss, val_acc = run_epoch(val_loader, train=False)

    scheduler.step(val_acc)

    print(f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.4f}")
    print(f"Val Loss   {val_loss:.4f} | Val Acc   {val_acc:.4f}")

    if val_acc > best_accuracy:
        best_accuracy = val_acc
        save_checkpoint(model, optimizer, epoch, best_accuracy, BEST_SAVE_PATH)
        print(f"Best model saved! Val Acc: {best_accuracy:.4f}")

save_checkpoint(model, optimizer, EPOCHS - 1, best_accuracy, FINAL_SAVE_PATH)
print(f"\nTraining complete. Best Val Acc: {best_accuracy:.4f}")
