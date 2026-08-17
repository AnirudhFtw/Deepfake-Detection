import random
import torch
import torch.nn as nn

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from dataset import DeepfakeDataset
from model import DeepfakeResNet
from utils import (
    seed_everything,
    AverageMeter,
    calculate_accuracy,
    save_checkpoint
)


# ===========================
# Configuration
# ===========================

DATASET_PATH = "/content/processed_dct"

BEST_SAVE_PATH = "/content/drive/MyDrive/Honours/models/dct_best.pth"
FINAL_SAVE_PATH = "/content/drive/MyDrive/Honours/models/dct_final.pth"

BATCH_SIZE = 32
EPOCHS = 30

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

seed_everything(42)


# ===========================
# Load Dataset
# ===========================

dataset = DeepfakeDataset(DATASET_PATH)

dataset.summary()

video_indices = dataset.get_video_indices()

video_keys = list(video_indices.keys())

real_videos = [
    k for k in video_keys
    if k[0] == "real"
]

fake_videos = [
    k for k in video_keys
    if k[0] == "fake"
]


# ===========================
# Video-Level Train/Val Split
# ===========================

train_real, val_real = train_test_split(
    real_videos,
    test_size=0.2,
    random_state=42
)

train_fake, val_fake = train_test_split(
    fake_videos,
    test_size=0.2,
    random_state=42
)

train_videos = train_real + train_fake
val_videos = val_real + val_fake


# ===========================
# Convert Videos → Frame Indices
# ===========================

train_indices = []

for key in train_videos:
    train_indices.extend(video_indices[key])


val_indices = []

for key in val_videos:
    val_indices.extend(video_indices[key])


train_dataset = Subset(
    dataset,
    train_indices
)

val_dataset = Subset(
    dataset,
    val_indices
)

print(f"\nTraining Frames : {len(train_dataset)}")
print(f"Validation Frames : {len(val_dataset)}")


# ===========================
# DataLoaders
# ===========================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)


# ===========================
# Model
# ===========================

print("Reached model creation")

model = DeepfakeResNet(
    pretrained=True,
    freeze_backbone=False
)

print("Model created")

model = model.to(DEVICE)


# ===========================
# Loss
# ===========================

criterion = nn.CrossEntropyLoss()


# ===========================
# Optimizer
# ===========================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ===========================
# Learning Rate Scheduler
# ===========================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=3
)


# ===========================
# Best Model Tracking
# ===========================

best_accuracy = 0.0
best_epoch = 0


# ===========================
# Training Function
# ===========================

def train_one_epoch():

    model.train()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    progress = tqdm(train_loader)

    for images, labels in progress:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        acc = calculate_accuracy(
            outputs,
            labels
        )

        loss_meter.update(
            loss.item(),
            images.size(0)
        )

        acc_meter.update(
            acc,
            images.size(0)
        )

        progress.set_description(
            f"Train Loss {loss_meter.avg:.4f} | "
            f"Acc {acc_meter.avg:.4f}"
        )

    return (
        loss_meter.avg,
        acc_meter.avg
    )


# ===========================
# Validation Function
# ===========================

@torch.no_grad()
def validate():

    model.eval()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    progress = tqdm(val_loader)

    for images, labels in progress:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        acc = calculate_accuracy(
            outputs,
            labels
        )

        loss_meter.update(
            loss.item(),
            images.size(0)
        )

        acc_meter.update(
            acc,
            images.size(0)
        )

        progress.set_description(
            f"Val Loss {loss_meter.avg:.4f} | "
            f"Acc {acc_meter.avg:.4f}"
        )

    return (
        loss_meter.avg,
        acc_meter.avg
    )


# ===========================
# Training Loop
# ===========================

print("\nStarting Training...\n")
print("About to start training loop")


for epoch in range(EPOCHS):

    print(
        f"\nEpoch [{epoch + 1}/{EPOCHS}]"
    )

    # -----------------------
    # Training
    # -----------------------

    train_loss, train_acc = train_one_epoch()

    # -----------------------
    # Validation
    # -----------------------

    val_loss, val_acc = validate()

    # -----------------------
    # Scheduler
    # -----------------------

    scheduler.step(val_acc)

    # -----------------------
    # Print Results
    # -----------------------

    print("\n-------------------------------")

    print(
        f"Train Loss : {train_loss:.4f}"
    )

    print(
        f"Train Acc  : {train_acc:.4f}"
    )

    print(
        f"Val Loss   : {val_loss:.4f}"
    )

    print(
        f"Val Acc    : {val_acc:.4f}"
    )

    print("-------------------------------")


    # =======================
    # Save Best Model
    # =======================

    if val_acc > best_accuracy:

        best_accuracy = val_acc

        best_epoch = epoch + 1

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_acc=best_accuracy,
            path=BEST_SAVE_PATH
        )

        print(
            f"\nBest model saved!"
            f" Validation Accuracy: "
            f"{best_accuracy:.4f}"
        )


# ===========================
# Save Final Model
# ===========================

save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=EPOCHS - 1,
    best_acc=best_accuracy,
    path=FINAL_SAVE_PATH
)


# ===========================
# Training Complete
# ===========================

print("\n===================================")

print("Training Complete!")

print(
    f"Best Validation Accuracy: "
    f"{best_accuracy:.4f}"
)

print(
    f"Best Epoch: {best_epoch}"
)

print(
    f"\nBest Model:"
    f"\n{BEST_SAVE_PATH}"
)

print(
    f"\nFinal Model:"
    f"\n{FINAL_SAVE_PATH}"
)

print("===================================")