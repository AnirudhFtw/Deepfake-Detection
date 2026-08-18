import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

from dataset import DeepfakeDataset
from model import DeepfakeResNet

# ===========================
# Configuration
# ===========================
DATASET_PATH = "/content/test_processed"
MODEL_PATH = "/content/drive/MyDrive/Honours/models/rgb_best.pth"

BATCH_SIZE = 32
NUM_WORKERS = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ===========================
# Dataset
# ===========================

dataset = DeepfakeDataset(DATASET_PATH)
dataset.summary()

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

# ===========================
# Model
# ===========================

model = DeepfakeResNet(
    pretrained=False,
    freeze_backbone=False
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(checkpoint["model_state_dict"])

model.to(DEVICE)
model.eval()

# ===========================
# Testing
# ===========================

y_true = []
y_pred = []
y_prob = []

with torch.no_grad():

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)

        _, predictions = torch.max(outputs, dim=1)

        y_true.extend(labels.cpu().numpy())

        y_pred.extend(predictions.cpu().numpy())

        y_prob.extend(probabilities[:, 1].cpu().numpy())

# ===========================
# Metrics
# ===========================

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(y_true, y_pred)

recall = recall_score(y_true, y_pred)

f1 = f1_score(y_true, y_pred)

roc_auc = roc_auc_score(y_true, y_prob)

cm = confusion_matrix(y_true, y_pred)

print("\n========== Test Results ==========")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {roc_auc:.4f}")

print("\nConfusion Matrix")
print(cm)

print("\nClassification Report")
print(classification_report(
    y_true,
    y_pred,
    target_names=["Real", "Fake"]
))