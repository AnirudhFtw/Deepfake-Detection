"""
Evaluates a trained frequency-branch checkpoint on the shared held-out
test split (processed/splits/test.json — see pipeline/split_dataset.py).
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "pipeline"))

import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

from config import FREQUENCY_DIR, SPLITS_DIR
from splits import load_split_keys, indices_for_keys
from dataset import DeepfakeDataset
from model import DeepfakeFrequencyCNN

MODEL_PATH = "../checkpoints/frequency_best.pth"
BATCH_SIZE = 32
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = DeepfakeDataset(FREQUENCY_DIR)
dataset.summary()

test_keys = load_split_keys(SPLITS_DIR, "test")
test_dataset = Subset(dataset, indices_for_keys(dataset.get_video_indices(), test_keys))

loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available()
)

model = DeepfakeFrequencyCNN()
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

y_true, y_pred, y_prob = [], [], []

with torch.no_grad():
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        _, predictions = torch.max(outputs, dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predictions.cpu().numpy())
        y_prob.extend(probabilities[:, 1].cpu().numpy())

print("\n========== Test Results (frequency) ==========")
print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred):.4f}")
print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
print(f"F1 Score : {f1_score(y_true, y_pred):.4f}")
print(f"ROC AUC  : {roc_auc_score(y_true, y_prob):.4f}")

print("\nConfusion Matrix")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report")
print(classification_report(y_true, y_pred, target_names=["Real", "Fake"]))
