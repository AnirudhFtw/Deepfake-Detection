"""
Evaluates a trained spatial-branch checkpoint on the shared held-out test
split (processed/splits/test.json — see pipeline/split_dataset.py).
"""
import argparse
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

from config import SPATIAL_DIR, SPLITS_DIR
from splits import load_split_keys, indices_for_keys
from common.dataset import DeepfakeDataset
from model import DeepfakeResNet

MODEL_PATH = "../checkpoints/spatial_best.pth"
BATCH_SIZE = 32
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--all-videos", action="store_true",
    help="Evaluate every processed video (use for held-out Celeb-DF).",
)
args = parser.parse_args()

dataset = DeepfakeDataset(SPATIAL_DIR)
dataset.summary()

if args.all_videos:
    test_indices = list(range(len(dataset)))
else:
    test_keys = load_split_keys(SPLITS_DIR, "test")
    test_indices = indices_for_keys(dataset.get_video_indices(), test_keys)
test_dataset = Subset(dataset, test_indices)

loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available()
)

model = DeepfakeResNet(pretrained=False, freeze_backbone=False)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

video_scores, video_labels = {}, {}
offset = 0

with torch.no_grad():
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        for index, label, probability in zip(
            test_indices[offset:offset + len(labels)],
            labels.cpu().numpy(),
            probabilities[:, 1].cpu().numpy(),
        ):
            video = dataset.samples[index]["video"]
            video_scores.setdefault(video, []).append(probability)
            video_labels[video] = label
        offset += len(labels)

y_true = [video_labels[video] for video in video_scores]
y_prob = [sum(scores) / len(scores) for scores in video_scores.values()]
y_pred = [probability >= 0.5 for probability in y_prob]

print("\n========== Video-Level Test Results (spatial) ==========")
print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_true, y_pred, zero_division=0):.4f}")
print(f"F1 Score : {f1_score(y_true, y_pred, zero_division=0):.4f}")
print(f"ROC AUC  : {roc_auc_score(y_true, y_prob):.4f}")

print("\nConfusion Matrix")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report")
print(classification_report(y_true, y_pred, target_names=["Real", "Fake"], zero_division=0))
