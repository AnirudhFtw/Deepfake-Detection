import os
import time
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from mtcnn import MTCNN

from model import DeepfakeResNet

# ==========================================================
# Configuration
# ==========================================================

VIDEO_PATH = "../videos/id20_id35_0004.mp4"

MODEL_PATH = "../models/rgb_best.pth"

IMG_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MIN_FACE_SIZE = 50

# ==========================================================
# Load Model
# ==========================================================

print("Loading model...")

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

print(f"Using Device : {DEVICE}")

# ==========================================================
# Initialize Face Detector
# ==========================================================

detector = MTCNN()

# ==========================================================
# Helper Functions
# ==========================================================

def preprocess_face(face):
    """
    Same preprocessing used during training.
    """

    face = cv2.resize(
        face,
        (IMG_SIZE, IMG_SIZE)
    )

    face = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB
    )

    face = face.astype(np.float32)

    face /= 255.0

    face = (face - 0.5) / 0.5

    face = np.transpose(face, (2, 0, 1))

    face = torch.from_numpy(face)

    face = face.unsqueeze(0)

    return face.to(DEVICE)


def extract_context_face(image, box):
    """
    Uses the same adaptive context cropping
    as the training pipeline.
    """

    x, y, w, h = box

    h_img, w_img, _ = image.shape

    x = max(0, x)
    y = max(0, y)

    w = min(w, w_img - x)
    h = min(h, h_img - y)

    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
        return None

    margin = int(
        min(
            150,
            max(
                80,
                0.7 * max(w, h)
            )
        )
    )

    x1 = max(0, x - margin)
    y1 = max(0, y - margin)

    x2 = min(w_img, x + w + margin)
    y2 = min(h_img, y + h + margin)

    face = image[y1:y2, x1:x2]

    if face.size == 0:
        return None

    return face


@torch.no_grad()
def predict_face(face):
    """
    Predict one face.
    Returns:
        prediction
        fake_probability
    """

    tensor = preprocess_face(face)

    output = model(tensor)

    probabilities = torch.softmax(
        output,
        dim=1
    )

    fake_probability = probabilities[0, 1].item()

    prediction = torch.argmax(
        output,
        dim=1
    ).item()

    return prediction, fake_probability


def detect_faces(frame):
    """
    Detect faces using MTCNN.
    """

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    detections = detector.detect_faces(rgb)

    return detections


print("\nModel Loaded Successfully.")
print("Ready for inference.\n")

# ==========================================================
# Video Processing
# ==========================================================

def process_video(video_path):

    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Unable to open video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing {total_frames} frames...\n")

    fake_probabilities = []

    fake_frames = 0
    real_frames = 0
    faces_detected = 0
    processed_frames = 0

    start_time = time.time()

    progress = tqdm(total=total_frames)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        processed_frames += 1

        detections = detect_faces(frame)

        if len(detections) == 0:

            progress.update(1)

            continue

        for detection in detections:

            face = extract_context_face(
                frame,
                detection["box"]
            )

            if face is None:
                continue

            prediction, fake_probability = predict_face(face)

            fake_probabilities.append(fake_probability)

            faces_detected += 1

            if prediction == 1:
                fake_frames += 1
            else:
                real_frames += 1

        progress.update(1)

    progress.close()

    cap.release()

    end_time = time.time()

    inference_time = end_time - start_time

    # ============================================
    # Final Prediction
    # ============================================

    if len(fake_probabilities) == 0:

        print("\nNo faces detected in the video.")

        return

    average_fake_probability = float(np.mean(fake_probabilities))

    prediction = (
        "FAKE"
        if average_fake_probability >= 0.5
        else "REAL"
    )

    confidence = (
        average_fake_probability
        if prediction == "FAKE"
        else 1 - average_fake_probability
    )

    # ============================================
    # Results
    # ============================================

    print("\n========================================")

    print("Video Analysis Complete")

    print("========================================")

    print(f"Frames Processed      : {processed_frames}")

    print(f"Faces Detected        : {faces_detected}")

    print(f"Real Predictions      : {real_frames}")

    print(f"Fake Predictions      : {fake_frames}")

    print()

    print(
        f"Average Fake Probability : "
        f"{average_fake_probability * 100:.2f}%"
    )

    print(
        f"Prediction Confidence    : "
        f"{confidence * 100:.2f}%"
    )

    print()

    print(f"FINAL PREDICTION : {prediction}")

    print()

    print(
        f"Inference Time : "
        f"{inference_time:.2f} seconds"
    )

    print("========================================")

    # ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 60)
    print("        Deepfake Video Detection using ResNet18")
    print("=" * 60)

    if not os.path.exists(VIDEO_PATH):
        print(f"\nVideo not found:\n{VIDEO_PATH}")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"\nModel not found:\n{MODEL_PATH}")
        return

    process_video(VIDEO_PATH)


if __name__ == "__main__":
    main()