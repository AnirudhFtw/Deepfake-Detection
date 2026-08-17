"""
preprocess.py

Deepfake detection preprocessing pipeline.

This version reads ALREADY-EXTRACTED FRAMES (images), not raw video files.

Pipeline:
Extracted frames -> read frame -> every N frames: detect face, else reuse last bbox
      -> expand bbox -> crop -> resize -> save RGB frame -> log to metadata.csv

Expected folder structure (label inferred from category folder name):
    dataset/
        Celeb-real/
            video_001/
                frame_0000.jpg
                frame_0001.jpg
                ...
            video_002/
                ...
        Celeb-synthesis/
            ...
        YouTube-real/
            ...

Each subfolder under a category is treated as one video's extracted frames,
and the frames inside it are processed in sorted (filename) order.
If your extracted-frames layout is different, adjust `list_frame_groups()`.

Usage:
    python preprocess.py
    python preprocess.py --input_dir dataset --output_dir processed \
        --detect_interval 10 --margin 0.3 --img_size 224

Dependencies:
    pip install opencv-python numpy tqdm
    # Preferred face detector (recommended):
    pip install facenet-pytorch torch
    # If facenet-pytorch is not installed, the script falls back to OpenCV's
    # DNN face detector, which requires these two files in the working dir:
    #   deploy.prototxt
    #   res10_300x300_ssd_iter_140000.caffemodel
"""

import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# EDIT THESE TO MATCH YOUR PROJECT — used as defaults, can be overridden by
# passing --input_dir / --output_dir on the command line.
# ---------------------------------------------------------------------------
INPUT_DIR = "dataset"        # root folder containing Celeb-real, Celeb-synthesis, YouTube-real
OUTPUT_DIR = "processed"     # where cropped RGB faces + metadata.csv will be saved

DETECT_INTERVAL = 10          # run face detector every N frames, reuse bbox otherwise
MARGIN = 0.3                  # bbox expansion margin (fraction of face size)
IMG_SIZE = 224                 # output crop size (IMG_SIZE x IMG_SIZE)
FRAME_EXTS = (".jpg", ".jpeg", ".png")  # extensions to look for when reading extracted frames

# ---------------------------------------------------------------------------
# Face detector setup
# ---------------------------------------------------------------------------
try:
    from facenet_pytorch import MTCNN
    import torch

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    _mtcnn_detector = MTCNN(keep_all=False, device=DEVICE)
    USE_MTCNN = True
    print(f"[INFO] Using MTCNN face detector on {DEVICE}")
except ImportError:
    USE_MTCNN = False
    PROTO = "deploy.prototxt"
    MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
    if os.path.exists(PROTO) and os.path.exists(MODEL):
        _cv_net = cv2.dnn.readNetFromCaffe(PROTO, MODEL)
        print("[INFO] facenet-pytorch not found, using OpenCV DNN face detector")
    else:
        _cv_net = None
        print("[WARN] No face detector available! Install facenet-pytorch, "
              "or place deploy.prototxt + res10_300x300_ssd_iter_140000.caffemodel "
              "in the working directory.")

LABEL_MAP = {
    "Celeb-real": "real",
    "YouTube-real": "real",
    "Celeb-synthesis": "fake",
}


# ---------------------------------------------------------------------------
# Face detection
# ---------------------------------------------------------------------------
def detect_face_mtcnn(frame_rgb):
    boxes, probs = _mtcnn_detector.detect(frame_rgb)
    if boxes is None:
        return None
    idx = int(np.argmax(probs))
    return boxes[idx]  # x1, y1, x2, y2


def detect_face_opencv(frame_bgr, conf_threshold=0.6):
    if _cv_net is None:
        return None
    h, w = frame_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame_bgr, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0),
    )
    _cv_net.setInput(blob)
    detections = _cv_net.forward()

    best_box, best_conf = None, 0.0
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > conf_threshold and conf > best_conf:
            best_box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            best_conf = conf
    return best_box


def detect_face(frame_bgr):
    if USE_MTCNN:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return detect_face_mtcnn(frame_rgb)
    return detect_face_opencv(frame_bgr)


# ---------------------------------------------------------------------------
# Bbox expansion (square, so resize doesn't distort the face)
# ---------------------------------------------------------------------------
def expand_bbox(box, frame_shape, margin=0.3):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    cx, cy = x1 + bw / 2, y1 + bh / 2

    side = max(bw, bh) * (1 + margin)
    nx1 = max(0, cx - side / 2)
    ny1 = max(0, cy - side / 2)
    nx2 = min(w, cx + side / 2)
    ny2 = min(h, cy + side / 2)
    return int(nx1), int(ny1), int(nx2), int(ny2)


# ---------------------------------------------------------------------------
# Discover per-video frame folders
# ---------------------------------------------------------------------------
def list_frame_groups(category_path):
    """
    Returns a list of (video_name, [sorted frame file paths]) for every
    subfolder inside a category folder (e.g. dataset/Celeb-real/*).
    """
    groups = []
    for video_dir in sorted(p for p in category_path.iterdir() if p.is_dir()):
        frame_files = sorted(
            p for p in video_dir.iterdir()
            if p.suffix.lower() in FRAME_EXTS
        )
        if frame_files:
            groups.append((video_dir.name, frame_files))
    return groups


# ---------------------------------------------------------------------------
# Per-video processing (reads extracted frame images, not a video file)
# ---------------------------------------------------------------------------
def process_frame_folder(video_name, frame_files, label, output_dir, writer,
                          detect_interval=10, margin=0.3, img_size=224):
    out_subdir = Path(output_dir) / label / video_name
    out_subdir.mkdir(parents=True, exist_ok=True)

    last_box = None
    saved_count = 0

    for frame_idx, frame_path in enumerate(frame_files):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"[WARN] Could not read {frame_path}")
            continue

        is_detect_frame = (frame_idx % detect_interval == 0) or (last_box is None)
        if is_detect_frame:
            box = detect_face(frame)
            if box is not None:
                last_box = box
            else:
                # detection failed on a detect-frame: fall back to last known bbox
                box = last_box
        else:
            box = last_box

        if box is not None:
            x1, y1, x2, y2 = expand_bbox(box, frame.shape, margin=margin)
            face = frame[y1:y2, x1:x2]
            if face.size != 0:
                face_resized = cv2.resize(face, (img_size, img_size))
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

                out_name = f"{video_name}_frame{frame_idx:05d}.png"
                out_path = out_subdir / out_name
                # imwrite expects BGR, so convert back before saving
                cv2.imwrite(str(out_path), cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR))

                writer.writerow({
                    "video": video_name,
                    "frame": frame_idx,
                    "label": label,
                    "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
                    "output_path": str(out_path),
                })
                saved_count += 1

    return saved_count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Deepfake detection preprocessing pipeline")
    parser.add_argument("--input_dir", default=INPUT_DIR, help="Root dataset dir (contains extracted frame folders)")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="Output dir for cropped faces")
    parser.add_argument("--detect_interval", type=int, default=DETECT_INTERVAL, help="Run detector every N frames")
    parser.add_argument("--margin", type=float, default=MARGIN, help="Bbox expansion margin (fraction)")
    parser.add_argument("--img_size", type=int, default=IMG_SIZE, help="Output image size (square)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "metadata.csv"
    fieldnames = ["video", "frame", "label", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "output_path"]

    with open(metadata_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for folder_name, label in LABEL_MAP.items():
            category_path = input_dir / folder_name
            if not category_path.exists():
                print(f"[WARN] {category_path} not found, skipping.")
                continue

            frame_groups = list_frame_groups(category_path)
            print(f"\nProcessing {folder_name} ({label}) - {len(frame_groups)} videos (as extracted frames)")

            total_saved = 0
            for video_name, frame_files in tqdm(frame_groups, desc=folder_name):
                total_saved += process_frame_folder(
                    video_name, frame_files, label, output_dir, writer,
                    detect_interval=args.detect_interval,
                    margin=args.margin,
                    img_size=args.img_size,
                )
            print(f"  -> saved {total_saved} face crops")

    print(f"\nDone. Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()