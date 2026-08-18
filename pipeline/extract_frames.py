"""
Stage: extract a fixed number of uniformly-spaced frames per video instead
of every frame. The old extract_frames.py / preprocessing/extract_frames.py
dumped every frame and threw most of them away at face-detection time
anyway — sampling up front is the efficiency fix.
"""
import csv
import os

import cv2
from tqdm import tqdm

from config import OUTPUT_DIR, FRAMES_DIR, FRAMES_PER_VIDEO

MANIFEST_PATH = os.path.join(OUTPUT_DIR, "manifest.csv")


def sample_frame_indices(total_frames, num_samples):
    if total_frames <= num_samples:
        return list(range(total_frames))
    step = total_frames / num_samples
    return [int(i * step) for i in range(num_samples)]


def extract_video(video_path, label, video_name):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return 0

    wanted = set(sample_frame_indices(total_frames, FRAMES_PER_VIDEO))

    out_dir = os.path.join(FRAMES_DIR, label, video_name)
    os.makedirs(out_dir, exist_ok=True)

    saved = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in wanted:
            out_path = os.path.join(out_dir, f"frame_{frame_idx:05d}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1

        frame_idx += 1

    cap.release()
    return saved


def main():
    with open(MANIFEST_PATH) as f:
        rows = list(csv.DictReader(f))

    for row in tqdm(rows, desc="Extracting frames"):
        saved = extract_video(row["video_path"], row["label"], row["video_name"])
        if saved == 0:
            print(f"[WARN] No frames extracted for {row['video_path']}")


if __name__ == "__main__":
    main()
