"""
Stage: normalize cropped faces into the spatial (RGB) domain tensors the
spatial branch trains on — resize -> BGR->RGB -> scale to [-1, 1], the same
normalization predict_video.py applies at inference time.
"""
import os

import cv2
import numpy as np
from tqdm import tqdm

from config import FACES_DIR, SPATIAL_DIR, IMG_SIZE


def process_face(image):
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = (image - 0.5) / 0.5
    return image


def main():
    for label in ("real", "fake"):
        label_dir = os.path.join(FACES_DIR, label)
        if not os.path.isdir(label_dir):
            continue

        videos = sorted(os.listdir(label_dir))
        for video in tqdm(videos, desc=f"Spatial ({label})"):
            video_dir = os.path.join(label_dir, video)
            out_dir = os.path.join(SPATIAL_DIR, label, video)
            os.makedirs(out_dir, exist_ok=True)

            for frame_file in sorted(os.listdir(video_dir)):
                image = cv2.imread(os.path.join(video_dir, frame_file))
                if image is None:
                    continue

                processed = process_face(image)
                out_name = os.path.splitext(frame_file)[0] + ".npy"
                np.save(os.path.join(out_dir, out_name), processed)


if __name__ == "__main__":
    main()
