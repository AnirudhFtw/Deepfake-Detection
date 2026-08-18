"""
Stage: build the frequency-domain representation of each face crop.

Fixes two problems in the old apply_dct.py:
  1. Whole-image 2D DCT concentrates low frequencies in one corner and
     destroys the shift-invariance/local-structure a CNN backbone relies
     on. This uses JPEG-style block-wise DCT instead, which keeps
     frequency information spatially localized (same idea as F3-Net's
     local frequency statistics).
  2. Raw DCT coefficients have a huge, unnormalized, sign-mixed dynamic
     range. This applies log-magnitude scaling and rescales to [-1, 1] so
     the values are actually learnable, instead of being saved raw.
"""
import os

import cv2
import numpy as np
from tqdm import tqdm

from config import FACES_DIR, FREQUENCY_DIR, IMG_SIZE, DCT_BLOCK_SIZE


def block_dct(channel, block_size):
    h, w = channel.shape
    out = np.zeros_like(channel, dtype=np.float32)

    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            block = channel[y:y + block_size, x:x + block_size]
            out[y:y + block_size, x:x + block_size] = cv2.dct(block)

    return out


def process_face(image):
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

    channels = []
    for c in range(3):
        dct = block_dct(image[:, :, c], DCT_BLOCK_SIZE)

        magnitude = np.log1p(np.abs(dct))
        magnitude = magnitude / (magnitude.max() + 1e-8)
        magnitude = (magnitude - 0.5) / 0.5

        channels.append(magnitude)

    return np.stack(channels, axis=-1)


def main():
    for label in ("real", "fake"):
        label_dir = os.path.join(FACES_DIR, label)
        if not os.path.isdir(label_dir):
            continue

        videos = sorted(os.listdir(label_dir))
        for video in tqdm(videos, desc=f"Frequency ({label})"):
            video_dir = os.path.join(label_dir, video)
            out_dir = os.path.join(FREQUENCY_DIR, label, video)
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
