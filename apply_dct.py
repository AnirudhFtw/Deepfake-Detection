import os
import numpy as np
import cv2
from tqdm import tqdm

INPUT_DIR = "../processed"
OUTPUT_DIR = "../processed_dct"


def apply_dct_channel(channel):
    return cv2.dct(channel)


def apply_dct(image):
    dct_channels = []

    for i in range(3):
        channel = image[:, :, i]
        dct_channel = apply_dct_channel(channel)
        dct_channels.append(dct_channel)

    return np.stack(dct_channels, axis=-1)


def process_all():
    for category in ["real", "fake"]:
        input_path = os.path.join(INPUT_DIR, category)
        output_path = os.path.join(OUTPUT_DIR, category)

        os.makedirs(output_path, exist_ok=True)

        videos = sorted(os.listdir(input_path))

        print(f"\nProcessing {category} (DCT)...")

        for video in tqdm(videos):
            video_input = os.path.join(input_path, video)
            video_output = os.path.join(output_path, video)

            os.makedirs(video_output, exist_ok=True)

            # Skip if already processed
            if len(os.listdir(video_output)) > 0:
                continue

            files = sorted(os.listdir(video_input))

            for file in files:
                if not file.endswith(".npy"):
                    continue

                file_path = os.path.join(video_input, file)
                img = np.load(file_path).astype(np.float32)

                # Convert [-1,1] → [0,1]
                img = (img + 1) / 2

                # Apply DCT
                dct_img = apply_dct(img)

                # Log scaling
                dct_img = np.log1p(np.abs(dct_img))

                # Normalize to [0,1]
                dct_img /= (np.max(dct_img) + 1e-8)

                save_path = os.path.join(video_output, file)
                np.save(save_path, dct_img.astype(np.float32))


if __name__ == "__main__":
    process_all()