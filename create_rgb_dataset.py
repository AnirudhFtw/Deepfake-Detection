import os
import numpy as np
import torch
from tqdm import tqdm

# ============================================
# Configuration
# ============================================

INPUT_DIR = "../processed"
OUTPUT_DIR = "../processed_pt"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "rgb_dataset.pt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================
# Create Dataset
# ============================================

samples = []

for category in ["real", "fake"]:

    label = 0 if category == "real" else 1

    category_path = os.path.join(INPUT_DIR, category)

    videos = sorted(os.listdir(category_path))

    print(f"\nProcessing {category}...")

    for video in tqdm(videos):

        video_path = os.path.join(category_path, video)

        files = sorted(os.listdir(video_path))

        for file in files:

            if not file.endswith(".npy"):
                continue

            file_path = os.path.join(video_path, file)

            image = np.load(file_path).astype(np.float32)

            image = torch.from_numpy(image)

            image = image.permute(2, 0, 1)

            samples.append(
                (
                    image,
                    label,
                    video
                )
            )

print("\nSaving dataset...")

torch.save(samples, OUTPUT_FILE)

print("\n========================================")
print("Dataset created successfully!")
print(f"Total Samples : {len(samples)}")
print(f"Saved to      : {OUTPUT_FILE}")
print("========================================")