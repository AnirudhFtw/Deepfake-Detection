import os
import shutil
import random

DATASET_DIR = "../dataset"

FAKE_FOLDERS = [
    "Deepfakes",
    "Face2Face",
    "FaceSwap",
    "FaceShifter",
    "NeuralTextures"
]

OUTPUT_DIR = "../dataset_small/fake"

VIDEOS_PER_FOLDER = 24  # 120 total / 5 folders


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_count = 0

    for folder in FAKE_FOLDERS:
        folder_path = os.path.join(DATASET_DIR, folder)

        if not os.path.exists(folder_path):
            print(f"Skipping {folder} (not found)")
            continue

        videos = [
            f for f in os.listdir(folder_path)
            if f.endswith((".mp4", ".avi", ".mov"))
        ]

        random.shuffle(videos)

        selected = videos[:VIDEOS_PER_FOLDER]

        print(f"{folder}: selecting {len(selected)} videos")

        for i, video in enumerate(selected):
            src = os.path.join(folder_path, video)

            # Unique naming
            new_name = f"{folder.lower()}_{i}.mp4"
            dst = os.path.join(OUTPUT_DIR, new_name)

            shutil.copy2(src, dst)

            total_count += 1

    print(f"\nTotal fake videos created: {total_count}")


if __name__ == "__main__":
    main()