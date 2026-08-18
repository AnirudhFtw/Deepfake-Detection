"""
Stage 1: sample a bounded set of source videos per class and write a
manifest that every later stage reads from. This is the single place video
selection happens now — replaces `initialdatasetsplit.py` and
`preprocessing/split_dataset.py`, which duplicated this logic.
"""
import csv
import os
import random

from config import RAW_DATASET_DIR, CATEGORY_MAP, NUM_VIDEOS_PER_CLASS, OUTPUT_DIR, SEED

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
MANIFEST_PATH = os.path.join(OUTPUT_DIR, "manifest.csv")


def list_videos(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(VIDEO_EXTENSIONS)
    )


def main():
    random.seed(SEED)

    rows_by_label = {"real": [], "fake": []}

    for category, label in CATEGORY_MAP.items():
        category_path = os.path.join(RAW_DATASET_DIR, category)
        videos = list_videos(category_path)

        if not videos:
            print(f"[WARN] No videos found in {category_path}, skipping.")
            continue

        for video in videos:
            rows_by_label[label].append({
                "category": category,
                "label": label,
                "video_name": os.path.splitext(video)[0],
                "video_path": os.path.join(category_path, video),
            })

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["category", "label", "video_name", "video_path"]
        )
        writer.writeheader()

        for label, rows in rows_by_label.items():
            random.shuffle(rows)
            selected = rows[:NUM_VIDEOS_PER_CLASS]

            if len(rows) < NUM_VIDEOS_PER_CLASS:
                print(
                    f"[WARN] Only {len(rows)} '{label}' videos available, "
                    f"wanted {NUM_VIDEOS_PER_CLASS}."
                )

            writer.writerows(selected)
            print(f"Selected {len(selected)} '{label}' videos.")

    print(f"\nManifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
