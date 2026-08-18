import os
import cv2
from tqdm import tqdm

DATASET_DIR = "../test_dataset"
FRAME_DIR = "../test_frames"

CATEGORY_MAP = {
    "Celeb-real": "real",
    "Celeb-synthesis": "fake"
}

REAL_VIDEOS = 60
FAKE_VIDEOS = 120


def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_folder = os.path.join(output_folder, video_name)

    os.makedirs(video_output_folder, exist_ok=True)

    frame_number = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_path = os.path.join(
            video_output_folder,
            f"frame_{frame_number:05d}.jpg"
        )

        cv2.imwrite(frame_path, frame)

        frame_number += 1

    cap.release()


def process_dataset():
    for category in CATEGORY_MAP:

        category_path = os.path.join(DATASET_DIR, category)

        if not os.path.exists(category_path):
            continue

        label = CATEGORY_MAP[category]

        output_category_path = os.path.join(FRAME_DIR, label)

        os.makedirs(output_category_path, exist_ok=True)

        videos = sorted([
            f for f in os.listdir(category_path)
            if f.lower().endswith((".mp4", ".avi", ".mov"))
        ])

        if category == "Celeb-real":
            videos = videos[:REAL_VIDEOS]
        elif category == "Celeb-synthesis":
            videos = videos[:FAKE_VIDEOS]

        print(f"\nProcessing {category}: {len(videos)} videos")

        for video in tqdm(videos):
            extract_frames(
                os.path.join(category_path, video),
                output_category_path
            )


if __name__ == "__main__":
    process_dataset()