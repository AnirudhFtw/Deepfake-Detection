import os
import cv2
import numpy as np
from tqdm import tqdm

FACE_DIR = "../test_faces"
PROCESSED_DIR = "../test_processed"

IMG_SIZE = 224


def process_face(img):
    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize to [-1,1]
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5

    return img


def process_all():
    for category in ["real", "fake"]:
        input_path = os.path.join(FACE_DIR, category)
        output_path = os.path.join(PROCESSED_DIR, category)

        os.makedirs(output_path, exist_ok=True)

        videos = sorted(os.listdir(input_path))

        print(f"\nProcessing {category}...")

        for video in tqdm(videos):
            video_input = os.path.join(input_path, video)
            video_output = os.path.join(output_path, video)

            os.makedirs(video_output, exist_ok=True)

            faces = sorted(
                f for f in os.listdir(video_input)
                if f.endswith(".jpg")
            )

            for face_file in faces:

                # if not face_file.endswith(".jpg"):
                #     continue

                img = cv2.imread(os.path.join(video_input, face_file))

                if img is None or img.size == 0:
                    continue

                processed = process_face(img)

                base = os.path.splitext(face_file)[0]

                np.save(
                    os.path.join(video_output, f"{base}.npy"),
                    processed
                )


if __name__ == "__main__":
    process_all()