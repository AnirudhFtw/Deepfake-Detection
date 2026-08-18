import os
import cv2
from mtcnn import MTCNN
from tqdm import tqdm

FRAME_DIR = "../test_frames"
FACE_DIR = "../test_faces"

detector = MTCNN()

MIN_FACE_SIZE = 50


def extract_faces_from_frames(input_folder, output_folder):
    videos = sorted(os.listdir(input_folder))

    for video in tqdm(videos):
        video_path = os.path.join(input_folder, video)
        output_video_folder = os.path.join(output_folder, video)

        # Skip if already processed
        if os.path.exists(output_video_folder) and len(os.listdir(output_video_folder)) > 0:
            continue

        os.makedirs(output_video_folder, exist_ok=True)

        frames = sorted([
            f for f in os.listdir(video_path)
            if f.lower().endswith(".jpg")
        ])

        for frame in frames:
            frame_path = os.path.join(video_path, frame)

            img = cv2.imread(frame_path)

            if img is None:
                print(f"Could not read: {frame_path}")
                continue

            # Skip invalid images
            if img.shape[0] < 24 or img.shape[1] < 24:
                print(f"Image too small: {frame_path}")
                continue

            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            try:
                results = detector.detect_faces(rgb_img)
            except Exception as e:
                print(f"\nMTCNN failed on:")
                print(frame_path)
                print(e)
                continue

            h_img, w_img, _ = img.shape

            for i, res in enumerate(results):
                x, y, w, h = res["box"]

                # Fix negative coordinates
                x = max(0, x)
                y = max(0, y)

                # Clamp width and height
                w = min(w, w_img - x)
                h = min(h, h_img - y)

                # Skip tiny detections
                if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                    continue

                # Adaptive context margin
                margin = int(min(150, max(80, 0.7 * max(w, h))))

                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(w_img, x + w + margin)
                y2 = min(h_img, y + h + margin)

                face = img[y1:y2, x1:x2]

                if face.size == 0:
                    continue

                face_filename = os.path.join(
                    output_video_folder,
                    f"{os.path.splitext(frame)[0]}_{i}.jpg"
                )

                cv2.imwrite(face_filename, face)


def process_all():
    for category in ["real", "fake"]:
        input_path = os.path.join(FRAME_DIR, category)
        output_path = os.path.join(FACE_DIR, category)

        os.makedirs(output_path, exist_ok=True)

        print(f"\nProcessing {category} faces...")
        extract_faces_from_frames(input_path, output_path)


if __name__ == "__main__":
    process_all()