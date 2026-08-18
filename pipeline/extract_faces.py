"""
Stage: detect and crop faces from the extracted frames, re-running MTCNN
only every FACE_DETECT_EVERY_N frames and reusing the last bounding box in
between. The root-level extract_faces.py ran MTCNN on every single frame
(the main preprocessing bottleneck); this applies the bbox-reuse idea
already prototyped in preprocessing/detect_faces.py consistently, plus the
adaptive context margin from predict_video.py.
"""
import os

import cv2
from mtcnn import MTCNN
from tqdm import tqdm

from config import FRAMES_DIR, FACES_DIR, FACE_DETECT_EVERY_N, MIN_FACE_SIZE, IMG_SIZE

detector = MTCNN()


def extract_context_face(image, box):
    x, y, w, h = box
    h_img, w_img = image.shape[:2]

    x, y = max(0, x), max(0, y)
    w = min(w, w_img - x)
    h = min(h, h_img - y)

    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
        return None

    margin = int(min(150, max(80, 0.7 * max(w, h))))

    x1, y1 = max(0, x - margin), max(0, y - margin)
    x2, y2 = min(w_img, x + w + margin), min(h_img, y + h + margin)

    face = image[y1:y2, x1:x2]
    return face if face.size else None


def detect_box(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(rgb)
    if not detections:
        return None
    return max(detections, key=lambda d: d["confidence"])["box"]


def process_video(video_dir, out_dir):
    frame_files = sorted(
        f for f in os.listdir(video_dir) if f.lower().endswith(".jpg")
    )

    last_box = None
    saved = 0

    for i, frame_file in enumerate(frame_files):
        frame = cv2.imread(os.path.join(video_dir, frame_file))
        if frame is None:
            continue

        if i % FACE_DETECT_EVERY_N == 0 or last_box is None:
            box = detect_box(frame)
            if box is not None:
                last_box = box
        else:
            box = last_box

        if box is None:
            continue

        face = extract_context_face(frame, box)
        if face is None:
            continue

        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, frame_file), face)
        saved += 1

    return saved


def main():
    for label in ("real", "fake"):
        label_dir = os.path.join(FRAMES_DIR, label)
        if not os.path.isdir(label_dir):
            continue

        videos = sorted(os.listdir(label_dir))
        for video in tqdm(videos, desc=f"Faces ({label})"):
            video_dir = os.path.join(label_dir, video)
            out_dir = os.path.join(FACES_DIR, label, video)

            if os.path.exists(out_dir) and os.listdir(out_dir):
                continue

            saved = process_video(video_dir, out_dir)
            if saved == 0:
                print(f"[WARN] No faces found for {video}")


if __name__ == "__main__":
    main()
