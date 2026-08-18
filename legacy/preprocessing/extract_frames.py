from pathlib import Path
import cv2
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

REAL_DIR = ROOT_DIR / "dataset" / "Celeb-real"
FAKE_DIR = ROOT_DIR / "dataset" / "Celeb-synthesis"

OUTPUT_DIR = ROOT_DIR / "processed" / "frames"

NUM_REAL = 120
NUM_FAKE = 240

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

# ============================================================


def get_video_list(folder: Path, limit: int):
    """
    Returns first N videos from a folder.
    """
    videos = sorted([
        file
        for file in folder.iterdir()
        if file.suffix.lower() in VIDEO_EXTENSIONS
    ])

    return videos[:limit]


def extract_video(video_path: Path, output_folder: Path):
    """
    Extract every frame from a video.
    """

    output_folder.mkdir(parents=True, exist_ok=True)

    # Resume support
    if any(output_folder.iterdir()):
        return "skipped", 0

    cap = cv2.VideoCapture(str(video_path))

    frame_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_name = output_folder / f"{frame_count:06d}.jpg"

        cv2.imwrite(str(frame_name), frame)

        frame_count += 1

    cap.release()

    return "processed", frame_count


def process_class(video_list, class_name):

    class_output = OUTPUT_DIR / class_name
    class_output.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    processed = 0
    skipped = 0

    print(f"\nProcessing {class_name.upper()} videos")

    for video_path in tqdm(video_list):

        video_name = video_path.stem

        output_folder = class_output / video_name

        status, frames = extract_video(video_path, output_folder)

        if status == "processed":
            processed += 1
            total_frames += frames
        else:
            skipped += 1

    print(f"\n{class_name.upper()} Summary")
    print("-" * 40)
    print(f"Videos Processed : {processed}")
    print(f"Videos Skipped   : {skipped}")
    print(f"Frames Extracted : {total_frames}")

    return processed, skipped, total_frames


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    real_videos = get_video_list(REAL_DIR, NUM_REAL)
    fake_videos = get_video_list(FAKE_DIR, NUM_FAKE)

    print("=" * 60)
    print("FRAME EXTRACTION")
    print("=" * 60)

    print(f"Real Videos : {len(real_videos)}")
    print(f"Fake Videos : {len(fake_videos)}")

    r_processed, r_skipped, r_frames = process_class(real_videos, "real")

    f_processed, f_skipped, f_frames = process_class(fake_videos, "fake")

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print(f"Processed Videos : {r_processed + f_processed}")
    print(f"Skipped Videos   : {r_skipped + f_skipped}")
    print(f"Frames Extracted : {r_frames + f_frames}")

    print("\nFrames saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()