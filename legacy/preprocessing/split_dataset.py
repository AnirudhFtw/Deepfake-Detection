from pathlib import Path
import pandas as pd

# ============================================================
# Configuration
# ============================================================

# Project Root
ROOT_DIR = Path(__file__).resolve().parents[2]

# Dataset Paths
REAL_DIR = ROOT_DIR / "dataset" / "Celeb-real"
FAKE_DIR = ROOT_DIR / "dataset" / "Celeb-synthesis"

# Output CSV
OUTPUT_CSV = ROOT_DIR / "dataset" / "selected_videos.csv"

# Number of Videos
NUM_REAL = 120
NUM_FAKE = 240

# Supported Video Extensions
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


# ============================================================
# Helper Function
# ============================================================

def get_video_files(folder: Path):
    """
    Returns a sorted list of all video filenames in the folder.
    """
    return sorted([
        file.name
        for file in folder.iterdir()
        if file.suffix.lower() in VIDEO_EXTENSIONS
    ])


# ============================================================
# Main
# ============================================================

def main():

    # Read videos
    real_videos = get_video_files(REAL_DIR)
    fake_videos = get_video_files(FAKE_DIR)

    print(f"Found {len(real_videos)} real videos")
    print(f"Found {len(fake_videos)} fake videos")

    # Select first N videos
    selected_real = real_videos[:NUM_REAL]
    selected_fake = fake_videos[:NUM_FAKE]

    rows = []

    video_id = 0

    # Real Videos
    for video in selected_real:

        rows.append({
            "id": video_id,
            "video_name": video,
            "label": 0,
            "class": "real",
            "video_path": str(REAL_DIR / video)
        })

        video_id += 1

    # Fake Videos
    for video in selected_fake:

        rows.append({
            "id": video_id,
            "video_name": video,
            "label": 1,
            "class": "fake",
            "video_path": str(FAKE_DIR / video)
        })

        video_id += 1

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Save CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\nDataset Split Complete!")
    print("-" * 50)
    print(f"Real Videos Selected : {len(selected_real)}")
    print(f"Fake Videos Selected : {len(selected_fake)}")
    print(f"Total Videos         : {len(df)}")
    print(f"CSV Saved To         : {OUTPUT_CSV}")

    print("\nFirst 5 Entries:")
    print(df.head())


if __name__ == "__main__":
    main()