"""
Runs every pipeline stage in order. Each stage is also runnable standalone
(e.g. `python extract_faces.py`) if you only need to redo one step.
"""
import build_frequency_domain
import build_spatial_domain
import extract_faces
import extract_frames
import select_videos
import split_dataset


def main():
    print("\n[1/6] Selecting videos...")
    select_videos.main()

    print("\n[2/6] Extracting frames...")
    extract_frames.main()

    print("\n[3/6] Detecting & cropping faces...")
    extract_faces.main()

    print("\n[4/6] Building spatial domain...")
    build_spatial_domain.main()

    print("\n[5/6] Building frequency domain...")
    build_frequency_domain.main()

    print("\n[6/6] Splitting dataset...")
    split_dataset.main()

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
