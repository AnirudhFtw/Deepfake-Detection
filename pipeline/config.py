"""
Single configuration point for every pipeline stage. Edit this file, then
run `python run_pipeline.py` (or any stage script standalone) from inside
this directory — replaces the per-script hardcoded constants scattered
across the old root-level scripts.
"""
import os

# ============================================================
# Raw dataset
# ============================================================

RAW_DATASET_DIR = "../dataset"

# Maps source folder name -> binary label. Add/remove entries to match
# whichever dataset(s) you're pointing at.
CATEGORY_MAP = {
    "Celeb-real": "real",
    "YouTube-real": "real",
    "Celeb-synthesis": "fake",
}

NUM_VIDEOS_PER_CLASS = 1000  # capped to however many videos actually exist

# ============================================================
# Output layout
# ============================================================

OUTPUT_DIR = "../processed"

FRAMES_DIR = os.path.join(OUTPUT_DIR, "frames")
FACES_DIR = os.path.join(OUTPUT_DIR, "faces")
SPATIAL_DIR = os.path.join(OUTPUT_DIR, "spatial")
FREQUENCY_DIR = os.path.join(OUTPUT_DIR, "frequency")
SPLITS_DIR = os.path.join(OUTPUT_DIR, "splits")

# ============================================================
# Sampling / detection
# ============================================================

FRAMES_PER_VIDEO = 32     # uniformly sampled, not every frame
FACE_DETECT_EVERY_N = 5   # re-run MTCNN every N frames, reuse bbox otherwise
MIN_FACE_SIZE = 50
IMG_SIZE = 224

# ============================================================
# Frequency domain
# ============================================================

DCT_BLOCK_SIZE = 8  # JPEG-style block DCT instead of whole-image DCT

# ============================================================
# Split
# ============================================================

VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42
