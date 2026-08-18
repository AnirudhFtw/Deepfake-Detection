# Dataset-safety implementation

These changes make the first FaceForensics++ → Celeb-DF experiment safe to
run and report. They do not add a new model or dependency.

## 1. Prepare FaceForensics++

Set the dataset paths in `pipeline/config.py`:

```python
RAW_DATASET_DIR = "/path/to/FaceForensics++"
CATEGORY_MAP = {
    "original_sequences/youtube/c23/videos": "real",
    "manipulated_sequences/Deepfakes/c23/videos": "fake",
    "manipulated_sequences/Face2Face/c23/videos": "fake",
    "manipulated_sequences/FaceSwap/c23/videos": "fake",
    "manipulated_sequences/NeuralTextures/c23/videos": "fake",
}
FFPP_SPLITS_DIR = "/path/to/FaceForensics++/splits"
OUTPUT_DIR = "/path/to/processed_ffpp"
```

`FFPP_SPLITS_DIR` must contain the official `train.json`, `val.json`, and
`test.json` pair lists. The pipeline refuses to guess a split for an FF++
clip it cannot assign unambiguously.

Run preprocessing from `pipeline/`, then train each branch from its own
directory:

```bash
cd pipeline && python run_pipeline.py
cd ../spatial && python train.py
cd ../frequency && python train.py
```

Every output folder now uses a `category__video_name` ID. This prevents clips
with matching filenames from separate source directories overwriting each
other or appearing in different splits.

## 2. Test on Celeb-DF

Change only the dataset configuration, keeping a distinct output location:

```python
RAW_DATASET_DIR = "/path/to/Celeb-DF-v2"
OUTPUT_DIR = "/path/to/processed_celebdf"
FFPP_SPLITS_DIR = None
```

Set `CATEGORY_MAP` to Celeb-DF's real and synthesized video directories.
Run the preprocessing pipeline but do **not** run either training script.
Then use the FF++ checkpoints and evaluate all processed Celeb-DF videos:

```bash
cd spatial && python test.py --all-videos
cd frequency && python test.py --all-videos
```

The test scripts now average frame fake-probabilities per video before
calculating accuracy, F1, and AUC. This matches video inference and prevents
videos with more detected faces from carrying more weight.

## Notes

- Rebuild `processed_ffpp` after this change; old manifests do not contain
  `video_id`.
- Keep `processed_ffpp` and `processed_celebdf` separate. Never train on the
  latter for the cross-dataset result.
- The regular test mode (`python test.py`) still evaluates the selected test
  split; `--all-videos` is for an entirely held-out dataset.
