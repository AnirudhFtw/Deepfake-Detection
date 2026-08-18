# Preprocessing pipeline

Consolidated, config-driven replacement for the scattered root-level
preprocessing scripts (see `tasks/IMPROVEMENT_PLAN.md`, Phase 0). Produces
aligned spatial and frequency domain datasets plus a shared video-level
split, so `spatial/` and `frequency/` (and eventually `temporal/`) always
train and evaluate on the same videos.

## Usage

1. Edit `config.py` — set `RAW_DATASET_DIR`, `CATEGORY_MAP` to match your
   dataset's folder layout, and `NUM_VIDEOS_PER_CLASS`.
2. From inside this directory:

```bash
python run_pipeline.py
```

Or run stages individually (each is standalone and re-runnable on its
own):

```bash
python select_videos.py          # -> ../processed/manifest.csv
python extract_frames.py         # -> ../processed/frames/
python extract_faces.py          # -> ../processed/faces/
python build_spatial_domain.py   # -> ../processed/spatial/
python build_frequency_domain.py # -> ../processed/frequency/
python split_dataset.py          # -> ../processed/splits/{train,val,test}.json
```

## Output layout

```
processed/
├── manifest.csv          # selected videos: category, label, video_name, video_path
├── frames/<label>/<video>/frame_00000.jpg, ...
├── faces/<label>/<video>/frame_00000.jpg, ...     (cropped, MTCNN + bbox reuse)
├── spatial/<label>/<video>/frame_00000.npy, ...   (RGB, [-1, 1])
├── frequency/<label>/<video>/frame_00000.npy, ...  (block-DCT, log-scaled, [-1, 1])
└── splits/{train,val,test}.json                   # [{"label": "...", "video": "..."}]
```

`spatial/` and `frequency/` (the domain directories at the repo root, not
the folders above) both point `DeepfakeDataset` at their respective output
directory and load the split via `splits.py` — see each directory's
`train.py`.

## Dependencies

`opencv-python`, `numpy`, `mtcnn`, `tqdm`. Not installed/verified in this
environment — every script here was syntax-checked (`py_compile`), and
`select_videos.py` / `split_dataset.py` (pure stdlib, no cv2/mtcnn) were
smoke-tested against synthetic data. Run the full pipeline once against
real videos on your own machine before relying on it.
