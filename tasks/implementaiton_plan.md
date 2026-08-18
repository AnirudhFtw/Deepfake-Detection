# Improvement Plan — Multi-Domain Deepfake Detection

Tracks the gap between what exists today (two independent, unfused
spatial/frequency classifiers) and the multi-domain fusion system this
project is aiming for. See the root `README.md` for the architecture this
plan builds toward, and `pipeline/`, `spatial/`, `frequency/`, `temporal/`
for the code each phase below produced or will produce.

Status legend: `[x]` done · `[~]` in progress · `[ ]` not started

## Phase 0 — Pipeline consolidation

**Why:** the repo had two `extract_frames.py`, two overlapping
face-extraction scripts, and hardcoded Colab/Windows/relative paths
repeated per script — no single source of truth for preprocessing, and no
guarantee the spatial and frequency branches were ever trained on the same
videos.

- [x] Single `pipeline/config.py` — one place to set dataset location,
  sample counts, frame/detection parameters instead of editing constants
  in five different files.
- [x] `pipeline/select_videos.py` — one video-sampling stage (replaces
  `initialdatasetsplit.py` + `preprocessing/split_dataset.py`).
- [x] `pipeline/extract_frames.py` — samples `FRAMES_PER_VIDEO` frames
  uniformly instead of dumping every frame and discarding most of them
  later at face-detection time.
- [x] `pipeline/extract_faces.py` — MTCNN every `FACE_DETECT_EVERY_N`
  frames with bbox reuse in between (the old root `extract_faces.py` ran
  MTCNN on every frame; `predict_video.py` still does — carried over
  from `preprocessing/detect_faces.py`'s approach, applied consistently
  everywhere now).
- [x] `pipeline/split_dataset.py` + `pipeline/splits.py` — one video-level
  train/val/test split, written once and shared by every domain, so
  spatial, frequency, and eventually temporal results are comparable and
  fusion is trained/evaluated on matching data.
- [x] Archive superseded root-level prototype scripts in `legacy/`; the
  current preprocessing and branch entry points now live only in
  `pipeline/`, `spatial/`, and `frequency/`.

## Phase 1 — Fix the frequency branch's input representation

**Why:** `apply_dct.py` ran `cv2.dct()` over the *whole* face crop.
Whole-image DCT concentrates low frequencies in one corner and breaks the
shift-invariance/local-consistency a CNN backbone depends on — the exact
problem F3-Net's Local Frequency Statistics module and the block-wise DCT
literature exist to solve. It also saved raw, unnormalized coefficients,
and `train.py` loaded ImageNet-pretrained weights for this branch
regardless — a pretrained prior that doesn't apply to DCT statistics.

- [x] `pipeline/build_frequency_domain.py` — JPEG-style block-wise
  (8×8) DCT instead of whole-image DCT, keeping frequency information
  spatially localized.
- [x] Log-magnitude scaling + rescale to [-1, 1] before saving — the
  normalization step the old pipeline skipped.
- [x] `frequency/model.py` — small CNN trained **from scratch** (no
  ImageNet pretraining) sized for frequency statistics rather than
  natural-image texture, and cheaper to train than reusing ResNet18.

## Phase 2 — Data augmentation

**Why:** `preprocess_faces.py`/`build_spatial_domain.py` only resize and
normalize — no compression, blur, noise, or re-encoding simulation.
Recent ablations ("Generalized Design Choices for Deepfake Detectors",
2025) found the augmentation pipeline moved generalization more than
backbone architecture did.

- [ ] Add a stochastic augmentation chain (JPEG re-compression, blur,
  noise, resize) applied at training time in `spatial/train.py` and
  `frequency/train.py` — training-time only, not baked into the cached
  `.npy` files, so severity can be varied per-epoch.
- [ ] Verify robustness by re-running `spatial/test.py` /
  `frequency/test.py` against artificially degraded copies of the test
  split.

## Phase 3 — Cross-dataset / cross-manipulation evaluation

**Why:** `test.py` only ever evaluated a held-out split of the same
source distribution. Every relevant benchmark shows detectors hitting
>99% AUC in-domain and collapsing to 0.5–0.7 AUC on a different dataset or
manipulation method — until this is measured, it's unknown whether this
model has that problem.

- [ ] Point `pipeline/config.py` at a second dataset (e.g. FaceForensics++
  if training on Celeb-DF, or vice versa) and run it **only** through
  `test.py`/`test`-side of the pipeline, never through training.
- [ ] Report both the in-domain AUC and this held-out cross-dataset AUC
  side by side in `README.md`'s Results section — the second number is
  the one that matters.

## Phase 4 — Temporal branch

**Why:** `predict_video.py` currently aggregates video-level predictions
with `np.mean()` over per-frame fake-probabilities — a fixed heuristic,
not a learned signal. FTCN, RealForensics, and CAST all show temporal
inconsistency is complementary to spatial/frequency artifacts and
specifically helps cross-manipulation generalization, since many forgery
pipelines don't enforce frame-to-frame coherence.

- [x] `temporal/model.py` — `TemporalAttentionPool` scaffolded (attention
  over a sequence of per-frame embeddings) so the interface it needs from
  the other branches — a per-frame **embedding**, not just a per-frame
  logit — is decided before they're wired up. Intentionally not trained
  yet (see `temporal/README.md`); the user is building this branch next.
- [ ] Add an embedding-extraction hook to `spatial/model.py` and
  `frequency/model.py` (return the pre-classifier feature vector).
- [ ] Build a per-video sequence dataset (stack of frame embeddings) and
  a training loop for the temporal aggregator.

## Phase 5 — Fusion

**Why:** today spatial and frequency are two independent models with no
shared representation or combined decision — the core premise of this
project isn't built yet.

- [ ] Start with concatenation of per-branch embeddings + a small MLP
  head. CBO-DD (2025) showed simple concatenation matches cross-attention
  fusion once branches already encode distinct information — don't build
  cross-attention fusion until concatenation demonstrably plateaus.
- [ ] If concatenation underperforms, upgrade to cross-attention fusion
  (F3-Net's MixBlock / CAST's cross-attention) between spatial and
  frequency branches before/instead of after temporal pooling.
- [ ] Re-run Phase 3's cross-dataset evaluation on the fused model and
  compare against each standalone branch — the fusion is only worth
  keeping if it beats the best single branch on the held-out dataset, not
  just on the training distribution.

## Dataset sizing (reference for Phase 0/3)

- Minimum credible prototype: ~1,000–2,000 videos/class across ≥3–4
  manipulation methods (FaceForensics++ scale).
- For a genuine cross-dataset generalization claim: train on one full
  dataset, hold out a second dataset **entirely** — never trained on — and
  report AUC on it.
- `pipeline/config.py`'s `NUM_VIDEOS_PER_CLASS` defaults to 1000; it will
  silently cap to however many videos actually exist per class and print
  a warning if fewer are available (e.g. Celeb-DF v2 only has 590 real
  videos total).
