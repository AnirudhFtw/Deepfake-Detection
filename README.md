# Multi-Domain Deepfake Video Detection

A face-forgery detection system that classifies videos as real or fake by
combining evidence from more than one representation of the same footage,
instead of relying on a single CNN staring at raw pixels.

## Background

Frame-level RGB classifiers (a CNN looking at a cropped face) plateau
quickly: once a manipulation method is visually clean, pixel-level
artifacts are too subtle to catch, and every published benchmark shows the
same failure pattern — near-perfect accuracy on the dataset a model was
trained on, and a collapse to near-random performance the moment it sees a
different manipulation method or a different dataset. Two independent
sources of evidence turn out to survive this collapse better than raw
pixels alone:

- **Frequency-domain artifacts.** Generative upsampling and blending leave
  statistical fingerprints in the spectrum (DCT/FFT) that are invisible in
  RGB but persist even under compression.
- **Temporal inconsistency.** Most manipulation pipelines operate
  frame-by-frame, so they don't guarantee coherent motion across frames —
  flickering, unstable blending boundaries, and unnatural micro-expressions
  show up only when several frames are considered jointly.

This project's premise is that a spatial branch, a frequency branch, and a
temporal branch each pick up on different failure modes of a forgery, and
that combining them generalizes better than any one branch alone — a
pattern consistently confirmed in recent literature (F3-Net, FTCN,
RealForensics, CAST, ForensicFlow — see [References](#references)).

## Objectives

1. Detect manipulated faces in video using complementary spatial,
   frequency, and temporal signals.
2. Fuse the three branches instead of training and shipping three separate
   detectors.
3. Prioritize cross-manipulation / cross-dataset generalization over
   in-domain accuracy, since in-domain accuracy is the easy, less
   meaningful number.
4. Keep the pipeline efficient enough to run on modest (single-GPU,
   Colab-class) hardware.

## System Architecture

```
video → face detection & tracking → aligned face crops (per frame)
                │
      ┌─────────┼─────────────────┐
      ▼         ▼                 ▼
  Spatial    Frequency        Temporal
  branch     branch           aggregator
  (RGB CNN)  (DCT sub-bands)  (attends over per-frame
      │         │              embeddings from both
      └────┬────┘              branches above)
           ▼
       Fusion head
           ▼
    real / fake + confidence
```

### Spatial branch (implemented)
ResNet18, ImageNet-pretrained, final FC replaced with `Dropout(0.5) →
Linear(2)`. Operates on face crops resized to 224×224, normalized to
[-1, 1]. See `model.py`.

### Frequency branch (implemented, standalone — not yet fused)
Per-channel 2D DCT of the processed RGB face crop (`apply_dct.py`),
currently trained through the same ResNet18 architecture as its own
separate model. This branch mines spectral artifacts (upsampling
signatures, GAN fingerprints, compression-noise mismatches) that are
invisible in RGB.

### Temporal branch (planned)
Not yet implemented. `predict_video.py` currently aggregates video-level
predictions by averaging per-frame fake-probabilities — a static heuristic,
not a learned temporal signal. The plan is a lightweight temporal
aggregator (attention pooling or a shallow transformer) over the sequence
of per-frame embeddings already produced by the spatial/frequency branches,
so no new heavy per-frame feature extractor is needed — see
[Roadmap](#roadmap).

### Fusion (planned)
The two-branch and eventual three-branch outputs need a shared fusion head;
today they are two independent models with no shared representation or
combined decision. See [Roadmap](#roadmap) for the concrete plan.

## Pipeline

| Stage | Script | Notes |
|---|---|---|
| Video sampling | `initialdatasetsplit.py`, `preprocessing/split_dataset.py` | Selects a bounded subset of videos per manipulation category |
| Frame extraction | `extract_frames.py`, `preprocessing/extract_frames.py` | OpenCV frame dump |
| Face detection & crop | `extract_faces.py`, `preprocessing/detect_faces.py` | MTCNN, adaptive context margin |
| Normalization | `preprocess_faces.py` | Resize 224×224, RGB, scale to [-1, 1] |
| Frequency transform | `apply_dct.py` | Per-channel 2D DCT |
| Tensor packaging | `create_rgb_dataset.py` | Bundles `.npy` frames into a `.pt` dataset |
| Training | `train.py` | Video-level 80/20 split, Adam + `ReduceLROnPlateau` |
| Evaluation | `test.py` | Accuracy, precision, recall, F1, AUC, confusion matrix |
| Inference | `predict_video.py` | End-to-end video → REAL/FAKE + confidence |

`DeepfakeDataset` (`dataset.py`) indexes samples per-video, which is what
makes the video-level train/val split possible — frames from the same
video never leak across the split.

## Setup

Dependencies (no `requirements.txt` yet — inferred from imports):

```
torch torchvision opencv-python mtcnn numpy tqdm scikit-learn pandas
```

Paths across the scripts are currently hardcoded per environment
(Colab `/content/...`, Windows `C:\Honours\...`, relative `../...`) —
edit the `Configuration` block at the top of each script before running it.

## Current Status & Results

- Spatial (RGB) and frequency (DCT) branches train and evaluate
  independently; no fused multi-branch model exists yet.
- Temporal modeling is not implemented — video-level prediction is a mean
  of per-frame probabilities.
- No cross-dataset or cross-manipulation evaluation has been run; `test.py`
  evaluates on a held-out split of the same source distribution as
  training.
- *(Fill in once available: in-domain accuracy/AUC per branch, and, once a
  held-out second dataset is used, the cross-dataset AUC — this second
  number is the one that actually matters for the write-up.)*

## Roadmap

1. **Fix the frequency branch's input representation** before fusing it
   with anything — full-image 2D DCT breaks the shift-invariance/locality
   a CNN backbone needs, and doesn't match the ImageNet statistics its
   pretrained weights expect. Move to block-wise (8×8, JPEG-style) DCT or
   frequency-band decomposition, and stop initializing it from ImageNet
   pretrained weights.
2. **Add the temporal branch** as an attention-pooling or shallow
   transformer layer over per-frame embeddings already produced by the
   spatial/frequency CNNs — this avoids a separate, expensive 3D-CNN
   pass over raw frames.
3. **Fuse**, starting with concatenation + a small MLP head before
   reaching for cross-attention fusion — simpler fusion has matched
   cross-attention fusion in the literature when the branches already
   encode distinct information.
4. **Evaluate cross-dataset**, not just cross-video-split — train on one
   dataset, report AUC on a completely held-out second dataset.
5. **Add data augmentation** (compression, blur, noise, resize) — this has
   moved the needle on generalization more than architecture choice in
   recent ablations.

## Limitations

- Small, single-source training set (a few hundred videos) relative to
  standard benchmarks (FaceForensics++: 1,000 real / 4,000 fake;
  Celeb-DF v2: 590 real / 5,639 fake).
- No robustness testing against compression, resizing, or re-encoding.
- No cross-manipulation or cross-dataset generalization numbers yet.
- Frequency and spatial models are currently separate artifacts, not a
  fused system.

## References

- Qian et al., *Thinking in Frequency: Face Forgery Detection by Mining
  Frequency-aware Clues (F3-Net)*, ECCV 2020.
- Zheng et al., *Exploring Temporal Coherence for More General Video Face
  Forgery Detection (FTCN)*, ICCV 2021.
- Haliassos et al., *Leveraging Real Talking Faces via Self-Supervision for
  Robust Forgery Detection (RealForensics)*, CVPR 2022.
- Rössler et al., *FaceForensics++: Learning to Detect Manipulated Facial
  Images*, ICCV 2019.
- Li et al., *Celeb-DF: A Large-scale Challenging Dataset for DeepFake
  Forensics*, CVPR 2020.
- *CAST: Cross-Attentive Spatio-Temporal Feature Fusion for Deepfake
  Detection*, 2025.
- *ForensicFlow: A Tri-Modal Adaptive Network for Robust Deepfake
  Detection*, 2025.
- Fernando et al., *Cross-Branch Orthogonality for Improved Generalization
  in Face Deepfake Detection*, 2025.
