# MicroPTSD — Face Micro-Expression PTSD Marker Detection

> **⚠️ Research prototype. Not a clinical diagnostic tool.**
> All model outputs require review by a qualified mental health professional
> before any use in clinical, legal, or employment contexts.

A dual-stream deep learning system that detects PTSD-associated micro-expression
patterns in short face video sequences. The model analyzes involuntary facial
muscle movements (duration 1/15–1/25 s) in response to emotionally charged stimuli.

---

## Table of Contents

- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Format](#data-format)
- [Training](#training)
- [Inference](#inference)
- [Metrics & Fairness](#metrics--fairness)
- [Design Decisions](#design-decisions)
- [Ethical Considerations](#ethical-considerations)
- [Known Limitations](#known-limitations)

---

## Architecture

The model uses a **two-stream fusion** design, motivated by the complementary
nature of temporal dynamics (micro-movements) and fine spatial detail (muscle
groups):

```
Input video (B, 3, T, H, W)
         │
    ┌────┴────┐
    │         │
    ▼         ▼
Stream 1    Stream 2
(2+1)D CNN  Video ViT
  + TSM     patch 8×8
    │         │
    └────┬────┘
         │ Cross-Attention
         │  (CNN queries ViT)
         ▼
    Temporal avg-pool
         │
    Linear head
         │
    Binary logit
```

### Stream 1 — (2+1)D CNN with Temporal Shift Module

Factorized 3D convolutions that separate spatial and temporal processing.
Each residual block has a **Temporal Shift Module (TSM)** prepended: a
zero-parameter operation that shifts 1/8 of channels one step forward and
1/8 one step backward along the time axis, giving the spatial conv access
to neighbouring frames without added cost.

| Layer   | Out channels | Stride |
|---------|-------------|--------|
| Stem    | 32          | s=2 spatial |
| Block 1 | 64          | s=1    |
| Block 2 | 128         | s=2    |
| Block 3 | 256         | s=2    |
| SpatialPool | 256    | (T', 1, 1) → (B, T', 256) |

### Stream 2 — Video Vision Transformer (patch size 8×8)

Tubelet embedding with spatial patch size **8×8 px** (half the standard
16×16) to preserve subtle per-pixel muscle activation. Temporal patch size
is 2 frames. Four transformer encoder layers with pre-norm and GELU.

| Parameter      | Value |
|----------------|-------|
| Patch size     | 8×8 px |
| Temporal patch | 2 frames |
| Embed dim      | 256 |
| Depth          | 4 layers |
| Heads          | 8 |
| Tokens (32 frames, 112px) | 196 × 16 = 3 136 |

### Cross-Attention Fusion

CNN temporal features act as **queries**; ViT spatial tokens as keys/values.
Each temporal position attends over all spatial tokens, producing a fused
sequence that is then averaged and classified.

### Loss Function

```
L = FocalLoss(γ=2, α=0.25)  +  λ · SupContrastiveLoss(τ=0.07)
                                 λ = 0.3  (default)
```

- **Focal Loss** down-weights easy negatives, focusing training on hard
  micro-expression patterns.
- **Supervised Contrastive Loss** pulls together embeddings from the same
  patient and pushes apart embeddings from different patients, encouraging
  subject-invariant representations.

---

## Project Structure

```
.
├── model_ptsd.py      # PTSDMicroExpressionModel (PyTorch Lightning)
│                      #   TemporalShiftModule, R2Plus1DBlock, CNNStream
│                      #   PatchEmbed3D, VideoViT, CrossAttentionFusion
│                      #   FocalLoss, ContrastiveLoss, DemographicParityGap
│                      #   check_input_quality
│
├── dataset.py         # PTSDVideoDataset, PTSDDataModule
│                      #   FaceAligner (MediaPipe), read_video_frames
│                      #   compute_difference_map
│                      #   TemporalAugmentor, SpatialAugmentor
│
├── train.py           # TrainConfig (dataclass), patient-level K-Fold CV
│                      #   prepare_cv_manifest, split_manifest_for_fold
│                      #   build_trainer, train_fold, run_cv
│
├── inference.py       # CLI + run_inference, video_to_tensor
│
└── README.md
```

---

## Installation

Tested on Python 3.10+, PyTorch 2.2+.

```bash
# 1. Create environment
python -m venv .venv && source .venv/bin/activate

# 2. Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install project dependencies
pip install \
    pytorch-lightning>=2.2 \
    torchmetrics>=1.3 \
    einops \
    timm \
    opencv-python \
    mediapipe>=0.10 \
    scikit-learn \
    pandas \
    click \
    rich

# 4. Verify
python - <<'EOF'
import torch, pytorch_lightning as pl, mediapipe as mp
print("torch:", torch.__version__, "| pl:", pl.__version__)
print("CUDA available:", torch.cuda.is_available())
EOF
```

---

## Data Format

### Manifest CSV

Training and inference require a CSV manifest file. Minimum required columns:

| Column       | Type   | Description |
|--------------|--------|-------------|
| `video_path` | str    | Absolute or relative path to the video file |
| `label`      | int    | `1` = PTSD markers present, `0` = absent |
| `patient_id` | int    | Unique patient identifier (used for cross-val grouping) |
| `split`      | str    | `train` / `val` / `test` |
| `gender`     | int    | *(optional)* Demographic group code for fairness logging |
| `race`       | int    | *(optional)* Demographic group code for fairness logging |

```csv
video_path,label,patient_id,split,gender,race
data/p001_trial1.mp4,1,1,train,0,2
data/p001_trial2.mp4,1,1,train,0,2
data/p002_trial1.mp4,0,2,val,1,0
...
```

> **Important:** the `split` column in the raw manifest is overwritten during
> cross-validation. For plain train/val/test, set it manually and run
> `train_fold()` directly instead of `run_cv()`.

### Video Requirements

| Requirement      | Minimum | Recommended |
|------------------|---------|-------------|
| Frame rate       | 25 fps  | 30+ fps |
| Duration         | ~2 s (50 f) | 2–4 s (60–120 f) |
| Resolution       | 112×112 after crop | 224×224 raw |
| Face visibility  | >50% of frames | >90% of frames |
| Format           | Any OpenCV-readable | MP4 / AVI |

Videos below 25 fps are **upsampled** via linear frame interpolation with a
warning. Videos where the face is undetected in >50% of frames return
`prediction = -1` (undetermined).

### Preprocessing Pipeline

For each video the following steps run at load time:

```
1. Read frames with OpenCV, resample to target_fps if needed
2. Uniform temporal sampling → exactly num_frames frames
3. Per-frame face alignment via MediaPipe Face Mesh (2D affine warp,
   anchored to eye landmarks)
4. BGR → RGB conversion (single pass, inside FaceAligner)
5. Normalise to [0, 1] float32
6. Compute difference map:  frame_t  −  mean(frames[0:5])
   then rescale [-1, 1] → [0, 1]
7. ImageNet mean/std normalisation
```

The difference map (step 6) removes static facial appearance and highlights
only temporal changes — the key signal for micro-expression detection.

---

## Training

### Quick start

```python
# train.py — edit the defaults or override inline:
from train import TrainConfig, run_cv

cfg = TrainConfig(
    manifest_path = "data/manifest.csv",
    output_dir    = "runs/exp01",
    max_epochs    = 50,
    n_folds       = 5,
    batch_size    = 8,
    num_workers   = 4,
)
run_cv(cfg)
```

Or from the command line (edit the bottom of `train.py`):

```bash
python train.py
```

For a smoke-test with minimal resources, uncomment the override block at the
bottom of `train.py`:

```python
cfg.max_epochs = 2
cfg.n_folds    = 2
cfg.batch_size = 2
cfg.num_workers = 0
cfg.precision  = "32"
```

### Cross-Validation Strategy

The training loop uses **patient-level GroupKFold** (`sklearn`). This is
critical: splitting by video clip instead of by patient causes data leakage,
because the same person's face appears in both train and val sets, and the
model learns identity rather than PTSD markers.

```
All videos
└── grouped by patient_id
    └── GroupKFold(n_splits=5)
        ├── Fold 0: patients [3,7,12,...] → val
        ├── Fold 1: patients [1,5,9,...]  → val
        └── ...
```

The CV summary (mean ± std across folds) is saved to
`{output_dir}/cv_summary.json`.

### Key Hyperparameters

| Parameter              | Default | Notes |
|------------------------|---------|-------|
| `lr`                   | 3e-4    | Peak LR after warmup |
| `warmup_epochs`        | 5       | Linear warmup |
| `max_epochs`           | 50      | Cosine decay after warmup |
| `weight_decay`         | 1e-4    | AdamW |
| `focal_gamma`          | 2.0     | Higher = more focus on hard examples |
| `focal_alpha`          | 0.25    | Class balance factor |
| `contrastive_weight`   | 0.3     | `λ` for SupCon loss |
| `d_model`              | 256     | Shared embedding dimension |
| `dropout`              | 0.1     | Applied in ViT and classifier head |
| `accumulate_grad_batches` | 2    | Effective batch = batch_size × 2 |

### Outputs

```
runs/
└── exp01/
    ├── manifest_cv.csv              ← full manifest with cv_fold column
    ├── manifest_cv_fold0.csv        ← per-fold split manifests
    ├── ...
    ├── fold_0/
    │   ├── fold0-epoch12-0.8731.ckpt   ← best checkpoint
    │   ├── last.ckpt
    │   ├── tb/                          ← TensorBoard logs
    │   └── csv/                         ← CSV logs
    ├── fold_1/ ...
    └── cv_summary.json              ← aggregated metrics
```

---

## Inference

### CLI

```bash
python inference.py \
  --checkpoint runs/exp01/fold_0/fold0-epoch12-0.8731.ckpt \
  --video      /path/to/patient_video.mp4 \
  --fps        25 \
  --num-frames 32

# Output:
{
  "prediction":  1,
  "probability": 0.812,
  "reason":      "ok"
}
```

**Exit codes:**
- `0` — prediction produced (`0` or `1`)
- `1` — fatal error (missing file, import failure)
- `2` — undetermined (`prediction = -1`)

### Python API

```python
from inference import run_inference

result = run_inference(
    checkpoint_path = "runs/exp01/fold_0/best.ckpt",
    video_path      = "data/new_patient.mp4",
    num_frames      = 32,
    img_size        = 112,
)

print(result["prediction"])   # 0 | 1 | -1
print(result["probability"])  # float or None
print(result["reason"])       # "ok" or rejection reason
```

### Undetermined output (`-1`)

The model returns `prediction = -1` and refuses to output `0` or `1` when:

| Condition | Threshold |
|-----------|-----------|
| Video FPS too low | < 25 fps |
| Too few frames | < 15 frames |
| Video is static (frozen/black) | > 50 consecutive near-identical frames |
| Face undetected | > 50% of frames |

---

## Metrics & Fairness

The following metrics are logged per epoch to TensorBoard and CSV:

| Metric | Description |
|--------|-------------|
| `{split}/loss` | Total loss (Focal + λ·Contrastive) |
| `{split}/acc`  | Binary accuracy |
| `{split}/auroc` | Area under ROC curve *(primary metric)* |
| `{split}/f1`   | F1 score |
| `val/tp`, `val/fp`, `val/tn`, `val/fn` | Confusion matrix cells |
| `val/demographic_parity_gap` | Max difference in positive-prediction rate across demographic groups |

### Demographic Parity

At the end of every validation epoch, the model computes the **Demographic
Parity Gap**: the maximum difference in predicted-positive rate between any
two demographic groups (e.g. gender codes `0` vs `1`).

```
gap = max_group(P̂[ŷ=1]) − min_group(P̂[ŷ=1])
```

A gap above **0.10** triggers a `WARNING` log. This does not stop training
but signals that the model may be making systematically different predictions
for different groups, warranting investigation before any deployment.

To enable this metric, set `demo_attr_col` in `TrainConfig` to the manifest
column you want to monitor (`"gender"`, `"race"`, etc.).

---

## Design Decisions

### Why (2+1)D + TSM instead of full 3D CNN?

Full R3D convolutions have ~3× more parameters than (2+1)D for the same
receptive field. TSM adds **zero parameters** while giving the spatial
convolution access to adjacent frames. For short micro-expression clips
(32 frames), this keeps the model trainable on small datasets.

### Why 8×8 patches instead of 16×16?

Standard ViT patch size 16×16 at 112px input gives only 49 tokens spatially.
At 8×8 we get 196 tokens — crucial for resolving subtle per-muscle activations
such as the corrugator supercilii (inner brow raise) or orbicularis oculi
(eye tightening) which span only 10–20 pixels in a 112px crop.

### Why difference maps?

Absolute pixel values encode identity (skin tone, face shape). Subtracting
the neutral baseline (mean of the first 5 frames) removes static appearance
and leaves only temporal change — precisely what micro-expression detection
requires. This also makes the model more robust across individuals.

### Why patient-level cross-validation?

With video-level splits, the same patient can appear in both train and val.
The model then learns **who** the person is rather than whether their
micro-expressions indicate PTSD. GroupKFold on `patient_id` ensures a
complete patient hold-out per fold.

---

## Ethical Considerations

This system is a **research prototype** built on a hypothetical dataset.
Before any real-world use:

1. **No standalone diagnosis.** Model output must be reviewed by a licensed
   psychiatrist or psychologist. PTSD diagnosis requires clinical interview,
   validated questionnaires (PCL-5, CAPS-5), and longitudinal assessment.

2. **Informed consent.** Video collection for this purpose requires explicit,
   informed consent from participants, with the right to withdraw data.

3. **Demographic auditing.** The `demographic_parity_gap` metric is a
   necessary but not sufficient fairness check. A full audit (equalized odds,
   calibration per subgroup) must be performed before deployment.

4. **Dataset bias.** Model behaviour on populations not represented in
   training is unknown and potentially harmful.

5. **Adversarial use.** This system must not be used for covert surveillance,
   employment screening, or any application where the subject is unaware of
   or has not consented to the analysis.

---

## Known Limitations

| Limitation | Impact | Mitigation |
|------------|--------|-----------|
| Hypothetical dataset | Cannot benchmark against real-world distribution | Requires IRB-approved data collection |
| MediaPipe alignment failure | Falls back to centre-crop; >50% failure → sample rejected | Pre-filter dataset; improve capture conditions |
| `num_workers > 0` and MediaPipe | Each worker spawns its own FaceMesh instance | Mitigated by lazy init; set `num_workers ≤ 4` on low-RAM machines |
| Static video detection | Heuristic based on mean pixel diff | May reject legitimate low-motion sessions |
| 25 fps minimum | Lower frame rate loses temporal resolution | Always capture at ≥30 fps |
| No temporal context across clips | Model sees each clip independently | Future: add session-level aggregation |
