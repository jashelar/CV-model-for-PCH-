# Methodology

This document describes how a video clip becomes a PTSD-risk screening score,
and the reasoning behind each stage.

## 1. Landmark extraction (`landmarks.py`)

Frames are decoded with OpenCV (configurable `frame_stride`) and passed to
MediaPipe's **FaceLandmarker** (Tasks API), which returns **478 3-D facial
landmarks** per frame. Frames with no detected face produce a NaN row rather
than being silently dropped, so downstream code can reason about missingness
explicitly. A clip whose missing-frame fraction exceeds
`extraction.max_missing_fraction` (default 0.30) is treated as low quality.

The model bundle (`face_landmarker.task`) is not committed; fetch it with
`scripts/download_model.py`.

## 2. Action Units from geometry (`action_units.py`)

Rather than feeding raw pixels or raw landmarks to a network, the pipeline
first computes **8 FACS-grounded Action Units** from landmark geometry. This is
a deliberate inductive bias toward clinically interpretable features.

Each frame is first put into a canonical frame of reference:

- **Alignment + scaling.** Landmarks are aligned and normalised by
  **inter-ocular distance**, making the AU measurements invariant to head
  scale and camera distance, and largely robust to in-plane pose.
- AU intensities are then read off as normalised distances / ratios between
  landmark groups (e.g. an eye-aspect-ratio style measure for the lid
  tightener, brow-to-eye distances for raise/knit, lip-corner geometry for
  pull/depress, lip aperture for jaw drop).

| AU channel | What it measures | Clinical reading |
|---|---|---|
| `AU1_2_brow_raise`     | inner+outer brow elevation | orienting / startle |
| `AU4_brow_knit`        | brow lowering / drawing together | tension, negative valence |
| `AU6_cheek_raise`      | cheek raise | Duchenne (genuine) smile marker |
| `AU7_lid_tighten`      | lower-lid tightening | **periocular tension** |
| `AU12_lip_corner_pull` | lip-corner pull | positive affect |
| `AU15_lip_corner_depress` | lip-corner depression | sadness |
| `AU20_lip_stretch`     | horizontal lip stretch | fear / startle |
| `AU25_26_jaw_drop`     | lips part / jaw lowers | surprise / affective response |

NaN frames propagate as NaN AU rows; the sequence builder masks them.

## 3. Feature engineering (`features.py`)

The per-frame AU rows form a `(T, 8)` sequence, processed as:

1. **Interpolate** short gaps (the masked NaN frames) linearly, reporting the
   filled fraction.
2. **Smooth** with a centred moving average (`smoothing_window`, default 5) to
   suppress single-frame landmark jitter.
3. **Resample** every clip to a fixed `sequence_length` (default 128) so clips
   of different duration share one tensor shape.
4. **Standardise** (z-score) each AU channel using statistics estimated **only
   on the training split**, then applied to val/test (no leakage).

For the gradient-boosting baseline the sequence is reduced to a **50-dim
aggregate vector**: per channel `mean, std, min, max, linear-trend slope,
dynamics (mean absolute first difference)` (8 × 6 = 48), plus two global
scalars — **expressivity** (mean of per-channel std-devs; low ⇒ blunted/flat
affect) and **total motion**. This makes "flat affect" and "sustained tension"
directly available as features.

## 4. Models

**Temporal 1D-CNN — `model_cnn.py` (headline).** `TemporalAUNet` runs 1-D
convolutions over the time axis of the `(8, T)` AU sequence (channels
`32→64→128`, kernel 5, dropout 0.3), global-pools, and outputs a single logit.
Trained with class-balanced loss (`pos_weight`), Adam, and **early stopping on
validation AUC**. PyTorch is imported lazily so the rest of the repo runs
without it.

**Gradient-boosting baseline — `model_baseline.py`.** A
`HistGradientBoostingClassifier` over the 50-dim aggregate features. Needs no
deep-learning stack and is the default for the demo and CI.

## 5. Splitting & evaluation (`dataset.py`, `evaluate.py`)

- **Subject-aware split.** `GroupShuffleSplit` on `subject_id` guarantees no
  subject appears in more than one split, so metrics aren't inflated by the
  model memorising identities. A unit test enforces this.
- **Operating point by FPR budget.** The decision threshold is chosen on the
  **validation** split as the *lowest* threshold whose false-positive rate
  stays within `train.target_fpr`; among thresholds meeting the budget the
  lowest maximises sensitivity. That threshold is then frozen and applied to
  the test split. This mirrors tuning thresholds with clinicians to keep false
  positives low, and makes FPR the primary controllable quantity.
- Reported metrics: ROC-AUC (threshold-free headline), plus accuracy,
  sensitivity/recall, specificity, F1, and FPR at the chosen threshold.

## 6. Reproducibility (`utils.py`)

Every run creates `runs/<timestamp>/` containing the **exact resolved config**,
a **data fingerprint**, the **git commit**, and `metrics.json`. Seeds are set
for Python, NumPy and (when present) PyTorch. The synthetic demo is
deterministic given its seed.

## 7. Why the synthetic demo is *not* trivially separable

The synthetic generator (`synthetic.py`) routes **all** class-discriminative
signal through a single scalar latent, `severity`, whose class-conditional
distributions deliberately **overlap**
(`control ~ N(0.25, 0.20)`, `ptsd_risk ~ N(0.70, 0.20)`). Every informative AU
channel is a noisy function of that one latent (a `label → severity →
channels` Markov chain), so no amount of feature-combining can separate the
classes better than `severity` itself does. This analytically caps ROC-AUC
(~0.93 here) and yields a realised demo AUC around **0.86** — a credible
operating point rather than a giveaway AUC = 1.0. A second, label-independent
latent (`expressiveness`) drives positive-affect bumps as pure nuisance,
further blurring the boundary.
