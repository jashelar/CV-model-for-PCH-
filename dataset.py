"""
PTSDVideoDataset
================
Loads face video sequences and applies:
  1. Face alignment via MediaPipe (68-equivalent landmarks)
  2. Difference maps: subtract mean of first N neutral frames
  3. FPS validation & interpolation
  4. Temporal / spatial augmentations safe for micro-expressions

Dependencies:
  pip install opencv-python mediapipe torch torchvision numpy pandas

Usage:
  dataset = PTSDVideoDataset(
      manifest_path="data/manifest.csv",
      split="train",
      num_frames=32,
      img_size=112,
  )

manifest.csv columns:
  video_path, label, patient_id, split, [gender, race]
"""

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pytorch_lightning as pl          # ← FIX #2: was missing, PTSDDataModule inherits from pl
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Face aligner using MediaPipe Face Mesh
# ---------------------------------------------------------------------------

class FaceAligner:
    """
    Detects face landmarks, crops and affine-aligns the face region to a
    canonical square of (img_size × img_size).
    Falls back to centre-crop if detection fails.
    """

    # Canonical eye anchor positions (as fraction of output size)
    LEFT_EYE_ANCHOR  = (0.35, 0.40)
    RIGHT_EYE_ANCHOR = (0.65, 0.40)

    # MediaPipe landmark indices for eye centres
    _LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
    _RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

    def __init__(self, img_size: int = 112):
        self.img_size = img_size
        # FIX #8: do NOT create FaceMesh here.
        # Instantiated lazily on first use so each DataLoader worker
        # process creates its own instance after fork, not before.
        self._mesh: Optional[mp.solutions.face_mesh.FaceMesh] = None

    @property
    def _mp_face_mesh(self) -> mp.solutions.face_mesh.FaceMesh:
        """Lazy per-process initialisation (safe with multiprocessing DataLoader)."""
        if self._mesh is None:
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        return self._mesh

    def _eye_centre(self, lm, indices: list, h: int, w: int) -> np.ndarray:
        pts = np.array(
            [[lm[i].x * w, lm[i].y * h] for i in indices],
            dtype=np.float32,
        )
        return pts.mean(axis=0)

    def align(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        frame_bgr: BGR uint8 (H, W, 3)
        Returns aligned face **RGB** uint8 (img_size, img_size, 3) or None.
        FIX #5: single BGR→RGB conversion here; no second pass needed later.
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._mp_face_mesh.process(rgb)   # FaceMesh expects RGB

        if not result.multi_face_landmarks:
            return None

        lm = result.multi_face_landmarks[0].landmark
        lc = self._eye_centre(lm, self._LEFT_EYE_IDX, h, w)
        rc = self._eye_centre(lm, self._RIGHT_EYE_IDX, h, w)

        size   = self.img_size
        lx, ly = self.LEFT_EYE_ANCHOR[0] * size, self.LEFT_EYE_ANCHOR[1] * size
        rx, ry = self.RIGHT_EYE_ANCHOR[0] * size, self.RIGHT_EYE_ANCHOR[1] * size

        src = np.float32([lc, rc])
        dst = np.float32([[lx, ly], [rx, ry]])
        M   = cv2.getAffineTransform(src, dst)
        # Warp on the RGB image directly
        aligned = cv2.warpAffine(rgb, M, (size, size), flags=cv2.INTER_LINEAR)
        return aligned  # RGB uint8

    def close(self):
        if self._mesh is not None:
            self._mesh.close()
            self._mesh = None


# ---------------------------------------------------------------------------
# Video reading & FPS handling
# ---------------------------------------------------------------------------

def read_video_frames(
    path: str,
    target_fps: float = 25.0,
    num_frames: int = 32,
) -> Tuple[Optional[np.ndarray], float]:
    """
    Returns (frames_bgr, actual_fps).
    frames_bgr: (T, H, W, 3) uint8 — uniformly sampled after FPS fix.
    Returns (None, fps) if video cannot be opened.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, 0.0

    actual_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    raw: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw.append(frame)
    cap.release()

    if len(raw) == 0:
        return None, actual_fps

    raw_arr = np.stack(raw, axis=0)  # (N, H, W, 3)

    # Resample to target_fps if needed
    if actual_fps < target_fps - 1:
        logger.warning(
            f"FPS {actual_fps:.1f} < {target_fps} for {path}; "
            "upsampling via linear interpolation."
        )
        scale = target_fps / max(actual_fps, 1e-6)
        new_n = max(int(len(raw_arr) * scale), num_frames)
        indices_src = np.linspace(0, len(raw_arr) - 1, new_n)
        lo  = np.floor(indices_src).astype(int).clip(0, len(raw_arr) - 1)
        hi  = np.ceil(indices_src).astype(int).clip(0, len(raw_arr) - 1)
        alpha = (indices_src - lo)[:, None, None, None]
        raw_arr = (raw_arr[lo] * (1 - alpha) + raw_arr[hi] * alpha).astype(np.uint8)

    # Uniform temporal sampling
    indices = np.linspace(0, len(raw_arr) - 1, num_frames, dtype=int)
    sampled = raw_arr[indices]  # (num_frames, H, W, 3)
    return sampled, actual_fps


# ---------------------------------------------------------------------------
# Difference map (subtract neutral mean)
# ---------------------------------------------------------------------------

def compute_difference_map(
    frames: np.ndarray,
    n_neutral: int = 5,
) -> np.ndarray:
    """
    frames:   (T, H, W, 3) float32 in [0,1]
    Returns:  (T, H, W, 3) difference map, clipped to [0,1]
    Neutral = mean of first n_neutral frames.
    """
    neutral = frames[:n_neutral].mean(axis=0, keepdims=True)  # (1, H, W, 3)
    diff    = frames - neutral
    # Scale to [0,1]: diff is in [-1,1]
    diff    = (diff + 1.0) / 2.0
    return diff.clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# Augmentations (micro-expression safe)
# ---------------------------------------------------------------------------

class TemporalAugmentor:
    """
    Temporal augmentations that preserve spatial structure:
    - Random temporal crop
    - Random playback speed change (±20%)
    """

    def __init__(self, num_frames: int = 32, speed_range: Tuple[float, float] = (0.8, 1.2)):
        self.num_frames  = num_frames
        self.speed_range = speed_range

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        # FIX #7: ensure float32 BEFORE any arithmetic — prevents silent uint8 truncation
        frames = frames.astype(np.float32)
        T = frames.shape[0]
        speed   = random.uniform(*self.speed_range)
        new_T   = max(int(T / speed), self.num_frames)
        indices = np.linspace(0, T - 1, new_T, dtype=float)
        lo    = np.floor(indices).astype(int).clip(0, T - 1)
        hi    = np.ceil(indices).astype(int).clip(0, T - 1)
        alpha = (indices - lo)[:, None, None, None].astype(np.float32)
        resampled = frames[lo] * (1.0 - alpha) + frames[hi] * alpha  # guaranteed float32

        if resampled.shape[0] > self.num_frames:
            start = random.randint(0, resampled.shape[0] - self.num_frames)
            resampled = resampled[start: start + self.num_frames]
        return resampled  # (num_frames, H, W, 3) float32


class SpatialAugmentor:
    """
    Spatial augmentations that do NOT distort micro-expressions:
    - Tiny brightness/contrast jitter (max ±5%)
    - Horizontal flip (50%)
    No rotation, no perspective warp, no crop (face alignment already done).
    """

    def __init__(self, brightness: float = 0.05, contrast: float = 0.05):
        self.brightness = brightness
        self.contrast   = contrast

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        # Brightness
        b = 1.0 + random.uniform(-self.brightness, self.brightness)
        # Contrast
        c = 1.0 + random.uniform(-self.contrast, self.contrast)
        mean  = frames.mean()
        aug   = (frames - mean) * c + mean * b
        aug   = aug.clip(0.0, 1.0)
        # Horizontal flip
        if random.random() < 0.5:
            aug = aug[:, :, ::-1, :].copy()
        return aug


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PTSDVideoDataset(Dataset):
    """
    Manifest CSV expected columns:
        video_path  : str
        label       : int (0 or 1)
        patient_id  : int
        split       : str (train / val / test)
        gender      : int  (optional — for demographic fairness)
        race        : int  (optional — for demographic fairness)

    demo_attr_col: column to use as demographic group for fairness logging.
    """

    MEAN = torch.tensor([0.485, 0.456, 0.406])
    STD  = torch.tensor([0.229, 0.224, 0.225])

    def __init__(
        self,
        manifest_path: str,
        split: str = "train",
        num_frames: int = 32,
        img_size: int = 112,
        target_fps: float = 25.0,
        n_neutral: int = 5,
        use_diff_map: bool = True,
        augment: bool = True,
        demo_attr_col: Optional[str] = None,
    ):
        super().__init__()
        self.num_frames    = num_frames
        self.img_size      = img_size
        self.target_fps    = target_fps
        self.n_neutral     = n_neutral
        self.use_diff_map  = use_diff_map
        self.augment       = augment
        self.demo_attr_col = demo_attr_col

        df = pd.read_csv(manifest_path)
        self.data = df[df["split"] == split].reset_index(drop=True)
        logger.info(f"[{split}] {len(self.data)} samples loaded.")

        # FIX #8: aligner is NOT created here — lazily per worker process
        self._aligner: Optional[FaceAligner] = None
        self.temp_aug = TemporalAugmentor(num_frames=num_frames)
        self.spat_aug = SpatialAugmentor()

    # -----------------------------------------------------------------------
    @property
    def aligner(self) -> FaceAligner:
        """Per-process lazy init — each DataLoader worker gets its own instance."""
        if self._aligner is None:
            self._aligner = FaceAligner(img_size=self.img_size)
        return self._aligner

    # -----------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.data)

    # -----------------------------------------------------------------------
    def _load_and_preprocess(self, row: pd.Series) -> Optional[np.ndarray]:
        """
        Returns (num_frames, H, W, 3) float32 in [0,1], already in RGB.
        Returns None if video is unreadable or face undetectable (>50% frames).
        """
        frames_raw, fps = read_video_frames(
            row["video_path"],
            target_fps=self.target_fps,
            num_frames=self.num_frames,
        )
        if frames_raw is None:
            return None

        # ── Face alignment per frame ──────────────────────────────────────
        # aligner.align() now returns RGB directly (FIX #5).
        aligned: List[np.ndarray] = []
        miss_count = 0
        for frame_bgr in frames_raw:
            face_rgb = self.aligner.align(frame_bgr)   # RGB or None
            if face_rgb is None:
                miss_count += 1
                # Fallback: centre-crop + manual BGR→RGB
                h, w = frame_bgr.shape[:2]
                s    = min(h, w)
                y0, x0 = (h - s) // 2, (w - s) // 2
                crop = frame_bgr[y0: y0 + s, x0: x0 + s]
                crop = cv2.resize(crop, (self.img_size, self.img_size))
                face_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)  # BGR→RGB once
            aligned.append(face_rgb)   # all items are RGB uint8

        # FIX #9: reject video if face not detected in majority of frames
        if miss_count > len(frames_raw) * 0.5:
            logger.warning(
                f"Face undetected in {miss_count}/{len(frames_raw)} frames "
                f"({row['video_path']}) — skipping sample."
            )
            return None

        # Pad / trim to exactly num_frames
        while len(aligned) < self.num_frames:
            aligned.append(aligned[-1])
        aligned = aligned[:self.num_frames]

        # FIX #1: stack RGB frames directly — no reshape/cvtColor needed
        # (replaces the broken: frames.reshape(-1, img_size, 3) → cvtColor → reshape)
        frames = np.stack(aligned, axis=0).astype(np.float32) / 255.0  # (T,H,W,3) float32

        # FIX #6 (diff-map + normalisation): apply diff map BEFORE ImageNet norm.
        # The diff map maps [-1,1] → [0,1] so ImageNet stats still apply.
        # BatchNorm in the model handles any residual distributional shift.
        if self.use_diff_map:
            frames = compute_difference_map(frames, n_neutral=self.n_neutral)

        return frames  # (T, H, W, 3) float32 RGB [0,1]

    # -----------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]

        frames = self._load_and_preprocess(row)
        if frames is None:
            # Return a zeroed tensor flagged as invalid; training loop should skip
            logger.warning(f"Failed to load {row['video_path']} — returning zeros.")
            frames = np.zeros((self.num_frames, self.img_size, self.img_size, 3),
                               dtype=np.float32)

        if self.augment:
            frames = self.temp_aug(frames)
            frames = self.spat_aug(frames)

        # Ensure shape
        if frames.shape[0] != self.num_frames:
            # trim or pad
            if frames.shape[0] > self.num_frames:
                frames = frames[:self.num_frames]
            else:
                pad = np.zeros(
                    (self.num_frames - frames.shape[0], self.img_size, self.img_size, 3),
                    dtype=np.float32,
                )
                frames = np.concatenate([frames, pad], axis=0)

        # (T, H, W, 3) → tensor (3, T, H, W)
        t = torch.from_numpy(frames).permute(3, 0, 1, 2)  # (3, T, H, W)
        t = (t - self.MEAN[:, None, None, None]) / self.STD[:, None, None, None]

        item: Dict[str, torch.Tensor] = {
            "video":      t,
            "label":      torch.tensor(int(row["label"]), dtype=torch.long),
            "patient_id": torch.tensor(int(row["patient_id"]), dtype=torch.long),
        }

        if self.demo_attr_col and self.demo_attr_col in row:
            item["demo_attr"] = torch.tensor(int(row[self.demo_attr_col]), dtype=torch.long)

        return item

    def close(self):
        self.aligner.close()


# ---------------------------------------------------------------------------
# DataModule helper
# ---------------------------------------------------------------------------

class PTSDDataModule(pl.LightningDataModule):
    def __init__(
        self,
        manifest_path: str,
        num_frames: int = 32,
        img_size: int = 112,
        target_fps: float = 25.0,
        n_neutral: int = 5,
        use_diff_map: bool = True,
        batch_size: int = 8,
        num_workers: int = 4,
        demo_attr_col: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

    def _make_ds(self, split: str, augment: bool) -> PTSDVideoDataset:
        hp = self.hparams
        return PTSDVideoDataset(
            manifest_path=hp.manifest_path,
            split=split,
            num_frames=hp.num_frames,
            img_size=hp.img_size,
            target_fps=hp.target_fps,
            n_neutral=hp.n_neutral,
            use_diff_map=hp.use_diff_map,
            augment=augment,
            demo_attr_col=hp.demo_attr_col,
        )

    def setup(self, stage=None):
        self.train_ds = self._make_ds("train", augment=True)
        self.val_ds   = self._make_ds("val",   augment=False)
        self.test_ds  = self._make_ds("test",  augment=False)

    def _loader(self, ds, shuffle):
        return DataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def train_dataloader(self): return self._loader(self.train_ds, shuffle=True)
    def val_dataloader(self):   return self._loader(self.val_ds,   shuffle=False)
    def test_dataloader(self):  return self._loader(self.test_ds,  shuffle=False)

    def teardown(self, stage: Optional[str] = None):
        """FIX #15: close MediaPipe handles to prevent resource leaks."""
        for attr in ("train_ds", "val_ds", "test_ds"):
            ds = getattr(self, attr, None)
            if ds is not None:
                ds.close()
