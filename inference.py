"""
inference.py — PTSD Micro-Expression Model Inference
=====================================================
Applies a trained PTSDMicroExpressionModel checkpoint to a new video.

Usage:
  python inference.py --checkpoint runs/fold_0/fold0-epoch10-0.8500.ckpt \
                      --video path/to/face_video.mp4 \
                      [--fps 25] [--num-frames 32] [--img-size 112]

Output (stdout + return value):
  {
    "prediction":  0 | 1 | -1,   # 0=no PTSD markers, 1=PTSD markers, -1=undetermined
    "probability": 0.73,          # sigmoid confidence (None if undetermined)
    "reason":      "ok"           # or explanation for -1
  }

ETHICAL DISCLAIMER:
  This output is a research tool only.  Predictions MUST NOT be used
  for clinical diagnosis, employment decisions, legal proceedings, or
  any high-stakes context without qualified expert review.

Dependencies:
  pip install torch torchvision pytorch-lightning opencv-python mediapipe \
              einops timm click rich
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click
import torch

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
from dataset import read_video_frames, FaceAligner, compute_difference_map
from model_ptsd import PTSDMicroExpressionModel, check_input_quality

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ImageNet stats (must match training)
_MEAN = torch.tensor([0.485, 0.456, 0.406])
_STD  = torch.tensor([0.229, 0.224, 0.225])


# ---------------------------------------------------------------------------
# Video → normalised tensor pipeline
# ---------------------------------------------------------------------------

def video_to_tensor(
    video_path: str,
    num_frames: int = 32,
    img_size:   int = 112,
    target_fps: float = 25.0,
    n_neutral:  int = 5,
    use_diff_map: bool = True,
) -> tuple[Optional[torch.Tensor], float]:
    """
    Loads and preprocesses a face video into a model-ready tensor.
    Returns (tensor (1, 3, T, H, W) float32, actual_fps).
    Returns (None, fps) if video is unreadable.
    """
    import cv2
    import numpy as np

    frames_raw, actual_fps = read_video_frames(
        video_path, target_fps=target_fps, num_frames=num_frames
    )
    if frames_raw is None:
        return None, 0.0

    aligner = FaceAligner(img_size=img_size)
    aligned = []
    miss_count = 0
    for frame_bgr in frames_raw:
        face_rgb = aligner.align(frame_bgr)
        if face_rgb is None:
            miss_count += 1
            h, w = frame_bgr.shape[:2]
            s = min(h, w)
            y0, x0 = (h - s) // 2, (w - s) // 2
            crop = frame_bgr[y0: y0 + s, x0: x0 + s]
            crop = cv2.resize(crop, (img_size, img_size))
            face_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        aligned.append(face_rgb)
    aligner.close()

    if miss_count > len(frames_raw) * 0.5:
        logger.warning(
            f"Face undetected in {miss_count}/{len(frames_raw)} frames. "
            "Result will be marked undetermined."
        )

    # Pad / trim
    while len(aligned) < num_frames:
        aligned.append(aligned[-1])
    aligned = aligned[:num_frames]

    frames = np.stack(aligned, axis=0).astype(np.float32) / 255.0  # (T,H,W,3)

    if use_diff_map:
        frames = compute_difference_map(frames, n_neutral=n_neutral)

    # (T,H,W,3) → (3,T,H,W)
    t = torch.from_numpy(frames).permute(3, 0, 1, 2).float()
    t = (t - _MEAN[:, None, None, None]) / _STD[:, None, None, None]
    return t.unsqueeze(0), actual_fps  # (1, 3, T, H, W)


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

def run_inference(
    checkpoint_path: str,
    video_path:      str,
    num_frames:      int   = 32,
    img_size:        int   = 112,
    target_fps:      float = 25.0,
    n_neutral:       int   = 5,
    use_diff_map:    bool  = True,
    device:          str   = "auto",
) -> dict:
    """
    Full inference pipeline.

    Returns dict:
        prediction  : int  — 0 | 1 | -1
        probability : float | None
        reason      : str
    """
    # ── Device ──────────────────────────────────────────────────────────────
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    logger.info(f"Running on {dev}")

    # ── Load model ──────────────────────────────────────────────────────────
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = PTSDMicroExpressionModel.load_from_checkpoint(
        checkpoint_path, map_location=dev
    )
    model.eval()
    logger.info(f"Loaded checkpoint: {checkpoint_path}")

    # ── Load video ──────────────────────────────────────────────────────────
    if not Path(video_path).exists():
        return {"prediction": -1, "probability": None, "reason": f"File not found: {video_path}"}

    tensor, actual_fps = video_to_tensor(
        video_path,
        num_frames=num_frames,
        img_size=img_size,
        target_fps=target_fps,
        n_neutral=n_neutral,
        use_diff_map=use_diff_map,
    )
    if tensor is None:
        return {"prediction": -1, "probability": None, "reason": "Cannot read video file"}

    # ── Quality guard ────────────────────────────────────────────────────────
    result = model.predict_with_guard(
        video=tensor.squeeze(0).to(dev),  # (3, T, H, W)
        fps=actual_fps,
    )

    logger.info(
        f"Result → prediction={result['prediction']}  "
        f"prob={result['probability']}  reason={result['reason']}"
    )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--checkpoint", "-c", required=True,
              help="Path to .ckpt checkpoint file.")
@click.option("--video",      "-v", required=True,
              help="Path to the face video file.")
@click.option("--fps",        default=25.0,  show_default=True,
              help="Minimum expected FPS; lower values trigger interpolation.")
@click.option("--num-frames", default=32,    show_default=True,
              help="Number of frames to sample from the video.")
@click.option("--img-size",   default=112,   show_default=True,
              help="Spatial resolution for face crops (pixels).")
@click.option("--device",     default="auto", show_default=True,
              help="Compute device: auto | cpu | cuda | cuda:0 …")
@click.option("--no-diff-map", is_flag=True, default=False,
              help="Disable neutral-face subtraction (diff maps).")
def cli(checkpoint, video, fps, num_frames, img_size, device, no_diff_map):
    """
    PTSD micro-expression inference.

    \b
    Exit codes:
      0  — prediction produced (check 'prediction' field)
      1  — fatal error (missing file, import error, etc.)
    """
    try:
        result = run_inference(
            checkpoint_path=checkpoint,
            video_path=video,
            num_frames=num_frames,
            img_size=img_size,
            target_fps=fps,
            use_diff_map=not no_diff_map,
            device=device,
        )
        print(json.dumps(result, indent=2))

        # Non-zero exit for undetermined to allow shell scripting
        if result["prediction"] == -1:
            sys.exit(2)

    except Exception as exc:
        logger.exception(f"Inference failed: {exc}")
        print(json.dumps({"prediction": -1, "probability": None, "reason": str(exc)}))
        sys.exit(1)


if __name__ == "__main__":
    cli()
