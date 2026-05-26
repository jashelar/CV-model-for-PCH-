"""
train.py — PTSD Micro-Expression Model Training
================================================
Features:
  - Configuration via dataclass (no argparse hell)
  - Patient-level K-Fold cross-validation
  - Reproducibility: seed everything
  - Early stopping on val/auroc
  - Checkpoint best model per fold
  - TensorBoard + CSV logging

Usage:
  python train.py

Dependencies:
  pip install pytorch-lightning scikit-learn pandas tensorboard

Manifest format (data/manifest.csv):
  video_path,label,patient_id,split,gender,race
"""

import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from sklearn.model_selection import GroupKFold
import torch

from dataset import PTSDDataModule, PTSDVideoDataset
from model_ptsd import PTSDMicroExpressionModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # ── Data ────────────────────────────────────────────────────────────────
    manifest_path:  str   = "data/manifest.csv"
    output_dir:     str   = "runs"
    num_frames:     int   = 32
    img_size:       int   = 112
    target_fps:     float = 25.0
    n_neutral:      int   = 5
    use_diff_map:   bool  = True
    demo_attr_col:  Optional[str] = "gender"   # column for fairness metric

    # ── Model ───────────────────────────────────────────────────────────────
    d_model:              int   = 256
    vit_depth:            int   = 4
    vit_heads:            int   = 8
    focal_gamma:          float = 2.0
    focal_alpha:          float = 0.25
    contrastive_weight:   float = 0.3
    contrastive_temp:     float = 0.07
    dropout:              float = 0.1

    # ── Optimiser ───────────────────────────────────────────────────────────
    lr:             float = 3e-4
    weight_decay:   float = 1e-4
    warmup_epochs:  int   = 5
    max_epochs:     int   = 50

    # ── Training ────────────────────────────────────────────────────────────
    batch_size:     int   = 8
    num_workers:    int   = 4
    n_folds:        int   = 5        # patient-level K-fold
    seed:           int   = 42
    precision:      str   = "16-mixed"  # "32" on CPU
    gradient_clip:  float = 1.0
    accumulate_grad_batches: int = 2

    # ── Callbacks ───────────────────────────────────────────────────────────
    patience:       int   = 10
    monitor_metric: str   = "val/auroc"

    # ── Hardware ────────────────────────────────────────────────────────────
    accelerator:    str   = "auto"
    devices:        int   = 1


# ---------------------------------------------------------------------------
# Manifest preparation: add cross-val fold column
# ---------------------------------------------------------------------------

def prepare_cv_manifest(
    manifest_path: str,
    n_folds: int,
    output_path: str,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Adds 'cv_fold' column to manifest using GroupKFold on patient_id.
    Writes updated manifest to output_path.
    """
    df = pd.read_csv(manifest_path)
    if "patient_id" not in df.columns:
        raise ValueError("manifest.csv must have 'patient_id' column")

    gkf  = GroupKFold(n_splits=n_folds)
    df["cv_fold"] = -1
    groups = df["patient_id"].values
    for fold_idx, (_, val_idx) in enumerate(
        gkf.split(df, groups=groups)
    ):
        df.loc[val_idx, "cv_fold"] = fold_idx

    df.to_csv(output_path, index=False)
    logger.info(f"CV manifest written → {output_path}")
    return df


def split_manifest_for_fold(
    df: pd.DataFrame,
    fold: int,
    cv_manifest_path: str,
) -> str:
    """
    Creates a temporary manifest for a given fold:
      val/test = fold, train = all other folds.
    Returns path to the temp manifest.
    """
    tmp = df.copy()
    tmp["split"] = tmp["cv_fold"].apply(
        lambda x: "val" if x == fold else "train"
    )
    path = cv_manifest_path.replace(".csv", f"_fold{fold}.csv")
    tmp.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Build trainer
# ---------------------------------------------------------------------------

def build_trainer(cfg: TrainConfig, fold: int, fold_dir: str) -> pl.Trainer:
    callbacks = [
        ModelCheckpoint(
            dirpath=fold_dir,
            filename=f"fold{fold}-{{epoch:02d}}-{{val/auroc:.4f}}",
            monitor=cfg.monitor_metric,
            mode="max",
            save_top_k=1,
            save_last=True,
        ),
        EarlyStopping(
            monitor=cfg.monitor_metric,
            patience=cfg.patience,
            mode="max",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        RichProgressBar(),
    ]

    loggers = [
        TensorBoardLogger(save_dir=fold_dir, name="tb"),
        CSVLogger(save_dir=fold_dir, name="csv"),
    ]

    return pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        precision=cfg.precision,
        gradient_clip_val=cfg.gradient_clip,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=5,
        deterministic=True,
    )


# ---------------------------------------------------------------------------
# Single fold training
# ---------------------------------------------------------------------------

def train_fold(cfg: TrainConfig, fold_manifest: str, fold: int) -> dict:
    fold_dir = str(Path(cfg.output_dir) / f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    model = PTSDMicroExpressionModel(
        img_size=cfg.img_size,
        num_frames=cfg.num_frames,
        d_model=cfg.d_model,
        vit_depth=cfg.vit_depth,
        vit_heads=cfg.vit_heads,
        focal_gamma=cfg.focal_gamma,
        focal_alpha=cfg.focal_alpha,
        contrastive_weight=cfg.contrastive_weight,
        contrastive_temp=cfg.contrastive_temp,
        lr=cfg.lr,
        warmup_epochs=cfg.warmup_epochs,
        max_epochs=cfg.max_epochs,
        weight_decay=cfg.weight_decay,
        dropout=cfg.dropout,
    )

    dm = PTSDDataModule(
        manifest_path=fold_manifest,
        num_frames=cfg.num_frames,
        img_size=cfg.img_size,
        target_fps=cfg.target_fps,
        n_neutral=cfg.n_neutral,
        use_diff_map=cfg.use_diff_map,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        demo_attr_col=cfg.demo_attr_col,
    )

    trainer = build_trainer(cfg, fold, fold_dir)
    trainer.fit(model, datamodule=dm)
    results = trainer.test(model, datamodule=dm, ckpt_path="best", verbose=True)
    logger.info(f"Fold {fold} test results: {results}")
    return results[0] if results else {}


# ---------------------------------------------------------------------------
# Main cross-validation loop
# ---------------------------------------------------------------------------

def run_cv(cfg: TrainConfig):
    pl.seed_everything(cfg.seed, workers=True)
    os.makedirs(cfg.output_dir, exist_ok=True)

    cv_manifest = str(Path(cfg.output_dir) / "manifest_cv.csv")
    df = prepare_cv_manifest(cfg.manifest_path, cfg.n_folds, cv_manifest, cfg.seed)

    fold_results: List[dict] = []
    for fold in range(cfg.n_folds):
        logger.info(f"\n{'='*60}\n  FOLD {fold + 1} / {cfg.n_folds}\n{'='*60}")
        fold_manifest = split_manifest_for_fold(df, fold, cv_manifest)
        result = train_fold(cfg, fold_manifest, fold)
        fold_results.append(result)

    # Aggregate across folds
    agg: dict = {}
    for key in fold_results[0]:
        vals = [r[key] for r in fold_results if key in r]
        agg[key] = {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals)),
        }

    summary_path = str(Path(cfg.output_dir) / "cv_summary.json")
    import json
    with open(summary_path, "w") as f:
        json.dump(agg, f, indent=2)

    logger.info(f"\nCross-Validation Summary:\n{json.dumps(agg, indent=2)}")
    logger.info(f"Summary saved → {summary_path}")
    return agg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = TrainConfig()

    # Override for quick smoke-test
    # cfg.max_epochs    = 2
    # cfg.n_folds       = 2
    # cfg.batch_size    = 2
    # cfg.num_workers   = 0
    # cfg.precision     = "32"

    logger.info("Training config:")
    for k, v in cfg.__dict__.items():
        logger.info(f"  {k:30s} = {v}")

    results = run_cv(cfg)
    logger.info("Training complete.")
