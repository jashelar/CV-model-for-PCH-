"""
PTSD Micro-Expression Detection Model
======================================
Dual-stream architecture:
  Stream 1: (2+1)D CNN with Temporal Shift Module (TSM)
  Stream 2: Video Vision Transformer (ViT) with 8x8 patch size
  Fusion:   Cross-attention + binary classification head

Dependencies (install before use):
  pip install torch torchvision pytorch-lightning einops timm
  pip install torchmetrics scikit-learn opencv-python mediapipe

Author: Generated for research purposes only.
ETHICAL NOTE: This system must never be used as a standalone clinical
diagnostic tool. All predictions require human expert review.
"""

import math
import logging
from typing import Optional, Dict, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC, F1Score
from torchmetrics.classification import BinaryConfusionMatrix

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Temporal Shift Module (TSM)
# ---------------------------------------------------------------------------

class TemporalShiftModule(nn.Module):
    """
    TSM: shifts a fraction of channels along the temporal dimension
    in-place, without adding parameters.
    fold_div=8 → shifts 1/8 channels forward, 1/8 backward.
    """

    def __init__(self, fold_div: int = 8):
        super().__init__()
        self.fold_div = fold_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        fold = C // self.fold_div
        out = x.clone()
        # shift forward (past → present)
        out[:, :fold, 1:, :, :] = x[:, :fold, :-1, :, :]
        out[:, :fold, 0, :, :] = 0.0
        # shift backward (future → present)
        out[:, fold:2 * fold, :-1, :, :] = x[:, fold:2 * fold, 1:, :, :]
        out[:, fold:2 * fold, -1, :, :] = 0.0
        return out


# ---------------------------------------------------------------------------
# 2. Stream 1 — (2+1)D CNN with TSM
# ---------------------------------------------------------------------------

class R2Plus1DBlock(nn.Module):
    """
    Factorized (2+1)D residual block with optional TSM injected before
    the spatial convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_tsm: bool = True,
    ):
        super().__init__()
        self.tsm = TemporalShiftModule() if use_tsm else nn.Identity()

        # FIX #12: original formula was a theoretically-motivated parameter-budget
        # estimate but produced near-zero values for small channel counts.
        # Replaced with the standard (2+1)D convention: spatial bottleneck at
        # out_channels//2, minimum 16 to keep representational capacity.
        mid_channels = max(out_channels // 2, 16)

        # Spatial 2D part
        self.spatial_conv = nn.Conv3d(
            in_channels, mid_channels,
            kernel_size=(1, 3, 3), stride=(1, stride, stride),
            padding=(0, 1, 1), bias=False,
        )
        self.bn1 = nn.BatchNorm3d(mid_channels)

        # Temporal 1D part
        self.temporal_conv = nn.Conv3d(
            mid_channels, out_channels,
            kernel_size=(3, 1, 1), stride=(stride if stride > 1 else 1, 1, 1),
            padding=(1, 0, 0), bias=False,
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample: Optional[nn.Module] = None
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv3d(
                    in_channels, out_channels,
                    kernel_size=1,
                    stride=(stride, stride, stride),
                    bias=False,
                ),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.tsm(x)
        out = self.relu(self.bn1(self.spatial_conv(out)))
        out = self.bn2(self.temporal_conv(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class CNNStream(nn.Module):
    """
    Lightweight (2+1)D CNN backbone producing a temporal feature sequence.
    Output: (B, T', d_cnn)
    """

    def __init__(self, d_out: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                      padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        self.layer1 = R2Plus1DBlock(32, 64, stride=1, use_tsm=True)
        self.layer2 = R2Plus1DBlock(64, 128, stride=2, use_tsm=True)
        self.layer3 = R2Plus1DBlock(128, d_out, stride=2, use_tsm=True)
        self.pool_spatial = nn.AdaptiveAvgPool3d((None, 1, 1))  # keep T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, T, H, W)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool_spatial(x)         # (B, d_out, T', 1, 1)
        x = x.squeeze(-1).squeeze(-1)    # (B, d_out, T')
        x = x.permute(0, 2, 1)           # (B, T', d_out)
        return x


# ---------------------------------------------------------------------------
# 3. Stream 2 — Video ViT with 8×8 patches
# ---------------------------------------------------------------------------

class PatchEmbed3D(nn.Module):
    """
    Tubelet embedding: patch_size=8×8 spatial, temporal_patch=2 frames.
    """

    def __init__(
        self,
        img_size: int = 112,
        patch_size: int = 8,
        temporal_patch: int = 2,
        in_chans: int = 3,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.num_patches_h = img_size // patch_size
        self.num_patches_w = img_size // patch_size
        self.temporal_patch = temporal_patch
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(temporal_patch, patch_size, patch_size),
            stride=(temporal_patch, patch_size, patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        x = self.proj(x)               # (B, embed_dim, T//tp, H//ps, W//ps)
        B, D, t, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, t*h*w, D)
        return x


class VideoViT(nn.Module):
    """
    Lightweight Video ViT for spatial micro-detail capture.
    Output: (B, N_tokens, d_vit)
    """

    def __init__(
        self,
        img_size: int = 112,
        num_frames: int = 32,
        patch_size: int = 8,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            temporal_patch=2,
            embed_dim=embed_dim,
        )
        n_spatial = (img_size // patch_size) ** 2
        n_temporal = num_frames // 2
        n_tokens = n_spatial * n_temporal
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x)
        # handle varying sequence length (pad pos_embed)
        n = tokens.shape[1]
        pe = self.pos_embed[:, :n, :]
        tokens = tokens + pe
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)
        return tokens  # (B, N, d_vit)


# ---------------------------------------------------------------------------
# 4. Cross-Attention Fusion
# ---------------------------------------------------------------------------

class CrossAttentionFusion(nn.Module):
    """
    Query: CNN temporal features  →  (B, T', d)
    Key/Value: ViT spatial tokens →  (B, N, d)
    Output: fused (B, T', d)
    """

    def __init__(self, d_model: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        cnn_feat: torch.Tensor,   # (B, T', d)
        vit_tokens: torch.Tensor,  # (B, N, d)
    ) -> torch.Tensor:
        # Cross-attention: CNN queries ViT
        residual = cnn_feat
        attended, _ = self.attn(
            query=self.norm1(cnn_feat),
            key=vit_tokens,
            value=vit_tokens,
        )
        x = residual + attended
        x = x + self.ffn(self.norm2(x))
        return x  # (B, T', d)


# ---------------------------------------------------------------------------
# 5. Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # FIX #11: clamp to prevent exp(-BCE) underflow / overflow at |logit| >> 10
        logits = torch.clamp(logits, -10.0, 10.0)
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        pt = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal = alpha_t * (1 - pt) ** self.gamma * bce
        return focal.mean()


# ---------------------------------------------------------------------------
# 6. Contrastive Loss (same/different patient pairs)
# ---------------------------------------------------------------------------

class ContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss: same-patient embeddings pulled together,
    different-patient embeddings pushed apart.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,  # (B, d) L2-normalised
        patient_ids: torch.Tensor,  # (B,) int
    ) -> torch.Tensor:
        B = embeddings.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=embeddings.device)

        sim = torch.matmul(embeddings, embeddings.T) / self.temperature  # (B, B)
        # mask diagonal
        mask_self = torch.eye(B, dtype=torch.bool, device=embeddings.device)
        sim.masked_fill_(mask_self, float("-inf"))

        # positive mask: same patient (but not self)
        pid_eq = patient_ids.unsqueeze(1) == patient_ids.unsqueeze(0)  # (B, B)
        pid_eq.masked_fill_(mask_self, False)

        if not pid_eq.any():
            return torch.tensor(0.0, device=embeddings.device)

        log_prob = F.log_softmax(sim, dim=1)
        # mean over positives
        loss = -(log_prob * pid_eq.float()).sum(1) / pid_eq.float().sum(1).clamp(min=1)
        return loss.mean()


# ---------------------------------------------------------------------------
# 7. Fairness Metric — Demographic Parity Gap
# ---------------------------------------------------------------------------

class DemographicParityGap:
    """
    Tracks predicted positive rate per demographic group.
    Logs the max gap across groups as 'demographic_parity_gap'.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._groups: Dict[Any, list] = {}

    def update(self, preds: torch.Tensor, demo_labels: torch.Tensor):
        """
        preds:       (B,) float in [0,1]
        demo_labels: (B,) int  — e.g. encoded race/gender group
        """
        for p, d in zip(preds.cpu().tolist(), demo_labels.cpu().tolist()):
            self._groups.setdefault(d, []).append(float(p))

    def compute(self) -> float:
        if len(self._groups) < 2:
            return 0.0
        rates = [sum(v) / len(v) for v in self._groups.values()]
        return max(rates) - min(rates)


# ---------------------------------------------------------------------------
# 8. Input Quality Guard
# ---------------------------------------------------------------------------

def check_input_quality(
    video: torch.Tensor,
    fps: float,
    min_fps: float = 25.0,
    static_threshold: int = 50,
) -> Tuple[bool, str]:
    """
    Returns (is_valid, reason).
    - video: (3, T, H, W)  normalised tensor
    - fps: frames per second of original video
    """
    if fps < min_fps:
        return False, f"fps={fps:.1f} < {min_fps} — interpolate or reject"

    T = video.shape[1]
    if T < 15:
        return False, "Too few frames (<15)"

    # Check for static/frozen video (micro-motion proxy)
    diffs = (video[:, 1:] - video[:, :-1]).abs().mean(dim=(0, 2, 3))  # (T-1,)
    static_frames = (diffs < 1e-4).sum().item()
    if static_frames > static_threshold:
        return False, f"Too many static frames ({static_frames}>{static_threshold})"

    return True, "ok"


# ---------------------------------------------------------------------------
# 9. Main Lightning Module
# ---------------------------------------------------------------------------

class PTSDMicroExpressionModel(pl.LightningModule):
    """
    Dual-stream PTSD micro-expression classifier.

    Inputs per batch:
        video      : (B, 3, T, H, W)  float32, normalised
        label      : (B,)              int64  {0, 1}
        patient_id : (B,)              int64  (for contrastive loss)
        demo_attr  : (B,)              int64  (optional demographic group)

    Outputs:
        logit      : (B,) float32
    """

    def __init__(
        self,
        img_size: int = 112,
        num_frames: int = 32,
        d_model: int = 256,
        vit_depth: int = 4,
        vit_heads: int = 8,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        contrastive_weight: float = 0.3,
        contrastive_temp: float = 0.07,
        lr: float = 3e-4,
        warmup_epochs: int = 5,
        max_epochs: int = 50,
        weight_decay: float = 1e-4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        # --- Streams ---
        self.cnn_stream = CNNStream(d_out=d_model)
        self.vit_stream = VideoViT(
            img_size=img_size,
            num_frames=num_frames,
            patch_size=8,
            embed_dim=d_model,
            depth=vit_depth,
            num_heads=vit_heads,
            dropout=dropout,
        )

        # --- Projection to common dim if needed ---
        # (both already output d_model, so identity; kept for extensibility)
        self.cnn_proj = nn.Identity()
        self.vit_proj = nn.Identity()

        # --- Fusion ---
        self.fusion = CrossAttentionFusion(d_model=d_model, num_heads=vit_heads, dropout=dropout)

        # --- Classification head ---
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        # --- Losses ---
        self.focal_loss = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        self.contrastive_loss = ContrastiveLoss(temperature=contrastive_temp)

        # --- Metrics ---
        for split in ("train", "val", "test"):
            setattr(self, f"{split}_acc",   Accuracy(task="binary"))
            setattr(self, f"{split}_auroc", AUROC(task="binary"))
            setattr(self, f"{split}_f1",    F1Score(task="binary"))

        self.val_cm = BinaryConfusionMatrix()
        self.dp_gap = DemographicParityGap()

    # -----------------------------------------------------------------------
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Returns logit (B,)."""
        cnn_feat  = self.cnn_proj(self.cnn_stream(video))    # (B, T', d)
        vit_feat  = self.vit_proj(self.vit_stream(video))    # (B, N, d)
        fused     = self.fusion(cnn_feat, vit_feat)          # (B, T', d)
        pooled    = fused.mean(dim=1)                        # (B, d) — temporal avg
        logit     = self.classifier(pooled).squeeze(-1)      # (B,)
        return logit

    # -----------------------------------------------------------------------
    def _shared_step(
        self,
        batch: Dict[str, torch.Tensor],
        split: str,
    ) -> torch.Tensor:
        video      = batch["video"]
        labels     = batch["label"]
        patient_id = batch.get("patient_id", torch.zeros(video.shape[0], dtype=torch.long))
        demo_attr  = batch.get("demo_attr",  None)

        logit = self(video)
        prob  = torch.sigmoid(logit)

        # Losses
        fl = self.focal_loss(logit, labels)

        # Normalised embedding for contrastive loss
        embed = F.normalize(
            self.cnn_stream(video).mean(dim=1),  # (B, d)
            dim=-1,
        )
        cl = self.contrastive_loss(embed, patient_id.to(video.device))
        loss = fl + self.hparams.contrastive_weight * cl

        # Metrics
        acc   = getattr(self, f"{split}_acc")
        auroc = getattr(self, f"{split}_auroc")
        f1    = getattr(self, f"{split}_f1")
        acc(prob,  labels)
        auroc(prob, labels)
        f1(prob,   labels)

        self.log(f"{split}/loss",         loss,        prog_bar=True)
        self.log(f"{split}/focal_loss",   fl)
        self.log(f"{split}/contrastive",  cl)
        self.log(f"{split}/acc",          acc,         prog_bar=True)
        self.log(f"{split}/auroc",        auroc)
        self.log(f"{split}/f1",           f1)

        if split == "val":
            self.val_cm(prob, labels)
            if demo_attr is not None:
                self.dp_gap.update(prob.detach(), demo_attr)

        return loss

    def training_step(self, batch, _):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, _):
        return self._shared_step(batch, "val")

    def test_step(self, batch, _):
        return self._shared_step(batch, "test")

    # -----------------------------------------------------------------------
    def on_validation_epoch_end(self):
        cm = self.val_cm.compute()
        tn, fp, fn, tp = cm.flatten().tolist()
        self.log("val/tn", tn); self.log("val/fp", fp)
        self.log("val/fn", fn); self.log("val/tp", tp)
        self.val_cm.reset()

        gap = self.dp_gap.compute()
        self.log("val/demographic_parity_gap", gap)
        if gap > 0.1:
            logger.warning(
                f"[FAIRNESS] Demographic parity gap={gap:.3f} > 0.10 — "
                "check for bias in predictions!"
            )
        self.dp_gap.reset()

    # -----------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        total_steps   = self.trainer.estimated_stepping_batches
        warmup_steps  = int(total_steps * self.hparams.warmup_epochs
                            / self.hparams.max_epochs)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer":  optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":  "step",
                "frequency": 1,
            },
        }

    # -----------------------------------------------------------------------
    def predict_with_guard(
        self,
        video: torch.Tensor,
        fps: float,
        demo_attr: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Safe inference wrapper. Returns dict:
          {
            "prediction": 0 | 1 | -1,   # -1 = undetermined
            "probability": float | None,
            "reason": str,
          }
        """
        valid, reason = check_input_quality(video.squeeze(0), fps)
        if not valid:
            return {"prediction": -1, "probability": None, "reason": reason}

        self.eval()
        with torch.no_grad():
            logit = self(video.unsqueeze(0) if video.dim() == 4 else video)
            prob  = torch.sigmoid(logit).item()

        return {
            "prediction":  int(prob >= 0.5),
            "probability": prob,
            "reason":      "ok",
        }
