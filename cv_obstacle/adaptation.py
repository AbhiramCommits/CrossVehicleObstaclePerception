"""Adversarial domain adaptation (DANN) for cross-vehicle transfer.

The detector's ROI-head embedding is the domain-invariant bottleneck: a
discriminator tries to tell source (sedan/KITTI) from target (truck/SUV BDD100K
with camera style transfer) features, while a gradient-reversal layer makes the
detector's head produce features that fool it.

A short pseudo-label self-training phase follows the adversarial phase.  It also
uses *only target images* (no target labels), and is reported separately so the
contribution of each phase is visible.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from .config import AdaptationConfig
from .models import ObjectDetectionModel
from .training import evaluate, make_optimizer, to_device
from .utils import seed_everything


# ---------------------------------------------------------------------------
# Gradient reversal
# ---------------------------------------------------------------------------
class _GradientReversal(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def gradient_reversal(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return _GradientReversal.apply(x, lambd)


# ---------------------------------------------------------------------------
# Domain discriminator
# ---------------------------------------------------------------------------
class DomainDiscriminator(nn.Module):
    """CNN over ROI embeddings predicting source (0) / target (1).

    ``input_dim`` is the flat ROI embedding width.  When given, the embedding is
    projected to ``num_channels`` and reshaped to a 2x2 map so the same
    convolutional topology is used for both detector families.
    """

    def __init__(self, num_channels: int = 256, num_layers: int = 3,
                 input_dim: Optional[int] = None, hidden_dim: int = 128):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.project: Optional[nn.Linear] = None
        if input_dim is not None:
            self.project = nn.Linear(input_dim, num_channels * 4)
        layers: List[nn.Module] = []
        in_ch = num_channels
        for i in range(num_layers):
            out_ch = num_channels if i < num_layers - 1 else num_channels
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=2 if i == 0 else 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(num_channels, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 2:
            if self.project is None:
                raise ValueError("2-D features require input_dim at construction")
            x = self.project(features).view(features.shape[0], self.num_channels, 2, 2)
        else:
            x = features
        x = self.pool(self.conv(x)).flatten(1)
        return self.classifier(x).squeeze(-1)


def build_discriminator(features: torch.Tensor, cfg: AdaptationConfig) -> DomainDiscriminator:
    input_dim = int(features.shape[-1]) if features.dim() == 2 else None
    return DomainDiscriminator(num_channels=cfg.num_discriminator_layers * 64 + 64,
                               num_layers=cfg.num_discriminator_layers,
                               input_dim=input_dim)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def adversarial_domain_adaptation_loss(source_features: torch.Tensor,
                                       target_features: torch.Tensor,
                                       discriminator: DomainDiscriminator,
                                       lambda_adv: float = 1.0,
                                       detector_loss: Optional[torch.Tensor | float] = None,
                                       grl_lambda: float = 1.0) -> Dict[str, torch.Tensor]:
    """Discriminator BCE + gradient-reversed adversarial detector loss.

    * discriminator loss: BCE(D(source)=0) + BCE(D(target)=1)
    * adversarial loss  : BCE(D(GRL(source))=1) + BCE(D(GRL(target))=0)
    * combined          : ``detector_loss + lambda_adv * adv_loss``
    """
    src_det = source_features.detach()
    tgt_det = target_features.detach()
    src_labels = torch.zeros(src_det.shape[0], device=src_det.device)
    tgt_labels = torch.ones(tgt_det.shape[0], device=tgt_det.device)
    src_logits = discriminator(src_det)
    tgt_logits = discriminator(tgt_det)
    d_loss = 0.5 * (F.binary_cross_entropy_with_logits(src_logits, src_labels) +
                    F.binary_cross_entropy_with_logits(tgt_logits, tgt_labels))

    src_adv_logits = discriminator(gradient_reversal(source_features, grl_lambda))
    tgt_adv_logits = discriminator(gradient_reversal(target_features, grl_lambda))
    # Flip labels: fooling means the discriminator's outputs become ambiguous.
    adv_loss = 0.5 * (F.binary_cross_entropy_with_logits(src_adv_logits, tgt_labels) +
                      F.binary_cross_entropy_with_logits(tgt_adv_logits, src_labels))
    if detector_loss is None:
        det = torch.zeros((), device=src_det.device)
    elif torch.is_tensor(detector_loss):
        det = detector_loss
    else:
        det = torch.tensor(float(detector_loss), device=src_det.device)
    total = det + lambda_adv * adv_loss
    with torch.no_grad():
        ambiguity = ((torch.sigmoid(src_adv_logits) - 0.5).abs().mean() +
                     (torch.sigmoid(tgt_adv_logits) - 0.5).abs().mean()) / 2
    return {
        "loss": total,
        "detector_loss": det,
        "adv_loss": adv_loss,
        "discriminator_loss": d_loss,
        "ambiguity": ambiguity,
        "source_domain_prob": torch.sigmoid(src_logits).mean().detach(),
        "target_domain_prob": torch.sigmoid(tgt_logits).mean().detach(),
    }


# ---------------------------------------------------------------------------
# Feature capture helpers
# ---------------------------------------------------------------------------
def _take(features: Optional[torch.Tensor], n: int = 64) -> Optional[torch.Tensor]:
    if features is None or features.numel() == 0:
        return None
    flat = features.reshape(features.shape[0], -1) if features.dim() > 2 else features
    return flat[:n]


# ---------------------------------------------------------------------------
# Adaptation loop
# ---------------------------------------------------------------------------
def adapt_model(model: ObjectDetectionModel,
                source_loader,
                target_loader,
                discriminator: Optional[DomainDiscriminator],
                cfg: AdaptationConfig,
                num_epochs: int,
                device: torch.device = torch.device("cpu"),
                target_eval_loader=None,
                log: bool = True) -> Dict[str, Any]:
    seed_everything(cfg.seed)
    model.unfreeze_head()
    model.train()
    discriminator = discriminator.to(device)
    opt_det = torch.optim.AdamW(model.trainable_parameters(), lr=cfg.learning_rate,
                                weight_decay=1e-4)
    opt_disc = torch.optim.AdamW(discriminator.parameters(), lr=cfg.learning_rate,
                                 weight_decay=1e-4)
    history: List[Dict[str, Any]] = []
    started = time.perf_counter()
    target_iter = iter(target_loader)

    def next_target():
        nonlocal target_iter
        try:
            return next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            return next(target_iter)

    for epoch in range(1, num_epochs + 1):
        t0 = time.perf_counter()
        epoch_stats: Dict[str, float] = {}
        steps = 0
        for source_batch in source_loader:
            if isinstance(source_batch, dict):
                src_images = [torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float().to(device) / 255.0
                              for im in source_batch["source_image"]]
                src_targets = [{"boxes": torch.as_tensor(b).float().to(device),
                                "labels": torch.as_tensor(l).long().to(device)}
                               for b, l in zip(source_batch["source_boxes"], source_batch["source_labels"])]
                tgt_images = [torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float().to(device) / 255.0
                              for im in source_batch["target_image"]]
            else:
                target_batch = next_target()
                src_images, src_targets = to_device(source_batch, device)
                tgt_images, _ = to_device(target_batch, device)

            model.clear_features()
            model.train()
            det_out = model(src_images, src_targets)
            det_loss = model.loss(det_out)
            src_feats = _take(model.roi_features)
            model.clear_features()
            # unlabelled target images: eval mode is required to obtain
            # detections (and therefore ROI embeddings) without targets
            model.model.eval()
            model(tgt_images, None)
            model.model.train()
            tgt_feats = _take(model.roi_features)
            model.clear_features()
            if src_feats is None or tgt_feats is None or src_feats.shape[0] == 0 or tgt_feats.shape[0] == 0:
                continue

            # ---- discriminator update -----------------------------------
            disc_out = adversarial_domain_adaptation_loss(
                src_feats.detach(), tgt_feats.detach(), discriminator,
                lambda_adv=cfg.lambda_adv,
            )
            opt_disc.zero_grad(set_to_none=True)
            disc_out["discriminator_loss"].backward()
            opt_disc.step()

            # ---- detector update (detection + adversarial) --------------
            adv_out = adversarial_domain_adaptation_loss(
                src_feats, tgt_feats, discriminator, lambda_adv=cfg.lambda_adv,
                grl_lambda=1.0,
            )
            total = det_loss + cfg.lambda_adv * adv_out["adv_loss"]
            opt_det.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 10.0)
            opt_det.step()

            for key, value in {
                "detector_loss": det_loss.detach(),
                "adv_loss": adv_out["adv_loss"].detach(),
                "discriminator_loss": disc_out["discriminator_loss"].detach(),
                "ambiguity": adv_out["ambiguity"],
            }.items():
                epoch_stats[key] = epoch_stats.get(key, 0.0) + float(value)
            steps += 1

        stats = {k: v / max(steps, 1) for k, v in epoch_stats.items()}
        stats["epoch"] = epoch
        if target_eval_loader is not None:
            metrics = evaluate(model, target_eval_loader, device, max_images=24)
            stats["target_map50"] = metrics["map50"]
            stats["target_precision"] = metrics["precision"]
            stats["target_recall"] = metrics["recall"]
        stats["epoch_seconds"] = round(time.perf_counter() - t0, 2)
        history.append(stats)
        if log:
            print(f"  adapt epoch {epoch:02d}/{num_epochs} det={stats.get('detector_loss', 0):.4f} "
                  f"adv={stats.get('adv_loss', 0):.4f} d={stats.get('discriminator_loss', 0):.4f} "
                  f"target_mAP50={stats.get('target_map50', float('nan')):.4f} "
                  f"({stats['epoch_seconds']}s)")
        model.train()

    return {
        "history": history,
        "adversarial_seconds": round(time.perf_counter() - started, 2),
        "num_discriminator_params": int(sum(p.numel() for p in discriminator.parameters())),
    }


# ---------------------------------------------------------------------------
# Pseudo-label self-training
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_pseudo_labels(model: ObjectDetectionModel, loader,
                           device: torch.device, threshold: float = 0.7
                           ) -> List[Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]]:
    model.eval()
    out: List[Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]] = []
    for batch in loader:
        images, _ = to_device(batch, device)
        preds = model(images)
        targets = []
        for pred in preds:
            keep = pred["scores"] >= threshold
            targets.append({"boxes": pred["boxes"][keep].detach(),
                            "labels": pred["labels"][keep].detach()})
        out.append((torch.stack(images, dim=0), targets))
    return out


def self_train(model: ObjectDetectionModel, target_loader, cfg: AdaptationConfig,
               epochs: int, device: torch.device = torch.device("cpu"),
               target_eval_loader=None, scope: str = "predictor",
               log: bool = True) -> Dict[str, Any]:
    """Train on high-confidence target pseudo-labels (target images only)."""
    if scope == "predictor":
        model.train_predictor_only()
    else:
        model.unfreeze_head()
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=cfg.learning_rate,
                                  weight_decay=1e-4)
    history: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        pseudo = generate_pseudo_labels(model, target_loader, device,
                                        cfg.pseudo_label_threshold)
        model.train()
        losses = []
        num_boxes = 0
        for stacked, targets in pseudo:
            images = [img for img in stacked]
            num_boxes += sum(int(len(t["boxes"])) for t in targets)
            loss_dict = model(images, targets)
            loss = model.loss(loss_dict)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 10.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        entry: Dict[str, Any] = {
            "epoch": epoch,
            "pseudo_label_loss": float(np.mean(losses)) if losses else None,
            "num_pseudo_boxes": num_boxes,
        }
        if target_eval_loader is not None:
            metrics = evaluate(model, target_eval_loader, device, max_images=24)
            entry["target_map50"] = metrics["map50"]
            entry["target_precision"] = metrics["precision"]
            entry["target_recall"] = metrics["recall"]
        history.append(entry)
        if log:
            print(f"  self-train epoch {epoch:02d}/{epochs} "
                  f"loss={entry['pseudo_label_loss']} boxes={num_boxes} "
                  f"target_mAP50={entry.get('target_map50', float('nan')):.4f}")
    return {"history": history, "self_training_seconds": round(time.perf_counter() - started, 2)}
