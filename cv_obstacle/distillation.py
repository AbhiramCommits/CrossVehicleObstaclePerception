"""Knowledge distillation: adapted teacher -> lightweight student.

Response-based distillation at the ROI head: the teacher's confident detections
define a set of boxes; both models' class logits at those boxes are compared with
temperature-scaled KL divergence, while the student also regresses hard targets.
The teacher is frozen for the whole run.
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
from torch.utils.data import Dataset, DataLoader

from .config import DistillationConfig
from .data import NUM_CLASSES, collate_detections, make_loader
from .models import ObjectDetectionModel, load_pretrained_detector, save_model
from .training import box_iou, evaluate, to_device
from .utils import seed_everything


def create_student_model(model_name: str = "yolov8n", num_classes: int = NUM_CLASSES,
                         pretrained: bool = True, min_size: int = 320,
                         max_size: int = 320) -> ObjectDetectionModel:
    """Create the lightweight student (slim ResNet-18 FPN Faster R-CNN)."""
    model, spec = load_pretrained_detector(model_name, num_classes=num_classes,
                                           pretrained=pretrained,
                                           min_size=min_size, max_size=max_size)
    return ObjectDetectionModel(model, spec)


def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                      targets: torch.Tensor, temperature: float = 4.0,
                      alpha: float = 0.7, return_components: bool = False):
    """(1 - alpha) * CE(student, targets) + alpha * T^2 * KL(student||teacher).

    Rows whose target is ``-1`` are ignored in both terms (unmatched proposals).
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(f"logit shape mismatch: {tuple(student_logits.shape)} vs "
                         f"{tuple(teacher_logits.shape)}")
    if student_logits.numel() == 0:
        zero = student_logits.sum() * 0.0
        return ({"loss": zero, "ce": zero, "kd": zero} if return_components else zero)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    valid = targets >= 0
    if valid.sum() == 0:
        zero = student_logits.sum() * 0.0
        return ({"loss": zero, "ce": zero, "kd": zero} if return_components else zero)

    ce = F.cross_entropy(student_logits[valid], targets[valid])
    s = student_logits[valid] / temperature
    t = teacher_logits[valid].detach() / temperature
    kd = F.kl_div(F.log_softmax(s, dim=-1), F.softmax(t, dim=-1),
                  reduction="batchmean") * (temperature ** 2)
    loss = (1.0 - alpha) * ce + alpha * kd
    if return_components:
        return {"loss": loss, "ce": ce, "kd": kd}
    return loss


# ---------------------------------------------------------------------------
# Teacher cache
# ---------------------------------------------------------------------------
@torch.no_grad()
def build_teacher_cache(dataset: Dataset, teacher: ObjectDetectionModel,
                        device: torch.device, max_props: int = 32
                        ) -> List[Dict[str, torch.Tensor]]:
    """Pre-compute teacher boxes/logits once (teacher is frozen)."""
    teacher.eval()
    cache: List[Dict[str, torch.Tensor]] = []
    for index in range(len(dataset)):
        image, boxes, labels, _ = dataset[index]
        h, w = image.shape[:2]
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().to(device) / 255.0
        teacher.clear_features()
        det = teacher([tensor])[0]
        t_boxes = det["boxes"][:max_props].detach()
        t_scores = det["scores"][:max_props].detach()
        t_labels = det["labels"][:max_props].detach()
        image_size = teacher.captured_image_sizes[0]
        sy, sx = image_size[0] / h, image_size[1] / w
        scaled = t_boxes.clone()
        scaled[:, 0::2] *= sx
        scaled[:, 1::2] *= sy
        logits = teacher.roi_class_logits(
            teacher._backbone_features, scaled, image_shapes=[image_size]
        ) if len(t_boxes) else torch.zeros((0, teacher.spec.num_classes + 1), device=device)
        cache.append({
            "teacher_boxes": t_boxes.cpu(),
            "teacher_boxes_scaled": scaled.cpu(),
            "teacher_scores": t_scores.cpu(),
            "teacher_labels": t_labels.cpu(),
            "teacher_logits": logits.detach().cpu(),
            "image_size": torch.tensor(image_size),
            "orig_size": torch.tensor([h, w]),
        })
        teacher.clear_features()
    return cache


class TeacherCacheDataset(Dataset):
    def __init__(self, base: Dataset, cache: List[Dict[str, torch.Tensor]]):
        if len(base) != len(cache):
            raise ValueError("cache length does not match dataset")
        self.base = base
        self.cache = cache

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, boxes, labels, _ = self.base[index]
        item = {
            "image": image,
            "boxes": boxes,
            "labels": labels,
        }
        item.update(self.cache[index])
        return item


def collate_kd(batch: Sequence[Dict[str, Any]]):
    return {
        "images": [torch.from_numpy(np.ascontiguousarray(item["image"])).permute(2, 0, 1)
                   for item in batch],
        "boxes": [torch.from_numpy(item["boxes"]) for item in batch],
        "labels": [torch.from_numpy(item["labels"]) for item in batch],
        "teacher_boxes": [item["teacher_boxes"] for item in batch],
        "teacher_boxes_scaled": [item["teacher_boxes_scaled"] for item in batch],
        "teacher_scores": [item["teacher_scores"] for item in batch],
        "teacher_labels": [item["teacher_labels"] for item in batch],
        "teacher_logits": [item["teacher_logits"] for item in batch],
    }


# ---------------------------------------------------------------------------
# Distillation training
# ---------------------------------------------------------------------------
def _kd_targets(teacher_boxes: torch.Tensor, gt_boxes: torch.Tensor,
                gt_labels: torch.Tensor, iou_threshold: float = 0.5) -> torch.Tensor:
    targets = torch.full((len(teacher_boxes),), -1, dtype=torch.long)
    if len(teacher_boxes) == 0 or len(gt_boxes) == 0:
        return targets
    ious = box_iou(teacher_boxes.numpy(), gt_boxes.numpy())
    for i in range(len(teacher_boxes)):
        best = int(np.argmax(ious[i]))
        if ious[i, best] >= iou_threshold:
            targets[i] = int(gt_labels[best])
    return targets


def train_with_distillation(student: ObjectDetectionModel, teacher: Optional[ObjectDetectionModel],
                            train_loader: DataLoader, cfg: DistillationConfig,
                            num_epochs: int, device: torch.device = torch.device("cpu"),
                            use_kd: bool = True,
                            val_loader: Optional[DataLoader] = None,
                            checkpoint_path: Optional[str | Path] = None,
                            log: bool = True) -> Dict[str, Any]:
    seed_everything(cfg.seed)
    student.unfreeze_head()
    optimizer = torch.optim.AdamW(student.trainable_parameters(), lr=cfg.learning_rate,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(num_epochs, 1))
    history: List[Dict[str, Any]] = []
    started = time.perf_counter()
    cache: Optional[List[Dict[str, torch.Tensor]]] = None
    loader = train_loader

    if use_kd:
        if teacher is None:
            raise ValueError("teacher is required when use_kd=True")
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        t0 = time.perf_counter()
        cache = build_teacher_cache(train_loader.dataset, teacher, device,
                                    cfg.max_teacher_proposals)
        cache_seconds = round(time.perf_counter() - t0, 2)
        if log:
            print(f"  cached teacher targets for {len(cache)} images in {cache_seconds}s")
        kd_dataset = TeacherCacheDataset(train_loader.dataset, cache)
        loader = DataLoader(kd_dataset, batch_size=cfg.batch_size, shuffle=True,
                            num_workers=0, collate_fn=collate_kd)
    else:
        cache_seconds = 0.0

    best_map = -1.0
    best_state = None
    for epoch in range(1, num_epochs + 1):
        t0 = time.perf_counter()
        student.train()
        totals: Dict[str, float] = {}
        steps = 0
        for batch in loader:
            if use_kd:
                images = [img.to(device).float() / 255.0 for img in batch["images"]]
                gt_boxes = [b.to(device).float() for b in batch["boxes"]]
                gt_labels = [l.to(device).long() for l in batch["labels"]]
                t_boxes_scaled = [b.to(device).float() for b in batch["teacher_boxes_scaled"]]
                t_logits = [l.to(device).float() for l in batch["teacher_logits"]]
                t_boxes = [b.to(device).float() for b in batch["teacher_boxes"]]
                targets = [{"boxes": b, "labels": l} for b, l in zip(gt_boxes, gt_labels)]
            else:
                images, targets = to_device(batch, device)
                gt_boxes = [t["boxes"] for t in targets]
                gt_labels = [t["labels"] for t in targets]

            student.clear_features()
            loss_dict = student(images, targets)
            det_loss = student.loss(loss_dict)
            total = det_loss
            kd_value = torch.zeros((), device=device)
            if use_kd:
                student_logits = student.logits_at_captured_boxes(t_boxes_scaled)
                per_image = []
                valid_rows = 0
                for s_logits, t_logit, tb, gb, gl in zip(student_logits, t_logits,
                                                         t_boxes, gt_boxes, gt_labels):
                    n = min(len(s_logits), len(t_logit))
                    if n == 0:
                        continue
                    targets_kd = _kd_targets(tb[:n].cpu(), gb.cpu(), gl.cpu())
                    per_image.append(distillation_loss(
                        s_logits[:n], t_logit[:n], targets_kd.to(device),
                        temperature=cfg.temperature, alpha=cfg.alpha,
                    ))
                    valid_rows += int((targets_kd >= 0).sum())
                if per_image:
                    kd_value = torch.stack(per_image).mean()
                    total = det_loss + cfg.loss_weight * kd_value
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(student.trainable_parameters(), 10.0)
            optimizer.step()
            student.clear_features()
            totals["loss"] = totals.get("loss", 0.0) + float(total.detach())
            totals["detector_loss"] = totals.get("detector_loss", 0.0) + float(det_loss.detach())
            totals["kd_loss"] = totals.get("kd_loss", 0.0) + float(kd_value.detach())
            steps += 1
        scheduler.step()
        entry: Dict[str, Any] = {"epoch": epoch,
                                 **{k: v / max(steps, 1) for k, v in totals.items()}}
        if val_loader is not None:
            metrics = evaluate(student, val_loader, device, max_images=16)
            entry["val_map50"] = metrics["map50"]
            entry["val_map"] = metrics["map"]
            entry["val_per_class"] = metrics["per_class"]
            if metrics["map50"] > best_map:
                best_map = metrics["map50"]
                best_state = copy.deepcopy(student.model.state_dict())
                if checkpoint_path is not None:
                    save_model(student, checkpoint_path, metadata={
                        "epoch": epoch, "val_map50": best_map,
                        "distillation": use_kd,
                        "temperature": cfg.temperature, "alpha": cfg.alpha,
                    })
        entry["epoch_seconds"] = round(time.perf_counter() - t0, 2)
        history.append(entry)
        if log:
            print(f"  distill epoch {epoch:02d}/{num_epochs} loss={entry.get('loss', 0):.4f} "
                  f"kd={entry.get('kd_loss', 0):.4f} "
                  f"val_mAP50={entry.get('val_map50', float('nan')):.4f} "
                  f"({entry['epoch_seconds']}s)")

    if best_state is not None:
        student.model.load_state_dict(best_state)
    return {
        "history": history,
        "best_val_map50": best_map,
        "teacher_cache_seconds": cache_seconds,
        "train_seconds": round(time.perf_counter() - started, 2),
        "params": student.parameter_counts(),
        "used_kd": use_kd,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
    }
