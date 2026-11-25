"""Training, COCO-style evaluation and transfer-learning orchestration."""
from __future__ import annotations

import copy
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from .config import TrainingConfig
from .data import CLASS_NAMES, NUM_CLASSES
from .models import ObjectDetectionModel, save_model
from .utils import peak_memory_mb, seed_everything

_METRIC_KEYS = ("map50", "map75", "map", "precision", "recall")


# ---------------------------------------------------------------------------
# Detection metrics (COCO-style, implemented locally)
# ---------------------------------------------------------------------------
def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area1 = np.clip(boxes1[:, 2] - boxes1[:, 0], 0, None) * np.clip(boxes1[:, 3] - boxes1[:, 1], 0, None)
    area2 = np.clip(boxes2[:, 2] - boxes2[:, 0], 0, None) * np.clip(boxes2[:, 3] - boxes2[:, 1], 0, None)
    union = area1[:, None] + area2[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def _match_detections(det_boxes: np.ndarray, det_scores: np.ndarray,
                      gt_boxes: np.ndarray, iou_thr: float) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy score-ordered matching. Returns tp flags and matched gt indices."""
    order = np.argsort(-det_scores)
    matched = np.zeros(len(gt_boxes), dtype=bool)
    tp = np.zeros(len(det_boxes), dtype=bool)
    gt_idx = np.full(len(det_boxes), -1, dtype=np.int64)
    if len(gt_boxes) == 0 or len(det_boxes) == 0:
        return tp, gt_idx
    ious = box_iou(det_boxes, gt_boxes)
    for det_i in order:
        best = -1
        best_iou = iou_thr
        for g in range(len(gt_boxes)):
            if matched[g]:
                continue
            if ious[det_i, g] >= best_iou:
                best_iou = ious[det_i, g]
                best = g
        if best >= 0:
            matched[best] = True
            tp[det_i] = True
            gt_idx[det_i] = best
    return tp, gt_idx


def _average_precision(tp: np.ndarray, scores: np.ndarray, num_gt: int) -> float:
    if num_gt == 0:
        return float("nan")
    order = np.argsort(-scores)
    tp = tp[order].astype(np.float64)
    fp = 1.0 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / max(num_gt, 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    # COCO 101-point interpolation
    recall_points = np.linspace(0, 1, 101)
    precision_interp = np.zeros_like(recall_points)
    for i, r in enumerate(recall_points):
        mask = recall >= r
        precision_interp[i] = precision[mask].max() if mask.any() else 0.0
    return float(precision_interp.mean())


def compute_map(predictions: Sequence[Dict[str, torch.Tensor]],
                targets: Sequence[Dict[str, torch.Tensor]],
                num_classes: int = NUM_CLASSES,
                iou_thresholds: Sequence[float] = (0.5, 0.75),
                max_dets: int = 100) -> Dict[str, Any]:
    """Per-class AP at each IoU threshold plus precision/recall at best F1."""
    all_thresholds = sorted({round(float(t), 2) for t in iou_thresholds} |
                            {round(float(t), 2) for t in np.arange(0.5, 0.96, 0.05)})
    per_class: Dict[str, Any] = {}

    for class_id in range(1, num_classes + 1):
        gt_by_image: List[np.ndarray] = []
        det_by_image: List[Tuple[np.ndarray, np.ndarray]] = []
        for pred, target in zip(predictions, targets):
            gt_boxes = target["boxes"].detach().cpu().numpy()
            gt_labels = target["labels"].detach().cpu().numpy()
            gt_by_image.append(gt_boxes[gt_labels == class_id])
            p_boxes = pred["boxes"].detach().cpu().numpy()
            p_labels = pred["labels"].detach().cpu().numpy()
            p_scores = pred["scores"].detach().cpu().numpy()
            mask = p_labels == class_id
            order = np.argsort(-p_scores[mask])[:max_dets]
            det_by_image.append((p_boxes[mask][order], p_scores[mask][order]))

        num_gt = int(sum(len(g) for g in gt_by_image))
        num_det = int(sum(len(d[0]) for d in det_by_image))
        if num_gt == 0:
            per_class[CLASS_NAMES.get(class_id, str(class_id))] = {
                "ap50": None, "ap75": None, "ap": None,
                "precision": None, "recall": None,
                "num_gt": 0, "num_detections": num_det,
            }
            continue

        ap_by_thr: Dict[float, float] = {}
        for thr in all_thresholds:
            flat_tp, flat_scores = [], []
            for (p_boxes, p_scores), gt_boxes in zip(det_by_image, gt_by_image):
                tp, _ = _match_detections(p_boxes, p_scores, gt_boxes, thr)
                flat_tp.append(tp)
                flat_scores.append(p_scores)
            tp_arr = np.concatenate(flat_tp) if flat_tp else np.zeros((0,), bool)
            sc_arr = np.concatenate(flat_scores) if flat_scores else np.zeros((0,), np.float32)
            ap_by_thr[thr] = _average_precision(tp_arr, sc_arr, num_gt)

        # precision / recall at the operating point that maximises F1 on IoU 0.5
        flat_tp, flat_scores = [], []
        for (p_boxes, p_scores), gt_boxes in zip(det_by_image, gt_by_image):
            tp, _ = _match_detections(p_boxes, p_scores, gt_boxes, 0.5)
            flat_tp.append(tp)
            flat_scores.append(p_scores)
        tp_arr = np.concatenate(flat_tp) if flat_tp else np.zeros((0,), bool)
        sc_arr = np.concatenate(flat_scores) if flat_scores else np.zeros((0,), np.float32)
        precision = recall = 0.0
        if len(tp_arr):
            order = np.argsort(-sc_arr)
            tp_sorted = tp_arr[order].astype(np.float64)
            tp_cum = np.cumsum(tp_sorted)
            prec = tp_cum / np.arange(1, len(tp_sorted) + 1)
            rec = tp_cum / num_gt
            f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
            best = int(np.argmax(f1))
            precision, recall = float(prec[best]), float(rec[best])

        per_class[CLASS_NAMES.get(class_id, str(class_id))] = {
            "ap50": ap_by_thr.get(0.5),
            "ap75": ap_by_thr.get(0.75),
            "ap": float(np.mean(list(ap_by_thr.values()))),
            "precision": precision,
            "recall": recall,
            "num_gt": num_gt,
            "num_detections": num_det,
        }

    def _mean(key: str) -> float:
        vals = [v[key] for v in per_class.values() if v[key] is not None]
        return float(np.mean(vals)) if vals else 0.0

    results: Dict[str, Any] = {
        "map50": _mean("ap50"),
        "map75": _mean("ap75"),
        "map": _mean("ap"),
        "per_class": per_class,
        "num_images": len(predictions),
    }
    precisions = [v["precision"] for v in per_class.values() if v["precision"] is not None]
    recalls = [v["recall"] for v in per_class.values() if v["recall"] is not None]
    results["precision"] = float(np.mean(precisions)) if precisions else 0.0
    results["recall"] = float(np.mean(recalls)) if recalls else 0.0
    return results


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------
def to_device(batch, device: torch.device):
    images, boxes, labels = batch
    images = [img.to(device).float() / 255.0 for img in images]
    targets = [
        {"boxes": b.to(device).float(), "labels": l.to(device).long()}
        for b, l in zip(boxes, labels)
    ]
    return images, targets


@torch.no_grad()
def collect_predictions(model: ObjectDetectionModel, loader: DataLoader,
                        device: torch.device, max_images: Optional[int] = None
                        ) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
    model.eval()
    preds: List[Dict[str, torch.Tensor]] = []
    gts: List[Dict[str, torch.Tensor]] = []
    seen = 0
    for batch in loader:
        images, targets = to_device(batch, device)
        outputs = model(images)
        for out, target in zip(outputs, targets):
            preds.append({k: v.detach().cpu() for k, v in out.items()})
            gts.append({k: v.detach().cpu() for k, v in target.items()})
            seen += 1
            if max_images is not None and seen >= max_images:
                return preds, gts
    return preds, gts


def evaluate(model: ObjectDetectionModel, dataloader: DataLoader,
             device: torch.device = torch.device("cpu"), iou_threshold=0.5,
             max_images: Optional[int] = None, num_classes: int = NUM_CLASSES) -> Dict[str, Any]:
    started = time.perf_counter()
    preds, gts = collect_predictions(model, dataloader, device, max_images=max_images)
    metrics = compute_map(preds, gts, num_classes=num_classes,
                          iou_thresholds=(0.5, 0.75))
    metrics["eval_seconds"] = round(time.perf_counter() - started, 3)
    metrics["peak_memory_mb"] = peak_memory_mb()
    return metrics


def per_image_success(pred: Dict[str, torch.Tensor], gt: Dict[str, torch.Tensor],
                      iou_threshold: float = 0.5, score_threshold: float = 0.5) -> Dict[str, Any]:
    """Detection yield for one frame: recall of GT objects at a score cut."""
    boxes = pred["boxes"].numpy()
    scores = pred["scores"].numpy()
    labels = pred["labels"].numpy()
    gt_boxes = gt["boxes"].numpy()
    gt_labels = gt["labels"].numpy()
    keep = scores >= score_threshold
    boxes, labels = boxes[keep], labels[keep]
    matched = 0
    for gt_box, gt_label in zip(gt_boxes, gt_labels):
        mask = labels == int(gt_label)
        if mask.sum():
            ious = box_iou(boxes[mask], gt_box[None, :])[:, 0]
            if len(ious) and ious.max() >= iou_threshold:
                matched += 1
    return {
        "num_gt": int(len(gt_boxes)),
        "matched": int(matched),
        "complete": bool(len(gt_boxes) == 0 or matched == len(gt_boxes)),
        "recall": float(matched / len(gt_boxes)) if len(gt_boxes) else 1.0,
        "num_false_positives": int(len(boxes)) - int(matched),
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def make_optimizer(params: Iterable[nn.Parameter], cfg: TrainingConfig) -> torch.optim.Optimizer:
    params = list(params)
    if cfg.optimizer.lower() == "sgd":
        return torch.optim.SGD(params, lr=cfg.learning_rate, momentum=cfg.momentum,
                               weight_decay=cfg.weight_decay, nesterov=True)
    return torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)


def _loss_dict(outputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {k: float(v.detach().cpu()) for k, v in outputs.items() if torch.is_tensor(v)}


def train_epoch(model: ObjectDetectionModel, dataloader: DataLoader,
                optimizer: torch.optim.Optimizer, device: torch.device,
                cfg: Optional[TrainingConfig] = None,
                probe_loader: Optional[DataLoader] = None,
                probe_images: int = 24) -> Dict[str, Any]:
    model.train()
    total_loss = 0.0
    steps = 0
    components: Dict[str, float] = {}
    for batch in dataloader:
        images, targets = to_device(batch, device)
        loss_dict = model(images, targets)
        loss = sum(v for k, v in loss_dict.items() if torch.is_tensor(v))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss: {loss_dict}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg is not None and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), cfg.grad_clip)
        optimizer.step()
        total_loss += float(loss.detach())
        steps += 1
        for key, value in _loss_dict(loss_dict).items():
            components[key] = components.get(key, 0.0) + value

    metrics: Dict[str, Any] = {"loss": total_loss / max(steps, 1), "steps": steps}
    for key, value in components.items():
        metrics[key] = value / max(steps, 1)
    if probe_loader is not None:
        probe = evaluate(model, probe_loader, device, max_images=probe_images)
        metrics["train_probe_map50"] = probe["map50"]
        metrics["train_probe_per_class"] = probe["per_class"]
        model.train()
    return metrics


def train_transfer(model: ObjectDetectionModel, train_loader: DataLoader,
                   val_loader: DataLoader, cfg: TrainingConfig, num_epochs: int,
                   device: torch.device = torch.device("cpu"),
                   checkpoint_path: Optional[str | Path] = None,
                   probe_loader: Optional[DataLoader] = None,
                   val_eval_images: Optional[int] = 48,
                   log: bool = True) -> Dict[str, Any]:
    seed_everything(cfg.seed)
    optimizer = make_optimizer(model.trainable_parameters(), cfg)
    scheduler = None
    if cfg.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(num_epochs, 1))
    history: List[Dict[str, Any]] = []
    started = time.perf_counter()

    # "do no harm" initialisation: measure the pretrained (post weight-surgery)
    # model first and keep it as the incumbent checkpoint.  With a small target
    # dataset, short fine-tuning can degrade a strong foundation model; the
    # checkpoint then stays at the pretrained weights instead of regressing.
    initial_val = evaluate(model, val_loader, device, max_images=val_eval_images)
    best_map = initial_val["map50"]
    best_state = copy.deepcopy(model.model.state_dict())
    best_epoch = 0
    if checkpoint_path is not None:
        save_model(model, checkpoint_path, metadata={
            "epoch": 0, "val_map50": best_map, "stage": "pretrained_incumbent",
        })

    for epoch in range(1, num_epochs + 1):
        t0 = time.perf_counter()
        train_metrics = train_epoch(model, train_loader, optimizer, device, cfg,
                                    probe_loader=probe_loader)
        if scheduler is not None:
            scheduler.step()
        val_metrics = evaluate(model, val_loader, device, max_images=val_eval_images)
        improved = val_metrics["map50"] > best_map
        if improved:
            best_map = val_metrics["map50"]
            best_state = copy.deepcopy(model.model.state_dict())
            best_epoch = epoch
            if checkpoint_path is not None:
                save_model(model, checkpoint_path, metadata={
                    "epoch": epoch, "val_map50": best_map, "stage": "transfer",
                })
        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_seconds": round(time.perf_counter() - t0, 2),
        })
        if log:
            print(f"  epoch {epoch:02d}/{num_epochs} loss={train_metrics['loss']:.4f} "
                  f"val_mAP50={val_metrics['map50']:.4f} "
                  f"({history[-1]['epoch_seconds']}s)")

    if best_state is not None:
        model.model.load_state_dict(best_state)
    final_val = evaluate(model, val_loader, device)
    return {
        "best_val_map50": best_map,
        "best_epoch": best_epoch,
        "initial_val": initial_val,
        "fine_tuning_improved": best_epoch > 0,
        "final_val": final_val,
        "history": history,
        "train_seconds": round(time.perf_counter() - started, 2),
        "params": model.parameter_counts(),
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
    }
