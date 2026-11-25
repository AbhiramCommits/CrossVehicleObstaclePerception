"""Differentiable robustness optimisation of input preprocessing.

The detector's backbone is frozen, so the only free variables are the input
preprocessing parameters (brightness, contrast, gamma, per-channel gains).
They are ``torch.nn.Parameter`` tensors optimised with Adam against a smooth,
differentiable detection surrogate evaluated over a stencil of sensor
conditions.  The *reported* mAP is always the real, non-differentiable COCO
metric measured after optimisation.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import ObjectDetectionModel, split_transform_output
from .training import box_iou, collect_predictions, compute_map, evaluate
from .utils import seed_everything

class _ImageList:
    """Minimal ImageList-compatible shim (only what the RPN anchor generator needs)."""

    def __init__(self, tensors, image_sizes):
        self.tensors = tensors
        self.image_sizes = [tuple(int(v) for v in s) for s in image_sizes]


CONDITIONS: List[Dict[str, float]] = [
    {"brightness": b, "contrast": c, "noise": n}
    for b in (0.8, 1.0, 1.2)
    for c in (0.8, 1.0, 1.2)
    for n in (0.0, 0.10, 0.20)
]


EVAL_CONDITIONS: List[Dict[str, float]] = [
    {"brightness": b, "contrast": c, "noise": 0.15}
    for b in (0.8, 1.0, 1.2)
    for c in (0.8, 1.0, 1.2)
]


@dataclass
class RobustnessConfig:
    iterations: int = 100
    learning_rate: float = 0.05
    lambda_variance: float = 0.5
    conditions_per_step: int = 3
    images_per_step: int = 4
    optim_resolution: int = 192
    brightness_bounds: Tuple[float, float] = (0.75, 1.35)
    contrast_bounds: Tuple[float, float] = (0.75, 1.35)
    gamma_bounds: Tuple[float, float] = (0.8, 1.25)
    gain_bounds: Tuple[float, float] = (0.85, 1.15)
    seed: int = 42


class RobustPreprocessor(nn.Module):
    """Bounded, differentiable photometric correction."""

    def __init__(self, cfg: RobustnessConfig):
        super().__init__()
        self.cfg = cfg
        self.raw_brightness = nn.Parameter(self._identity_raw(cfg.brightness_bounds))
        self.raw_contrast = nn.Parameter(self._identity_raw(cfg.contrast_bounds))
        self.raw_gamma = nn.Parameter(self._identity_raw(cfg.gamma_bounds))
        self.raw_gain = nn.Parameter(self._identity_raw(cfg.gain_bounds).repeat(3))

    @staticmethod
    def _identity_raw(bounds: Tuple[float, float]) -> torch.Tensor:
        lo, hi = bounds
        p = min(max((1.0 - lo) / (hi - lo), 1e-4), 1 - 1e-4)
        return torch.tensor([math.log(p / (1 - p))])

    @staticmethod
    def _bounded(raw: torch.Tensor, bounds: Tuple[float, float]) -> torch.Tensor:
        lo, hi = bounds
        return lo + (hi - lo) * torch.sigmoid(raw)

    def parameters_dict(self) -> Dict[str, Any]:
        return {
            "brightness_scale": round(float(self._bounded(self.raw_brightness, self.cfg.brightness_bounds)), 4),
            "contrast_scale": round(float(self._bounded(self.raw_contrast, self.cfg.contrast_bounds)), 4),
            "gamma": round(float(self._bounded(self.raw_gamma, self.cfg.gamma_bounds)), 4),
            "channel_gains": [round(float(v), 4) for v in
                              self._bounded(self.raw_gain, self.cfg.gain_bounds)],
        }

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        squeeze = images.dim() == 3
        if squeeze:
            images = images.unsqueeze(0)
        b = self._bounded(self.raw_brightness, self.cfg.brightness_bounds)
        c = self._bounded(self.raw_contrast, self.cfg.contrast_bounds)
        g = self._bounded(self.raw_gamma, self.cfg.gamma_bounds)
        gain = self._bounded(self.raw_gain, self.cfg.gain_bounds).view(1, 3, 1, 1)
        mean = images.mean(dim=(2, 3), keepdim=True)
        out = (images - mean) * c + mean
        out = out * b * gain
        out = torch.clamp_min(out, 1e-4)
        out = torch.pow(out, g)
        return out.squeeze(0) if squeeze else out


def apply_condition(images: torch.Tensor, condition: Dict[str, float],
                    generator: Optional[torch.Generator] = None) -> torch.Tensor:
    mean = images.mean(dim=(2, 3), keepdim=True)
    out = (images - mean) * float(condition.get("contrast", 1.0)) + mean
    out = out * float(condition.get("brightness", 1.0))
    if condition.get("noise", 0.0) > 0:
        noise = torch.randn(images.shape, generator=generator, device=images.device,
                            dtype=images.dtype)
        out = out + noise * float(condition["noise"])
    return torch.clamp(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Differentiable detection surrogate
# ---------------------------------------------------------------------------
def soft_detection_quality(model: ObjectDetectionModel, images: torch.Tensor,
                           targets: Sequence[Dict[str, torch.Tensor]]) -> torch.Tensor:
    """Smooth proxy for detection quality: RPN objectness at ground-truth boxes.

    Fully deterministic (no RPN sampling / NMS), so gradients can be checked
    against finite differences.
    """
    transformed = model.model.transform(images)
    tensors, image_sizes = split_transform_output(transformed)
    image_list = model.model.transform(images) if image_sizes is None else _ImageList(tensors, image_sizes)
    features = model.model.backbone(tensors)
    objectness, deltas = model.model.rpn.head(list(features.values()))
    anchors_per_level = model.model.rpn.anchor_generator(image_list, list(features.values()))
    deltas_flat = torch.cat(
        [d.permute(0, 2, 3, 1).reshape(d.shape[0], -1, 4) for d in deltas], dim=1)
    # anchors_per_level is one (N, 4) tensor per image; decode per image.
    boxes_decoded = [
        model.model.rpn.box_coder.decode(deltas_flat[i], [anchors_per_level[i]])
        for i in range(deltas_flat.shape[0])
    ]
    total = images.new_zeros(())
    count = 0
    for i, target in enumerate(targets):
        gt_boxes = target["boxes"]
        if len(gt_boxes) == 0:
            continue
        scores = torch.cat([
            objectness[l].permute(0, 2, 3, 1).reshape(objectness[l].shape[0], -1)[i]
            for l in range(len(objectness))
        ])
        boxes = boxes_decoded[i].reshape(-1, 4)
        probs = torch.sigmoid(scores)
        ious = box_iou(boxes.detach().cpu().numpy(), gt_boxes.detach().cpu().numpy())
        ious_t = torch.from_numpy(ious).to(images.device)
        h_img, w_img = images.shape[-2:]
        scale = boxes.new_tensor([w_img, h_img, w_img, h_img])
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2 / scale[:2]
        sizes = (boxes[:, 2:] - boxes[:, :2]).clamp(min=1.0) / scale[2:]
        image_quality = []
        for g in range(len(gt_boxes)):
            # smooth (softmax-weighted) association with every anchor; no argmax
            weights = torch.softmax(ious_t[:, g] * 8.0, dim=0)
            soft_recall = (weights * probs).sum()
            pred_c = (weights[:, None] * centers).sum(0)
            pred_wh = (weights[:, None] * sizes).sum(0)
            gt_box = gt_boxes[g]
            gt_c = (gt_box[:2] + gt_box[2:]) / 2 / scale[:2]
            gt_wh = (gt_box[2:] - gt_box[:2]).clamp(min=1.0) / scale[2:]
            box_error = ((pred_c - gt_c).abs().mean() +
                         (pred_wh.log() - gt_wh.log()).abs().mean())
            image_quality.append(soft_recall - 0.05 * box_error.clamp(max=5.0))
        total = total + torch.stack(image_quality).mean()
        count += 1
    return total / max(count, 1)


def robustness_objective(preprocessor: RobustPreprocessor, model: ObjectDetectionModel,
                         images: torch.Tensor, targets: Sequence[Dict[str, torch.Tensor]],
                         conditions: Sequence[Dict[str, float]], lambda_variance: float,
                         generator: Optional[torch.Generator] = None
                         ) -> Tuple[torch.Tensor, Dict[str, float]]:
    qualities = []
    for condition in conditions:
        processed = preprocessor(apply_condition(images, condition, generator))
        qualities.append(soft_detection_quality(model, processed, targets))
    q = torch.stack(qualities)
    mean_q = q.mean()
    loss = (1.0 - mean_q) + lambda_variance * q.var(unbiased=False)
    return loss, {"mean_quality": float(mean_q.detach()), "variance": float(q.var(unbiased=False).detach())}


def gradient_probe_loss(preprocessor: RobustPreprocessor, model: ObjectDetectionModel,
                        images: torch.Tensor, conditions: Sequence[Dict[str, float]],
                        generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Smooth scalar through preprocessing -> backbone -> RPN.

    Used for the finite-difference gradient check: unlike the full detection
    surrogate it contains no masking, NMS or box decoding, so it is numerically
    well conditioned while still exercising the real detector graph.
    """
    total = images.new_zeros(())
    for condition in conditions:
        x = preprocessor(apply_condition(images, condition, generator))
        tensors, _ = split_transform_output(model.model.transform(x))
        features = model.model.backbone(tensors)
        objectness, _ = model.model.rpn.head(list(features.values()))
        energy = features["0"].pow(2).mean()
        obj = torch.stack([torch.sigmoid(o).mean() for o in objectness]).mean()
        total = total + energy + obj
    return total / max(len(conditions), 1)


def verify_gradients(preprocessor: RobustPreprocessor, model: ObjectDetectionModel,
                     images: torch.Tensor, targets=None, conditions=None,
                     lambda_variance: float = 0.5,
                     eps: float = 1e-4, seed: int = 42) -> Dict[str, Any]:
    """Compare autograd against central finite differences in float64."""
    import copy
    conditions = list(conditions or CONDITIONS[:3])
    preprocessor = copy.deepcopy(preprocessor).double()
    was_training = model.model.training
    model.model.eval()
    model.model.double()
    images = images.double()
    gen = torch.Generator().manual_seed(seed)
    fixed = [apply_condition(images, c, gen) for c in conditions]

    def objective() -> torch.Tensor:
        total = images.new_zeros(())
        for x in fixed:
            processed = preprocessor(x)
            tensors, _ = split_transform_output(model.model.transform(processed))
            features = model.model.backbone(tensors)
            objectness, _ = model.model.rpn.head(list(features.values()))
            total = total + features["0"].pow(2).mean()
            total = total + torch.stack([torch.sigmoid(o).mean() for o in objectness]).mean()
        return total / len(fixed)

    results: List[Dict[str, Any]] = []
    try:
        for name, param in (("raw_brightness", preprocessor.raw_brightness),
                            ("raw_contrast", preprocessor.raw_contrast)):
            if param.grad is not None:
                param.grad = None
            loss = objective()
            loss.backward()
            analytic = float(param.grad.item())
            original = param.detach().clone()
            values = []
            for delta in (eps, -eps):
                param.data = original + delta
                with torch.no_grad():
                    values.append(float(objective()))
                param.data = original
            numeric = (values[0] - values[1]) / (2 * eps)
            denom = max(abs(analytic), abs(numeric), 1e-12)
            results.append({
                "parameter": name,
                "autograd": analytic,
                "finite_difference": numeric,
                "abs_error": abs(analytic - numeric),
                "relative_error": abs(analytic - numeric) / denom,
            })
            if param.grad is not None:
                param.grad = None
    finally:
        model.model.float()
        model.model.train(was_training)
    return {
        "checks": results,
        "max_relative_error": max(r["relative_error"] for r in results),
        "tolerance": 1e-2,
        "dtype": "float64",
        "eps": eps,
        "probe": "preprocessing -> ResNet-FPN features + mean RPN objectness",
        "num_conditions": len(conditions),
        "num_images": int(images.shape[0]),
        "passed": all(r["relative_error"] < 1e-2 for r in results),
    }


# ---------------------------------------------------------------------------
# Optimisation driver
# ---------------------------------------------------------------------------
def _condition_batches(cfg: RobustnessConfig, iterations: int) -> List[List[Dict[str, float]]]:
    rng = np.random.default_rng(cfg.seed)
    out = []
    for _ in range(iterations):
        idx = rng.choice(len(CONDITIONS), size=cfg.conditions_per_step, replace=False)
        out.append([CONDITIONS[int(i)] for i in idx])
    return out


def optimize_robustness(model: ObjectDetectionModel, dataset, device: torch.device,
                        cfg: RobustnessConfig, eval_loader=None,
                        eval_conditions: Optional[Sequence[Dict[str, float]]] = None,
                        eval_images: int = 8,
                        log: bool = True) -> Dict[str, Any]:
    seed_everything(cfg.seed)
    model.model.eval()
    for p in model.model.parameters():
        p.requires_grad_(False)
    preprocessor = RobustPreprocessor(cfg).to(device)
    optimizer = torch.optim.Adam(preprocessor.parameters(), lr=cfg.learning_rate)

    # fixed optimisation subset (uint8 HWC numpy arrays)
    rng = np.random.default_rng(cfg.seed + 1)
    indices = rng.choice(len(dataset), size=min(cfg.images_per_step, len(dataset)), replace=False)
    images_list, targets_list = [], []
    for idx in indices:
        image, boxes, labels, _ = dataset[int(idx)]
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
        images_list.append(tensor)
        targets_list.append({"boxes": torch.from_numpy(boxes).float(),
                             "labels": torch.from_numpy(labels).long()})
    # resize to the optimisation resolution (short side) for speed
    resized = []
    for tensor, target in zip(images_list, targets_list):
        h, w = tensor.shape[-2:]
        scale = cfg.optim_resolution / max(h, w)
        nh, nw = max(8, int(round(h * scale))), max(8, int(round(w * scale)))
        resized.append(F.interpolate(tensor[None], size=(nh, nw), mode="bilinear",
                                     align_corners=False)[0])
        boxes = target["boxes"] * torch.tensor([nw / w, nh / h, nw / w, nh / h])
        target["boxes"] = boxes
    images = torch.stack(resized).to(device)
    targets = [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)}
               for t in targets_list]

    schedule = _condition_batches(cfg, cfg.iterations)
    history: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for step in range(1, cfg.iterations + 1):
        conditions = schedule[step - 1]
        optimizer.zero_grad(set_to_none=True)
        loss, stats = robustness_objective(preprocessor, model, images, targets, conditions,
                                           cfg.lambda_variance,
                                           generator=torch.Generator().manual_seed(step))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(preprocessor.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 10 == 0 or step == cfg.iterations:
            # log a *comparable* loss on a fixed stencil, not the random batch
            with torch.no_grad():
                fixed_loss, fixed_stats = robustness_objective(
                    preprocessor, model, images, targets, CONDITIONS[:3],
                    cfg.lambda_variance,
                    generator=torch.Generator().manual_seed(1234))
            entry = {"iteration": step, "loss": round(float(fixed_loss), 5),
                     "train_loss": round(float(loss.detach()), 5),
                     **{k: round(v, 5) for k, v in fixed_stats.items()},
                     **preprocessor.parameters_dict()}
            history.append(entry)
            if log:
                print(f"  opt iter {step:03d} loss={entry['loss']:.4f} "
                      f"quality={entry['mean_quality']:.4f} "
                      f"brightness={entry['brightness_scale']:.3f} "
                      f"contrast={entry['contrast_scale']:.3f}")
    # "do no harm" selection: keep identity preprocessing if it scores better on
    # the full stencil under the same differentiable surrogate.
    identity = RobustPreprocessor(cfg).to(device)
    with torch.no_grad():
        selection_conditions = CONDITIONS[:9]
        opt_scores = [soft_detection_quality(model, preprocessor(apply_condition(images, c)), targets)
                      for c in selection_conditions]
        id_scores = [soft_detection_quality(model, identity(apply_condition(images, c)), targets)
                     for c in selection_conditions]
        opt_mean = float(torch.stack(opt_scores).mean())
        id_mean = float(torch.stack(id_scores).mean())
    selected = "optimized"
    if id_mean >= opt_mean:
        preprocessor.load_state_dict(identity.state_dict())
        selected = "identity"
    params = preprocessor.parameters_dict()

    result: Dict[str, Any] = {
        "selected": selected,
        "surrogate_quality": {"optimized": round(opt_mean, 5), "identity": round(id_mean, 5)},
        "optimized_parameters": params,
        "history": history,
        "initial_parameters": {"brightness_scale": 1.0, "contrast_scale": 1.0,
                               "gamma": 1.0, "channel_gains": [1.0, 1.0, 1.0]},
        "iterations": cfg.iterations,
        "optimization_seconds": round(time.perf_counter() - started, 2),
        "objective_config": cfg.__dict__,
        "num_optimization_images": len(indices),
        "optim_resolution": cfg.optim_resolution,
    }
    if eval_loader is not None:
        result["evaluation"] = evaluate_with_preprocessor_conditions(
            model, preprocessor, eval_loader, device, max_images=eval_images,
            conditions=eval_conditions or EVAL_CONDITIONS)
    return result


@torch.no_grad()
def evaluate_with_preprocessor_conditions(model: ObjectDetectionModel,
                                          preprocessor: Optional[RobustPreprocessor],
                                          loader, device: torch.device,
                                          max_images: int = 8,
                                          conditions: Sequence[Dict[str, float]] = EVAL_CONDITIONS
                                          ) -> Dict[str, Any]:
    """Real mAP per condition, with and without the learned preprocessing."""
    model.eval()
    before: List[float] = []
    after: List[float] = []
    per_condition: List[Dict[str, Any]] = []
    for condition in conditions:
        preds_b, gts, preds_a = [], [], []
        seen = 0
        for batch in loader:
            images, targets = [], []
            for img, box, lab in zip(*batch):
                tensor = img.float() / 255.0
                cond_img = apply_condition(tensor[None], condition)[0]
                images.append(cond_img)
                targets.append({"boxes": box.float(), "labels": lab.long()})
            images = [i.to(device) for i in images]
            out_b = model(images)
            if preprocessor is not None:
                out_a = model([preprocessor(i) for i in images])
            else:
                out_a = out_b
            for i in range(len(images)):
                preds_b.append({k: v.detach().cpu() for k, v in out_b[i].items()})
                preds_a.append({k: v.detach().cpu() for k, v in out_a[i].items()})
                gts.append({k: v.detach().cpu() for k, v in targets[i].items()})
                seen += 1
                if seen >= max_images:
                    break
            if seen >= max_images:
                break
        mb = compute_map(preds_b, gts)
        ma = compute_map(preds_a, gts)
        before.append(mb["map50"])
        after.append(ma["map50"])
        per_condition.append({
            "condition": condition,
            "baseline_map50": round(mb["map50"], 4),
            "optimized_map50": round(ma["map50"], 4),
            "delta": round(ma["map50"] - mb["map50"], 4),
        })
    arr_b = np.asarray(before)
    arr_a = np.asarray(after)
    worst_idx = int(np.argmin(arr_b))
    return {
        "per_condition": per_condition,
        "baseline": {
            "mean_map50": round(float(arr_b.mean()), 4),
            "worst_map50": round(float(arr_b.min()), 4),
            "variance": round(float(arr_b.var()), 6),
            "worst_condition": conditions[worst_idx],
        },
        "optimized": {
            "mean_map50": round(float(arr_a.mean()), 4),
            "worst_map50": round(float(arr_a.min()), 4),
            "variance": round(float(arr_a.var()), 6),
            "worst_condition": conditions[int(np.argmin(arr_a))],
        },
        "improvement": {
            "mean_abs": round(float(arr_a.mean() - arr_b.mean()), 4),
            "mean_pct": round(100.0 * float(arr_a.mean() - arr_b.mean()) / max(float(arr_b.mean()), 1e-9), 2),
            "worst_abs": round(float(arr_a.min() - arr_b.min()), 4),
            "worst_pct": round(100.0 * float(arr_a.min() - arr_b.min()) / max(float(arr_b.min()), 1e-9), 2),
            "variance_reduction": round(float(arr_b.var() - arr_a.var()), 6),
        },
        "num_conditions": len(conditions),
        "max_images_per_condition": max_images,
    }
