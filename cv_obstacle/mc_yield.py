"""Monte Carlo obstacle-detection yield under randomised driving conditions.

Each draw = one real frame + one sampled sensor condition (exposure, contrast,
noise, blur, rain, fog, shadow).  A draw is a *success* when every ground-truth
obstacle is detected at IoU >= 0.5 with score >= 0.5.  The yield is the success
probability with a Wilson 95 % confidence interval; mAP is additionally measured
over bootstrap batches of draws.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from .training import box_iou, compute_map
from .utils import seed_everything

WEATHER = ["clear", "rain", "fog", "shadow", "blur"]
WEATHER_P = [0.55, 0.15, 0.10, 0.10, 0.10]


def sample_condition(rng: np.random.Generator) -> Dict[str, Any]:
    return {
        "brightness": float(rng.uniform(0.60, 1.40)),
        "contrast": float(rng.uniform(0.65, 1.35)),
        "noise": float(rng.uniform(0.0, 0.20)),
        "blur_kernel": int(rng.choice([0, 0, 1, 3, 5])),
        "weather": str(rng.choice(WEATHER, p=WEATHER_P)),
        "weather_strength": float(rng.uniform(0.2, 0.8)),
    }


def apply_condition(image: np.ndarray, cond: Dict[str, Any],
                    rng: Optional[np.random.Generator] = None) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    img = image.astype(np.float32) / 255.0
    mean = img.mean(axis=(0, 1), keepdims=True)
    img = (img - mean) * cond.get("contrast", 1.0) + mean
    img = img * cond.get("brightness", 1.0)
    if cond.get("noise", 0.0) > 0:
        img = img + rng.normal(0, cond["noise"], img.shape).astype(np.float32)
    img = np.clip(img, 0, 1)
    weather = cond.get("weather", "clear")
    strength = cond.get("weather_strength", 0.5)
    if weather == "rain":
        out = (img * 255).astype(np.uint8)
        h, w = out.shape[:2]
        for _ in range(int(120 * strength)):
            x, y = rng.integers(0, w), rng.integers(0, h)
            length = int(rng.integers(6, 22))
            slant = int(rng.integers(-4, 5))
            cv2.line(out, (int(x), int(y)), (int(x + slant), int(y + length)),
                     (200, 210, 220), 1)
        img = out.astype(np.float32) / 255.0
    elif weather == "fog":
        white = np.ones_like(img)
        img = img * (1 - 0.65 * strength) + white * (0.65 * strength)
    elif weather == "shadow":
        h, w = img.shape[:2]
        x0, y0 = int(rng.integers(0, max(1, w // 2))), int(rng.integers(0, max(1, h // 2)))
        x1, y1 = x0 + int(rng.integers(w // 4, w // 2)), y0 + int(rng.integers(h // 4, h // 2))
        mask = np.zeros((h, w), np.float32)
        cv2.rectangle(mask, (x0, y0), (min(x1, w), min(y1, h)), 1.0, -1)
        mask = cv2.GaussianBlur(mask, (31, 31), 0)[..., None]
        img = img * (1 - 0.55 * strength * mask)
    if cond.get("blur_kernel", 0):
        k = int(cond["blur_kernel"]) * 2 + 1
        img = cv2.GaussianBlur(img, (k, k), 0)
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def _wilson(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@torch.no_grad()
def run_monte_carlo_yield(model, dataset, device: torch.device, num_draws: int = 200,
                          seed: int = 42, iou_threshold: float = 0.5,
                          score_threshold: float = 0.5,
                          batch_size: int = 8,
                          map_batch: int = 20) -> Dict[str, Any]:
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    model.model.eval()
    started = time.perf_counter()
    draws: List[Dict[str, Any]] = []
    buffers: List[Dict[str, Any]] = []

    for start in range(0, num_draws, batch_size):
        size = min(batch_size, num_draws - start)
        tensors, gts, conds, names = [], [], [], []
        for _ in range(size):
            index = int(rng.integers(0, len(dataset)))
            image, boxes, labels, _ = dataset[index]
            cond = sample_condition(rng)
            aug = apply_condition(image, cond, rng)
            tensors.append(torch.from_numpy(np.ascontiguousarray(aug)).permute(2, 0, 1).float() / 255.0)
            gts.append({"boxes": torch.from_numpy(boxes).float(),
                        "labels": torch.from_numpy(labels).long()})
            conds.append(cond)
            names.append(index)
        preds = model([t.to(device) for t in tensors])
        preds = [{k: v.detach().cpu() for k, v in p.items()} for p in preds]
        buffers.append({"preds": preds, "gts": gts})
        for pred, gt, cond, index in zip(preds, gts, conds, names):
            boxes = pred["boxes"].numpy()
            scores = pred["scores"].numpy()
            labels = pred["labels"].numpy()
            gt_boxes = gt["boxes"].numpy()
            gt_labels = gt["labels"].numpy()
            keep = scores >= score_threshold
            kept_boxes, kept_labels = boxes[keep], labels[keep]
            matched = 0
            for gt_box, gt_label in zip(gt_boxes, gt_labels):
                mask = kept_labels == int(gt_label)
                if mask.sum():
                    ious = box_iou(kept_boxes[mask], gt_box[None, :])[:, 0]
                    if ious.max() >= iou_threshold:
                        matched += 1
            draws.append({
                "image_index": int(index),
                "num_gt": int(len(gt_boxes)),
                "matched": matched,
                "recall": float(matched / len(gt_boxes)) if len(gt_boxes) else 1.0,
                "complete": bool(len(gt_boxes) == 0 or matched == len(gt_boxes)),
                "false_positives": int(len(kept_boxes)) - matched,
                **{f"cond_{k}": v for k, v in cond.items()},
                "missed": int(len(gt_boxes)) - matched,
            })

    successes = sum(int(d["complete"]) for d in draws)
    yield_est = successes / max(len(draws), 1)
    lo, hi = _wilson(successes, len(draws))

    # mAP over disjoint batches of draws (each batch is a mini validation set)
    maps: List[float] = []
    for b in range(0, len(buffers)):
        preds = [p for item in buffers[b:b + 1] for p in item["preds"]]
        gts = [g for item in buffers[b:b + 1] for g in item["gts"]]
        if len(gts) >= 1:
            maps.append(compute_map(preds, gts)["map50"])
    mean_map = float(np.mean(maps)) if maps else 0.0
    std_map = float(np.std(maps)) if maps else 0.0

    def _stratify(key: str, bins: Sequence[float]) -> List[Dict[str, Any]]:
        rows = []
        values = [d[key] for d in draws]
        for lo_b, hi_b in zip(bins[:-1], bins[1:]):
            subset = [d for d, v in zip(draws, values) if lo_b <= v < hi_b]
            if subset:
                rows.append({
                    "range": [round(float(lo_b), 2), round(float(hi_b), 2)],
                    "num_draws": len(subset),
                    "yield": round(sum(int(d["complete"]) for d in subset) / len(subset), 4),
                    "mean_recall": round(float(np.mean([d["recall"] for d in subset])), 4),
                })
        return rows

    weather_rows = []
    for weather in WEATHER:
        subset = [d for d in draws if d["cond_weather"] == weather]
        if subset:
            weather_rows.append({
                "weather": weather,
                "num_draws": len(subset),
                "yield": round(sum(int(d["complete"]) for d in subset) / len(subset), 4),
                "mean_recall": round(float(np.mean([d["recall"] for d in subset])), 4),
            })

    return {
        "num_draws": len(draws),
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        "yield": round(yield_est, 4),
        "successes": successes,
        "failures": len(draws) - successes,
        "yield_ci95": [round(lo, 4), round(hi, 4)],
        "mean_recall": round(float(np.mean([d["recall"] for d in draws])), 4),
        "mean_missed_obstacles": round(float(np.mean([d["missed"] for d in draws])), 3),
        "mean_false_positives": round(float(np.mean([d["false_positives"] for d in draws])), 3),
        "map50_mean": round(mean_map, 4),
        "map50_std": round(std_map, 4),
        "map50_batches": len(maps),
        "stratified": {
            "brightness": _stratify("cond_brightness", [0.6, 0.8, 1.0, 1.2, 1.4, 1.41]),
            "contrast": _stratify("cond_contrast", [0.65, 0.85, 1.05, 1.25, 1.36]),
            "noise": _stratify("cond_noise", [0.0, 0.05, 0.10, 0.15, 0.2001]),
            "weather": weather_rows,
        },
        "per_draw_summary": [
            {k: d[k] for k in ("image_index", "num_gt", "matched", "recall", "complete",
                               "cond_brightness", "cond_contrast", "cond_noise",
                               "cond_weather")}
            for d in draws
        ],
        "elapsed_s": round(time.perf_counter() - started, 2),
    }
