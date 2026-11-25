"""Figure generation from recorded result JSONs (150 dpi, labelled axes)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import CLASS_NAMES
from .utils import load_json

DPI = 150
COLORS = {1: (66, 135, 245), 2: (240, 80, 80), 3: (40, 180, 120), 4: (250, 180, 40)}
COLOR_NAMES = {1: "tab:blue", 2: "tab:red", 3: "tab:green", 4: "tab:orange"}


def _save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _maybe(results_dir: Path, name: str) -> Optional[Dict[str, Any]]:
    path = results_dir / name
    return load_json(path) if path.exists() else None


# ---------------------------------------------------------------------------
def plot_transfer_curves(results: Dict[str, Any], out: Path) -> Optional[str]:
    history = results.get("training", {}).get("history") or []
    if not history:
        return None
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(epochs, [h["train"]["loss"] for h in history], "o-", label="train loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("detection loss")
    axes[0].set_title("Transfer fine-tuning loss (KITTI, frozen backbone)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, [h["train"].get("train_probe_map50", np.nan) for h in history],
                 "o-", label="train probe mAP@0.5")
    axes[1].plot(epochs, [h["val"]["map50"] for h in history], "s-", label="val mAP@0.5")
    scratch = results.get("scratch", {}).get("training", {}).get("history") or []
    if scratch:
        axes[1].plot([h["epoch"] for h in scratch], [h["val"]["map50"] for h in scratch],
                     "^--", label="from scratch val mAP@0.5")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("mAP@0.5")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Transfer learning vs training from scratch")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    return _save(fig, out)


def plot_adaptation(results: Dict[str, Any], out: Path) -> Optional[str]:
    before = results.get("target_before", {})
    after = results.get("target_after", {})
    if not before.get("per_class"):
        return None
    classes = list(before["per_class"].keys())
    b = [before["per_class"][c]["ap50"] or 0.0 for c in classes]
    a = [after["per_class"][c]["ap50"] or 0.0 for c in classes]
    x = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(x - 0.2, b, 0.4, label=f"before adaptation (mAP {before['map50']:.3f})",
           color="tab:gray")
    ax.bar(x + 0.2, a, 0.4, label=f"after adaptation (mAP {after['map50']:.3f})",
           color="tab:purple")
    ax.set_xticks(x, classes)
    ax.set_ylabel("AP@0.5")
    ax.set_ylim(0, 1)
    ax.set_title("Cross-domain adaptation: KITTI sedan -> BDD100K truck camera")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    return _save(fig, out)


def plot_distillation(results: Dict[str, Any], out: Path) -> Optional[str]:
    teacher = results.get("teacher_map", {}).get("map50")
    no_kd = results.get("student_no_kd_map", {}).get("map50")
    kd = results.get("student_kd_map", {}).get("map50")
    if teacher is None or kd is None:
        return None
    labels = ["Teacher\n(adapted)", "Student\n(no KD)", "Student\n(+KD)"]
    values = [teacher, no_kd or 0.0, kd]
    colors = ["tab:blue", "tab:gray", "tab:green"]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    bars = ax.bar(labels, values, color=colors)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}",
                ha="center", fontsize=9)
    ax.set_ylabel("mAP@0.5 (target val)")
    ax.set_ylim(0, max(1.0, max(values) * 1.2))
    ax.set_title(f"Knowledge distillation (T={results.get('temperature')}, "
                 f"alpha={results.get('alpha')})")
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, out)


def plot_latency(results: Dict[str, Any], out: Path) -> Optional[str]:
    rows = []
    if results.get("teacher"):
        rows.append(("Teacher\nPyTorch", results["teacher"]["latency"], "tab:blue"))
    if results.get("student"):
        rows.append(("Student\nPyTorch", results["student"]["latency"], "tab:green"))
    if results.get("student_onnx"):
        rows.append(("Student\nONNX", results["student_onnx"]["latency"], "tab:orange"))
    if results.get("student_int8", {}).get("latency"):
        rows.append(("Student\nint8", results["student_int8"]["latency"], "tab:red"))
    if not rows:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    names = [r[0] for r in rows]
    means = [r[1]["mean_ms"] for r in rows]
    p95 = [r[1]["p95_ms"] for r in rows]
    x = np.arange(len(rows))
    axes[0].bar(x - 0.2, means, 0.4, label="mean", color=[r[2] for r in rows])
    axes[0].bar(x + 0.2, p95, 0.4, label="p95", color=[r[2] for r in rows], alpha=0.45)
    axes[0].set_xticks(x, names)
    axes[0].set_ylabel("latency (ms / image)")
    axes[0].set_title(f"Latency at {tuple(results.get('input_size', []))} on {results.get('device')}")
    for i, value in enumerate(means):
        axes[0].text(i - 0.2, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()

    ips = [r[1].get("throughput_ips", 0.0) for r in rows]
    axes[1].bar(x, ips, color=[r[2] for r in rows])
    axes[1].set_xticks(x, names)
    axes[1].set_ylabel("images / second")
    axes[1].set_title("Single-image throughput")
    for i, value in enumerate(ips):
        axes[1].text(i, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)
    return _save(fig, out)


def plot_optimization(results: Dict[str, Any], out: Path) -> Optional[str]:
    history = results.get("history") or []
    if not history:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot([h["iteration"] for h in history], [h["loss"] for h in history], "o-",
                 color="tab:blue")
    axes[0].set_xlabel("Adam iteration")
    axes[0].set_ylabel("robustness objective")
    axes[0].set_title("Differentiable pre-processing optimisation")
    axes[0].grid(alpha=0.3)

    evaluation = results.get("evaluation", {})
    per_condition = evaluation.get("per_condition", [])
    if per_condition:
        labels = [f"b{c['condition']['brightness']:.1f}\nc{c['condition']['contrast']:.1f}"
                  for c in per_condition]
        before = [c["baseline_map50"] for c in per_condition]
        after = [c["optimized_map50"] for c in per_condition]
        x = np.arange(len(labels))
        axes[1].bar(x - 0.2, before, 0.4, label="fixed preprocessing", color="tab:gray")
        axes[1].bar(x + 0.2, after, 0.4, label="optimised preprocessing", color="tab:green")
        axes[1].set_xticks(x, labels, fontsize=7)
        axes[1].set_ylabel("mAP@0.5")
        axes[1].set_title("Real mAP per sensor condition")
        axes[1].grid(axis="y", alpha=0.3)
        axes[1].legend(fontsize=8)
    return _save(fig, out)


def plot_robustness_heatmap(results: Dict[str, Any], out: Path) -> Optional[str]:
    per_condition = results.get("evaluation", {}).get("per_condition", [])
    if not per_condition:
        return None
    brightness = sorted({c["condition"]["brightness"] for c in per_condition})
    contrast = sorted({c["condition"]["contrast"] for c in per_condition})
    if len(brightness) * len(contrast) != len(per_condition):
        # fall back to a 1-D line plot
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(range(len(per_condition)), [c["baseline_map50"] for c in per_condition],
                "o-", label="baseline")
        ax.plot(range(len(per_condition)), [c["optimized_map50"] for c in per_condition],
                "s-", label="optimised")
        ax.set_xlabel("condition index")
        ax.set_ylabel("mAP@0.5")
        ax.legend()
        return _save(fig, out)
    before = np.full((len(brightness), len(contrast)), np.nan)
    after = np.full_like(before, np.nan)
    for c in per_condition:
        i = brightness.index(c["condition"]["brightness"])
        j = contrast.index(c["condition"]["contrast"])
        before[i, j] = c["baseline_map50"]
        after[i, j] = c["optimized_map50"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    vmin = float(np.nanmin([before.min(), after.min()]))
    vmax = float(np.nanmax([before.max(), after.max()]))
    for ax, data, title in ((axes[0], before, "Baseline preprocessing"),
                            (axes[1], after, "Optimised preprocessing")):
        im = ax.imshow(data, origin="lower", vmin=vmin, vmax=vmax, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(contrast)), [f"{v:.1f}" for v in contrast])
        ax.set_yticks(range(len(brightness)), [f"{v:.1f}" for v in brightness])
        ax.set_xlabel("contrast scale")
        ax.set_ylabel("brightness scale")
        ax.set_title(title)
        for i in range(len(brightness)):
            for j in range(len(contrast)):
                ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center",
                        color="white" if data[i, j] < (vmin + vmax) / 2 else "black", fontsize=7)
    fig.colorbar(im, ax=axes, shrink=0.8, label="mAP@0.5")
    fig.suptitle("Obstacle mAP under sensor-noise / lighting perturbations")
    return _save(fig, out)


def plot_monte_carlo(results: Dict[str, Any], out: Path) -> Optional[str]:
    strat = results.get("stratified", {})
    if not strat:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    weather = strat.get("weather", [])
    if weather:
        axes[0].bar([w["weather"] for w in weather], [w["yield"] for w in weather],
                    color="teal")
        axes[0].set_ylim(0, 1.05)
        axes[0].set_ylabel("detection yield")
        axes[0].set_title(f"Yield by weather (N={results.get('num_draws')}, "
                          f"overall={results.get('yield')})")
        axes[0].grid(axis="y", alpha=0.3)
    noise = strat.get("noise", [])
    if noise:
        labels = [f"{n['range'][0]:.2f}-{n['range'][1]:.2f}" for n in noise]
        axes[1].plot(labels, [n["yield"] for n in noise], "o-", label="yield")
        axes[1].plot(labels, [n["mean_recall"] for n in noise], "s--", label="mean recall")
        axes[1].set_xlabel("Gaussian noise std")
        axes[1].set_ylabel("probability / recall")
        axes[1].set_ylim(0, 1.05)
        axes[1].set_title("Monte Carlo reliability vs sensor noise")
        axes[1].grid(alpha=0.3)
        axes[1].legend()
    return _save(fig, out)


# ---------------------------------------------------------------------------
# Image-based figures
# ---------------------------------------------------------------------------
def _to_rgb(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(image[..., ::-1] if image.shape[-1] == 3 else image)


def draw_predictions(image: np.ndarray, pred: Dict[str, torch.Tensor],
                     gt: Optional[Dict[str, torch.Tensor]] = None,
                     score_threshold: float = 0.4) -> np.ndarray:
    canvas = _to_rgb(image).copy()
    if gt is not None:
        for box, label in zip(gt["boxes"].numpy(), gt["labels"].numpy()):
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 1)
    boxes = pred["boxes"].detach().cpu().numpy()
    scores = pred["scores"].detach().cpu().numpy()
    labels = pred["labels"].detach().cpu().numpy()
    for box, score, label in zip(boxes, scores, labels):
        if score < score_threshold:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        color = COLORS.get(int(label), (200, 200, 200))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        name = CLASS_NAMES.get(int(label), str(label))
        cv2.putText(canvas, f"{name} {score:.2f}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return canvas


def plot_detection_examples(model, datasets: Dict[str, Any], out: Path,
                            device: str = "cpu") -> Optional[str]:
    panels = []
    for name, dataset in datasets.items():
        if dataset is None or len(dataset) == 0:
            continue
        indices = np.linspace(0, len(dataset) - 1, num=min(2, len(dataset)), dtype=int)
        model.model.eval()
        for idx in indices:
            image, boxes, labels, _ = dataset[int(idx)]
            tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
            with torch.no_grad():
                pred = model([tensor.to(device)])[0]
            gt = {"boxes": torch.from_numpy(boxes), "labels": torch.from_numpy(labels)}
            canvas = draw_predictions(image, pred, gt)
            panels.append((f"{name} #{int(idx)} (mAP view)", canvas))
            if len(panels) >= 4:
                break
        if len(panels) >= 4:
            break
    if not panels:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, (title, canvas) in zip(axes.ravel(), panels):
        ax.imshow(canvas)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle("Obstacle detection outputs (white = ground truth, colour = predicted)")
    return _save(fig, out)


def plot_domain_shift(kitti_ds, bdd_ds, out: Path) -> Optional[str]:
    if kitti_ds is None or bdd_ds is None or len(kitti_ds) == 0 or len(bdd_ds) == 0:
        return None
    src, _, _, _ = kitti_ds[0]
    tgt, _, _, _ = bdd_ds[0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    axes[0].imshow(_to_rgb(src))
    axes[0].set_title("Source domain: KITTI sedan camera")
    axes[0].axis("off")
    axes[1].imshow(_to_rgb(tgt))
    axes[1].set_title("Target domain: BDD100K + truck-camera style transfer")
    axes[1].axis("off")
    return _save(fig, out)
