"""Tests for differentiable robustness optimisation."""
from __future__ import annotations

import numpy as np
import torch

from cv_obstacle.data import KittiDataset
from cv_obstacle.models import ObjectDetectionModel, load_pretrained_detector
from cv_obstacle.optimization import (CONDITIONS, EVAL_CONDITIONS, RobustnessConfig, RobustPreprocessor,
                                      evaluate_with_preprocessor_conditions,
                                      optimize_robustness, verify_gradients)
from cv_obstacle.utils import seed_everything


def _model(size: int = 96) -> ObjectDetectionModel:
    seed_everything(13)
    model, spec = load_pretrained_detector("yolov8n", num_classes=4, pretrained=False,
                                           min_size=size, max_size=size)
    return ObjectDetectionModel(model, spec)


def _batch(dataset, size: int = 96, count: int = 2):
    images, targets = [], []
    for i in range(count):
        image, boxes, labels, _ = dataset[i]
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
        h, w = image.shape[:2]
        tensor = torch.nn.functional.interpolate(tensor[None], size=(size, size),
                                                 mode="bilinear", align_corners=False)[0]
        scaled = torch.from_numpy(boxes).float() * torch.tensor([size / w, size / h, size / w, size / h])
        images.append(tensor)
        targets.append({"boxes": scaled, "labels": torch.from_numpy(labels).long()})
    return torch.stack(images), targets


def test_gradient_check_matches_finite_differences(toy_dataset):
    dataset = KittiDataset(toy_dataset / "kitti", split="val")
    model = _model()
    images, targets = _batch(dataset)
    cfg = RobustnessConfig(seed=1)
    pre = RobustPreprocessor(cfg)
    result = verify_gradients(pre, model, images, targets, CONDITIONS[:2], seed=1)
    assert result["passed"], result


def test_objective_decreases_and_bounds_hold(toy_dataset):
    dataset = KittiDataset(toy_dataset / "kitti", split="val")
    model = _model()
    cfg = RobustnessConfig(iterations=8, learning_rate=0.05, images_per_step=2,
                           conditions_per_step=2, optim_resolution=96, seed=2)
    result = optimize_robustness(model, dataset, torch.device("cpu"), cfg, log=False)
    history = result["history"]
    assert history[-1]["loss"] < history[0]["loss"]
    params = result["optimized_parameters"]
    assert 0.75 <= params["brightness_scale"] <= 1.35
    assert 0.75 <= params["contrast_scale"] <= 1.35
    assert 0.8 <= params["gamma"] <= 1.25
    assert all(0.85 <= g <= 1.15 for g in params["channel_gains"])
    assert result["selected"] in {"optimized", "identity"}


def test_optimized_mAP_is_not_worse_on_same_conditions(toy_dataset):
    dataset = KittiDataset(toy_dataset / "kitti", split="val")
    model = _model()
    cfg = RobustnessConfig(iterations=4, images_per_step=2, conditions_per_step=2,
                           optim_resolution=96, seed=3)
    from cv_obstacle.data import make_loader
    loader = make_loader(dataset, batch_size=2, shuffle=False)
    result = optimize_robustness(model, dataset, torch.device("cpu"), cfg,
                                 eval_loader=loader, log=False)
    evaluation = result["evaluation"]
    # per-condition real mAP, averaged; identity preprocessing is the fallback
    assert evaluation["optimized"]["mean_map50"] >= evaluation["baseline"]["mean_map50"] - 1e-9
    assert evaluation["num_conditions"] == len(EVAL_CONDITIONS)
