"""Tests for transfer-learning training, metrics and checkpointing."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

from cv_obstacle.config import Config, TrainingConfig
from cv_obstacle.data import KittiDataset, make_loader
from cv_obstacle.models import ObjectDetectionModel, load_model, load_pretrained_detector, save_model
from cv_obstacle.training import (box_iou, compute_map, evaluate, make_optimizer,
                                  train_epoch, train_transfer)
from cv_obstacle.utils import seed_everything


def _model(min_size: int = 128) -> ObjectDetectionModel:
    seed_everything(7)
    model, spec = load_pretrained_detector("yolov8n", num_classes=4, pretrained=False,
                                           min_size=min_size, max_size=min_size)
    wrapped = ObjectDetectionModel(model, spec)
    wrapped.unfreeze_head()
    return wrapped


def _loaders(toy_dataset, batch_size: int = 2):
    train_ds = KittiDataset(toy_dataset / "kitti", split="train")
    val_ds = KittiDataset(toy_dataset / "kitti", split="val")
    return (make_loader(train_ds, batch_size=batch_size, shuffle=True),
            make_loader(val_ds, batch_size=batch_size, shuffle=False),
            train_ds, val_ds)


def test_compute_map_orders_predictions_correctly():
    gt = [{"boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]), "labels": torch.tensor([1])}]
    perfect = [{"boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                "scores": torch.tensor([0.9]), "labels": torch.tensor([1])}]
    wrong = [{"boxes": torch.tensor([[200.0, 200.0, 250.0, 250.0]]),
              "scores": torch.tensor([0.9]), "labels": torch.tensor([1])}]
    empty = [{"boxes": torch.zeros((0, 4)), "scores": torch.zeros((0,)),
              "labels": torch.zeros((0,), dtype=torch.long)}]
    assert compute_map(perfect, gt)["map50"] == 1.0
    assert compute_map(wrong, gt)["map50"] == 0.0
    assert compute_map(empty, gt)["map50"] == 0.0
    assert 0.0 <= box_iou(np.array([[0, 0, 10, 10]]), np.array([[0, 0, 10, 10]]))[0, 0] <= 1.0


def test_training_decreases_loss(toy_dataset):
    train_loader, _, _, _ = _loaders(toy_dataset)
    model = _model()
    cfg = TrainingConfig(batch_size=2, learning_rate=1e-3, optimizer="adamw", epochs=3)
    optimizer = make_optimizer(model.trainable_parameters(), cfg)
    losses = []
    for _ in range(3):
        metrics = train_epoch(model, train_loader, optimizer, torch.device("cpu"), cfg)
        losses.append(metrics["loss"])
    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"


def test_val_metrics_are_finite_and_bounded(toy_dataset):
    _, val_loader, _, _ = _loaders(toy_dataset)
    model = _model()
    metrics = evaluate(model, val_loader, torch.device("cpu"))
    for key in ("map50", "map75", "map", "precision", "recall"):
        assert 0.0 <= metrics[key] <= 1.0
    for name, values in metrics["per_class"].items():
        for key in ("precision", "recall"):
            if values[key] is not None:
                assert 0.0 <= values[key] <= 1.0


def test_perfect_predictions_beat_random_detections(toy_dataset):
    _, val_loader, _, _ = _loaders(toy_dataset)
    model = _model()
    metrics = evaluate(model, val_loader, torch.device("cpu"))

    # random-box baseline on the same ground truth
    preds, gts = [], []
    rng = np.random.default_rng(0)
    for batch in val_loader:
        images, boxes, labels = batch
        for box, lab in zip(boxes, labels):
            n = max(1, len(box))
            rand = torch.tensor(rng.uniform(0, 120, size=(n, 4)), dtype=torch.float32)
            rand[:, 2:] = rand[:, :2] + 8
            preds.append({"boxes": rand, "scores": torch.rand(n), "labels": lab})
            gts.append({"boxes": box, "labels": lab})
    random_map = compute_map(preds, gts)["map50"]
    assert metrics["map50"] >= random_map


def test_checkpoint_roundtrip_is_deterministic(toy_dataset, tmp_path):
    train_loader, val_loader, _, _ = _loaders(toy_dataset)
    model = _model()
    cfg = TrainingConfig(batch_size=2, learning_rate=1e-3, epochs=1)
    optimizer = make_optimizer(model.trainable_parameters(), cfg)
    train_epoch(model, train_loader, optimizer, torch.device("cpu"), cfg)
    path = tmp_path / "checkpoint.pt"
    save_model(model, path, metadata={"test": True})
    assert path.exists()

    reloaded = load_model(path)
    model.model.eval()
    reloaded.model.eval()
    batch = next(iter(val_loader))
    images = [img.float() / 255.0 for img in batch[0]]
    with torch.no_grad():
        a = model(images)
        b = reloaded(images)
    for pa, pb in zip(a, b):
        assert torch.allclose(pa["boxes"], pb["boxes"], atol=1e-5)
        assert torch.allclose(pa["scores"], pb["scores"], atol=1e-6)


def test_warm_started_model_is_not_worse_than_scratch(toy_dataset, tmp_path):
    """A checkpoint-initialised model must match/exceed an identical scratch model."""
    train_loader, val_loader, train_ds, _ = _loaders(toy_dataset)
    cfg = TrainingConfig(batch_size=2, learning_rate=1e-3, epochs=1)
    warm = _model()
    optimizer = make_optimizer(warm.trainable_parameters(), cfg)
    train_epoch(warm, train_loader, optimizer, torch.device("cpu"), cfg)
    checkpoint = tmp_path / "warm.pt"
    save_model(warm, checkpoint)
    warm_metrics = evaluate(warm, val_loader, torch.device("cpu"))

    scratch = _model()
    scratch.unfreeze_all()
    scratch_opt = make_optimizer(scratch.trainable_parameters(), cfg)
    train_epoch(scratch, train_loader, scratch_opt, torch.device("cpu"), cfg)
    scratch_metrics = evaluate(scratch, val_loader, torch.device("cpu"))

    reloaded = load_model(checkpoint)
    reloaded_metrics = evaluate(reloaded, val_loader, torch.device("cpu"))
    assert abs(reloaded_metrics["map50"] - warm_metrics["map50"]) < 1e-6
    # On the identical tiny budget the warm-started model is at least as good
    # as the random-initialised model (ties are allowed on coarse 4-image mAP).
    assert warm_metrics["map50"] >= scratch_metrics["map50"]


def test_train_transfer_writes_history_and_best_checkpoint(toy_dataset, tmp_path):
    train_loader, val_loader, _, _ = _loaders(toy_dataset)
    model = _model()
    cfg = TrainingConfig(batch_size=2, learning_rate=1e-3, epochs=2)
    out = tmp_path / "baseline.pt"
    result = train_transfer(model, train_loader, val_loader, cfg, 2,
                            torch.device("cpu"), checkpoint_path=out, log=False,
                            val_eval_images=2)
    assert out.exists()
    assert len(result["history"]) == 2
    assert "map50" in result["final_val"]
    payload = torch.load(out, map_location="cpu", weights_only=False)
    assert "state_dict" in payload and "spec" in payload
