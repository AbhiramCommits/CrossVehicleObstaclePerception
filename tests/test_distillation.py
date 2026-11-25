"""Tests for knowledge distillation."""
from __future__ import annotations

import numpy as np
import torch

from cv_obstacle.config import DistillationConfig
from cv_obstacle.data import BDD100KDataset, KittiDataset, make_loader
from cv_obstacle.distillation import (build_teacher_cache, create_student_model,
                                      distillation_loss, train_with_distillation)
from cv_obstacle.models import ObjectDetectionModel, count_parameters, load_pretrained_detector
from cv_obstacle.training import evaluate
from cv_obstacle.utils import seed_everything


def _teacher() -> ObjectDetectionModel:
    seed_everything(5)
    model, spec = load_pretrained_detector("yolov8m", num_classes=4, pretrained=False,
                                           min_size=128, max_size=128)
    wrapped = ObjectDetectionModel(model, spec)
    wrapped.unfreeze_head()
    return wrapped


def _student(min_size: int = 128) -> ObjectDetectionModel:
    seed_everything(5)
    return create_student_model("yolov8n", num_classes=4, pretrained=False,
                                min_size=min_size, max_size=min_size)


def test_distillation_loss_is_defined_for_all_softmax_outputs():
    torch.manual_seed(0)
    student = torch.randn(8, 5, requires_grad=True)
    teacher = torch.randn(8, 5)
    targets = torch.tensor([0, 1, 2, 3, 4, -1, 1, 2])
    out = distillation_loss(student, teacher, targets, temperature=4.0, alpha=0.7,
                            return_components=True)
    assert torch.isfinite(out["loss"]) and torch.isfinite(out["ce"]) and torch.isfinite(out["kd"])
    out["loss"].backward()
    assert torch.isfinite(student.grad).all()
    # alpha=0 is pure cross-entropy
    ce_only = distillation_loss(student, teacher, targets, alpha=0.0)
    assert torch.allclose(ce_only, out["ce"], atol=1e-6)
    # temperature scaling does not change the CE component
    out2 = distillation_loss(student, teacher, targets, temperature=1.0, alpha=1.0,
                             return_components=True)
    assert torch.allclose(out2["ce"], out["ce"], atol=1e-6)


def test_distillation_loss_rejects_shape_mismatch():
    try:
        distillation_loss(torch.randn(4, 3), torch.randn(4, 4), torch.zeros(4, dtype=torch.long))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_student_has_roughly_half_the_parameters():
    teacher = _teacher()
    student = _student()
    teacher_params = count_parameters(teacher.model)
    student_params = count_parameters(student.model)
    assert student_params < 0.6 * teacher_params, (student_params, teacher_params)
    assert student_params < teacher_params


def test_teacher_is_frozen_during_distillation(toy_dataset):
    teacher = _teacher()
    student = _student()
    ds = BDD100KDataset(toy_dataset / "bdd100k", split="train", camera="truck")
    loader = make_loader(ds, batch_size=2, shuffle=True)
    cfg = DistillationConfig(epochs=1, batch_size=2, temperature=4.0, alpha=0.7,
                             max_teacher_proposals=8, seed=2)
    train_with_distillation(student, teacher, loader, cfg, 1, torch.device("cpu"),
                            use_kd=True, val_loader=None, log=False)
    assert all(not p.requires_grad for p in teacher.parameters())
    assert all(p.grad is None for p in teacher.parameters())


def test_kd_training_improves_over_no_kd(toy_dataset):
    teacher = _teacher()
    kitti = KittiDataset(toy_dataset / "kitti", split="train")
    bdd_val = BDD100KDataset(toy_dataset / "bdd100k", split="val", camera="truck")
    loader = make_loader(kitti, batch_size=2, shuffle=True)
    val_loader = make_loader(bdd_val, batch_size=2, shuffle=False)
    cfg = DistillationConfig(epochs=4, batch_size=2, learning_rate=1e-3,
                             temperature=4.0, alpha=0.7, max_teacher_proposals=8, seed=2)

    plain = _student()
    train_with_distillation(plain, None, loader, cfg, 4, torch.device("cpu"),
                            use_kd=False, val_loader=None, log=False)
    plain_map = evaluate(plain, val_loader, torch.device("cpu"))["map50"]

    student = _student()
    train_with_distillation(student, teacher, loader, cfg, 4, torch.device("cpu"),
                            use_kd=True, val_loader=None, log=False)
    kd_map = evaluate(student, val_loader, torch.device("cpu"))["map50"]
    assert kd_map >= plain_map - 1e-9, (plain_map, kd_map)


def test_teacher_cache_is_stable(toy_dataset):
    teacher = _teacher()
    ds = KittiDataset(toy_dataset / "kitti", split="val")
    cache = build_teacher_cache(ds, teacher, torch.device("cpu"), max_props=8)
    assert len(cache) == len(ds)
    for item in cache:
        assert item["teacher_boxes"].shape[1] == 4
        assert item["teacher_logits"].shape[1] == 5
        assert item["teacher_boxes"].shape[0] == item["teacher_logits"].shape[0]
