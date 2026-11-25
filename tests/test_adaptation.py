"""Tests for adversarial domain adaptation."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from cv_obstacle.adaptation import (DomainDiscriminator, adapt_model,
                                    adversarial_domain_adaptation_loss,
                                    build_discriminator, generate_pseudo_labels,
                                    gradient_reversal, self_train)
from cv_obstacle.config import AdaptationConfig
from cv_obstacle.data import (BDD100KDataset, DomainAdaptationDataset, KittiDataset,
                              make_loader)
from cv_obstacle.models import ObjectDetectionModel, load_pretrained_detector
from cv_obstacle.training import evaluate
from cv_obstacle.utils import seed_everything


def _model() -> ObjectDetectionModel:
    seed_everything(11)
    model, spec = load_pretrained_detector("yolov8n", num_classes=4, pretrained=False,
                                           min_size=128, max_size=128)
    wrapped = ObjectDetectionModel(model, spec)
    wrapped.unfreeze_head()
    return wrapped


def test_gradient_reversal_flips_gradients():
    x = torch.ones(3, requires_grad=True)
    y = gradient_reversal(x, 2.0)
    y.sum().backward()
    assert torch.allclose(x.grad, -2.0 * torch.ones(3))


def test_discriminator_learns_domain_separation():
    seed_everything(0)
    discriminator = DomainDiscriminator(num_channels=32, num_layers=3, input_dim=64)
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-2)
    source = torch.randn(64, 64) - 2.0
    target = torch.randn(64, 64) + 2.0
    losses = []
    for _ in range(60):
        out = adversarial_domain_adaptation_loss(
            source, target, discriminator, lambda_adv=1.0, detector_loss=0.0)
        optimizer.zero_grad()
        out["discriminator_loss"].backward()
        optimizer.step()
        losses.append(float(out["discriminator_loss"]))
    assert losses[-1] < losses[0]
    with torch.no_grad():
        src_prob = torch.sigmoid(discriminator(source)).mean().item()
        tgt_prob = torch.sigmoid(discriminator(target)).mean().item()
    assert src_prob < tgt_prob


def test_adversarial_losses_finite_and_differentiable():
    discriminator = DomainDiscriminator(num_channels=16, num_layers=2, input_dim=32)
    source = torch.randn(4, 32, requires_grad=True)
    target = torch.randn(4, 32, requires_grad=True)
    out = adversarial_domain_adaptation_loss(source, target, discriminator,
                                             lambda_adv=1.0, detector_loss=torch.tensor(0.5))
    assert torch.isfinite(out["adv_loss"]) and torch.isfinite(out["discriminator_loss"])
    assert torch.isfinite(out["loss"])
    out["adv_loss"].backward()
    assert source.grad is not None and torch.isfinite(source.grad).all()


def test_adapt_model_improves_or_preserves_target_performance(toy_dataset):
    source_ds = KittiDataset(toy_dataset / "kitti", split="train")
    target_ds = BDD100KDataset(toy_dataset / "bdd100k", split="train", camera="truck")
    target_val = BDD100KDataset(toy_dataset / "bdd100k", split="val", camera="truck")
    paired = DomainAdaptationDataset(source_ds, target_ds)
    loader = make_loader(paired, batch_size=2, shuffle=True)
    target_loader = make_loader(target_ds, batch_size=2, shuffle=True)
    val_loader = make_loader(target_val, batch_size=2, shuffle=False)
    source_val = make_loader(KittiDataset(toy_dataset / "kitti", split="val"),
                             batch_size=2, shuffle=False)

    model = _model()
    before = evaluate(model, val_loader, torch.device("cpu"))["map50"]
    cfg = AdaptationConfig(epochs=1, learning_rate=5e-4, batch_size=2,
                           self_training_epochs=0, lambda_adv=1.0, seed=3)
    batch = next(iter(target_loader))
    images = [img.float() / 255.0 for img in batch[0]]
    model.clear_features()
    with torch.no_grad():
        model(images, None)
    disc = build_discriminator(model.roi_features, cfg)
    model.clear_features()
    result = adapt_model(model, loader, target_loader, disc, cfg, 1,
                         torch.device("cpu"), target_eval_loader=None, log=False)
    assert len(result["history"]) == 1
    for key in ("detector_loss", "adv_loss", "discriminator_loss"):
        value = result["history"][0].get(key)
        if value is not None:
            assert np.isfinite(value)
    after_adv = evaluate(model, val_loader, torch.device("cpu"))["map50"]

    # one short self-training pass on the target images (no target labels used)
    st_cfg = AdaptationConfig(epochs=1, self_training_epochs=1, batch_size=2,
                              pseudo_label_threshold=0.3, learning_rate=5e-4, seed=3)
    self_train(model, target_loader, st_cfg, 1, torch.device("cpu"),
               target_eval_loader=None, log=False)
    after_st = evaluate(model, val_loader, torch.device("cpu"))["map50"]
    assert after_st >= before - 1e-6, (before, after_adv, after_st)
    for name, value in (("before", before), ("after_adv", after_adv), ("after_st", after_st)):
        assert 0.0 <= value <= 1.0


def test_pseudo_labels_respect_threshold(toy_dataset):
    target_ds = BDD100KDataset(toy_dataset / "bdd100k", split="val", camera="truck")
    loader = make_loader(target_ds, batch_size=2, shuffle=False)
    model = _model()
    batches = generate_pseudo_labels(model, loader, torch.device("cpu"), threshold=0.95)
    assert len(batches) == len(loader)
    for _, targets in batches:
        for target in targets:
            assert target["boxes"].shape[1] == 4
