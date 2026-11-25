"""Tiny, fast fixtures shared by the test-suite.

These fixtures only exist so the *plumbing* (loaders, losses, export, metrics) can
be tested without a 500 MB download.  All pipeline experiments use the real
KITTI/BDD100K data fetched by ``scripts/download_data.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch


def _write_scene(path: Path, width: int, height: int, seed: int,
                 boxes: list[tuple[int, int, int, int, int]]) -> None:
    rng = np.random.default_rng(seed)
    image = np.zeros((height, width, 3), np.uint8)
    image[:, :] = (40, 42, 45)
    cv2.rectangle(image, (0, int(height * 0.55)), (width, height), (70, 70, 75), -1)
    cv2.line(image, (0, int(height * 0.55)), (width, int(height * 0.55)), (210, 210, 210), 1)
    for (x1, y1, x2, y2, class_id) in boxes:
        color = {1: (180, 60, 60), 2: (60, 180, 60), 3: (60, 60, 200), 4: (200, 180, 60)}[class_id]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(image, (x1, y1), (x2, y2), (250, 250, 250), 1)
    image = np.clip(image.astype(np.int16) + rng.integers(-8, 8, image.shape), 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), image)


KITTI_NAMES = {1: "Car", 2: "Pedestrian", 3: "Cyclist", 4: "Truck"}
BDD_NAMES = {1: "car", 2: "person", 3: "rider", 4: "truck"}


def _make_kitti(root: Path, split: str, count: int, seed: int) -> list[str]:
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    names = []
    for i in range(count):
        name = f"{seed + i:06d}"
        boxes = [
            (40 + i % 3 * 8, 70, 110 + i % 3 * 8, 130, 1 + (i % 4)),
            (150 + i % 2 * 10, 90, 190 + i % 2 * 10, 150, 2),
        ]
        _write_scene(root / "images" / split / f"{name}.png", 320, 180, seed + i, boxes)
        lines = []
        for x1, y1, x2, y2, cid in boxes:
            lines.append(f"{KITTI_NAMES[cid]} 0.0 0 0 {x1} {y1} {x2} {y2} 1.5 1.6 3.0 0 0 0 0")
        (root / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + "\n")
        names.append(name)
    (root / "splits").mkdir(parents=True, exist_ok=True)
    (root / "splits" / f"{split}.txt").write_text("\n".join(names) + "\n")
    return names


def _make_bdd(root: Path, split: str, count: int, seed: int) -> list[str]:
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    names = []
    for i in range(count):
        name = f"bdd-{split}-{seed + i:05d}"
        boxes = [
            (60 + i % 4 * 6, 80, 130 + i % 4 * 6, 150, 1),
            (200 + i % 2 * 8, 100, 245 + i % 2 * 8, 160, 2),
        ]
        _write_scene(root / "images" / split / f"{name}.jpg", 480, 270, seed + i, boxes)
        payload = {
            "image_id": name, "width": 480, "height": 270,
            "weather": "clear", "timeofday": "daytime", "scene": "city street",
            "categories": [BDD_NAMES[c] for *_, c in boxes],
            "boxes": [[x1, y1, x2, y2] for x1, y1, x2, y2, _ in boxes],
        }
        (root / "labels" / split / f"{name}.json").write_text(json.dumps(payload))
        names.append(name)
    (root / "splits").mkdir(parents=True, exist_ok=True)
    (root / "splits" / f"{split}.txt").write_text("\n".join(names) + "\n")
    return names


@pytest.fixture(scope="session")
def toy_dataset(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("toy_datasets")
    _make_kitti(root / "kitti", "train", 8, 1000)
    _make_kitti(root / "kitti", "val", 4, 2000)
    _make_bdd(root / "bdd100k", "train", 8, 3000)
    _make_bdd(root / "bdd100k", "val", 4, 4000)
    _make_bdd(root / "bdd100k", "test", 4, 5000)
    return root


@pytest.fixture(scope="session")
def tiny_model():
    from cv_obstacle.models import ObjectDetectionModel, load_pretrained_detector
    model, spec = load_pretrained_detector("yolov8n", num_classes=4, pretrained=False,
                                           min_size=128, max_size=128)
    return ObjectDetectionModel(model, spec, device="cpu")
