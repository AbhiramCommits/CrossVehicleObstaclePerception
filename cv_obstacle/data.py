"""Dataset loaders for KITTI and BDD100K obstacle detection.

Both datasets expose the same tuple interface::

    image    : np.ndarray (H, W, 3) uint8 RGB
    boxes    : np.ndarray (N, 4) float32, absolute xyxy pixels
    class_ids: np.ndarray (N,) int64 in [1, num_classes]
    obstacles: list[Obstacle] aligned with ``boxes``

The ``truck`` camera domain applies a deterministic photometric /
geometric style transfer (documented in configs/truck_camera.yaml) to
simulate a different vehicle-mounted sensor.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Class ontology
# ---------------------------------------------------------------------------
CLASS_NAMES: Dict[int, str] = {1: "car", 2: "pedestrian", 3: "cyclist", 4: "truck"}
CLASS_IDS: Dict[str, int] = {v: k for k, v in CLASS_NAMES.items()}
NUM_CLASSES = 4

# KITTI label-file names -> our ontology.  Van is grouped with car because it
# shares the same visual/physical obstacle role; Tram with truck (large
# rail vehicle) and Person_sitting with pedestrian.
KITTI_NAME_TO_ID: Dict[str, int] = {
    "Car": 1,
    "Van": 1,
    "Pedestrian": 2,
    "Person_sitting": 2,
    "Cyclist": 3,
    "Truck": 4,
    "Tram": 4,
}
KITTI_IGNORE = {"DontCare", "Misc"}

# BDD100K detection categories -> our ontology.
BDD_NAME_TO_ID: Dict[str, int] = {
    "car": 1,
    "truck": 4,
    "bus": 4,
    "person": 2,
    "rider": 3,
    "bike": 3,
    "motor": 3,
}
BDD_IGNORE = {"traffic light", "traffic sign", "train", "other vehicle", "trailer"}


@dataclass
class Obstacle:
    """One annotated obstacle."""

    class_id: int
    class_name: str
    box: Tuple[float, float, float, float]

    def as_dict(self) -> Dict[str, Any]:
        return {"class_id": self.class_id, "class_name": self.class_name, "box": list(self.box)}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def clip_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """Clip xyxy boxes to the image and drop degenerate ones."""
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, width)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, height)
    keep = (boxes[:, 2] - boxes[:, 0] > 0.5) & (boxes[:, 3] - boxes[:, 1] > 0.5)
    return boxes[keep]


def _filter_pairs(
    boxes: np.ndarray,
    class_ids: np.ndarray,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(boxes) == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)
    raw = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    ids = np.asarray(class_ids, dtype=np.int64).reshape(-1)
    # A single-pixel (or slightly negative) box is retained when it is
    # physically valid by expanding it to at least 1x1 pixel.
    raw[:, 0::2] = np.clip(raw[:, 0::2], 0, max(width - 1, 0))
    raw[:, 1::2] = np.clip(raw[:, 1::2], 0, max(height - 1, 0))
    w = raw[:, 2] - raw[:, 0]
    h = raw[:, 3] - raw[:, 1]
    raw[:, 2] = np.where(w < 1.0, np.minimum(raw[:, 0] + 1.0, width), raw[:, 2])
    raw[:, 3] = np.where(h < 1.0, np.minimum(raw[:, 1] + 1.0, height), raw[:, 3])
    keep = (raw[:, 2] > raw[:, 0]) & (raw[:, 3] > raw[:, 1]) & (ids >= 1) & (ids <= NUM_CLASSES)
    return raw[keep], ids[keep]


# ---------------------------------------------------------------------------
# Camera-domain style transfer
# ---------------------------------------------------------------------------
@dataclass
class CameraStyle:
    """Deterministic truck/SUV camera simulation."""

    brightness: float = 1.10        # +10 %
    contrast: float = 1.05          # +5 %
    saturation: float = 0.92        # colour shift
    hue_shift: int = 6              # +- degrees, OpenCV H range
    zoom: float = 1.05              # +5 % crop-in
    gamma: float = 1.0

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "CameraStyle":
        names = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in names})

    def apply(self, image: np.ndarray, boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the style and adjust boxes to the new framing."""
        img = image.astype(np.float32) / 255.0
        img = np.clip(img * self.brightness, 0, 1)
        mean = img.mean(axis=(0, 1), keepdims=True)
        img = np.clip((img - mean) * self.contrast + mean, 0, 1)
        if self.gamma != 1.0:
            img = np.power(img, self.gamma)
        hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        hsv = hsv.astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + self.hue_shift) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * self.saturation, 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        h, w = img.shape[:2]
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
        if self.zoom > 1.0:
            nw, nh = int(round(w / self.zoom)), int(round(h / self.zoom))
            x0 = (w - nw) // 2
            y0 = (h - nh) // 2
            sx, sy = nw / w, nh / h
            img = img[y0:y0 + nh, x0:x0 + nw]
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            boxes[:, 0::2] = (boxes[:, 0::2] - x0) / sx * (w / nw)
            boxes[:, 1::2] = (boxes[:, 1::2] - y0) / sy * (h / nh)
            boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, w)
            boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, h)
        return np.ascontiguousarray(img.astype(np.uint8)), boxes


def _seed_for(name: str, seed: int = 42) -> int:
    return int(hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class _DetectionDataset(Dataset):
    """Shared split-file / annotation handling."""

    def __init__(self, root: str | Path, split: str, camera: str = "sedan",
                 style: CameraStyle | None = None, limit: int | None = None):
        self.root = Path(root)
        self.split = split
        self.camera = camera
        self.style = style or CameraStyle()
        self.records: List[Tuple[str, str]] = self._discover()
        if limit is not None:
            self.records = self.records[:limit]
        if not self.records:
            raise FileNotFoundError(
                f"no {split} samples under {self.root}; run `python scripts/download_data.py`"
            )

    def _discover(self) -> List[Tuple[str, str]]:
        raise NotImplementedError

    def _read(self, image_path: Path, label_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        image_path, label_path = self.records[index]
        image = cv2.cvtColor(cv2.imread(str(image_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        boxes, ids = self._read(image_path, Path(label_path))
        if self.camera == "truck":
            image, boxes = self.style.apply(image, boxes)
        boxes, ids = _filter_pairs(boxes, ids, image.shape[1], image.shape[0])
        obstacles = [Obstacle(int(i), CLASS_NAMES[int(i)], tuple(map(float, b)))
                     for b, i in zip(boxes, ids)]
        return image, boxes, ids, obstacles


class KittiDataset(_DetectionDataset):
    """KITTI 2D object detection (official label_2 annotations)."""

    def _discover(self) -> List[Tuple[str, str]]:
        img_dir = self.root / "images" / self.split
        lbl_dir = self.root / "labels" / self.split
        split_file = self.root / "splits" / f"{self.split}.txt"
        if split_file.exists():
            names = [ln.strip() for ln in split_file.read_text().splitlines() if ln.strip()]
        else:
            names = sorted(p.stem for p in img_dir.glob("*.png"))
        out = []
        for name in names:
            img = img_dir / f"{name}.png"
            lbl = lbl_dir / f"{name}.txt"
            if img.exists() and lbl.exists():
                out.append((str(img), str(lbl)))
        return out

    def _read(self, image_path: Path, label_path: Path):
        boxes, ids = [], []
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 15:
                continue
            name = parts[0]
            if name in KITTI_IGNORE:
                continue
            cid = KITTI_NAME_TO_ID.get(name)
            if cid is None:
                continue
            boxes.append([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])
            ids.append(cid)
        return np.asarray(boxes, np.float32).reshape(-1, 4), np.asarray(ids, np.int64)


class BDD100KDataset(_DetectionDataset):
    """BDD100K detection subset stored as per-image JSON (official schema)."""

    def _discover(self) -> List[Tuple[str, str]]:
        img_dir = self.root / "images" / self.split
        lbl_dir = self.root / "labels" / self.split
        split_file = self.root / "splits" / f"{self.split}.txt"
        if split_file.exists():
            names = [ln.strip() for ln in split_file.read_text().splitlines() if ln.strip()]
        else:
            names = sorted(p.stem for p in img_dir.glob("*.jpg"))
        out = []
        for name in names:
            img = img_dir / f"{name}.jpg"
            lbl = lbl_dir / f"{name}.json"
            if img.exists() and lbl.exists():
                out.append((str(img), str(lbl)))
        return out

    def _read(self, image_path: Path, label_path: Path):
        raw = json.loads(label_path.read_text())
        boxes, ids = [], []
        for cat, bbox in zip(raw.get("categories", []), raw.get("boxes", [])):
            cid = BDD_NAME_TO_ID.get(cat)
            if cid is None:
                continue
            boxes.append([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])])
            ids.append(cid)
        return np.asarray(boxes, np.float32).reshape(-1, 4), np.asarray(ids, np.int64)


class DomainAdaptationDataset(Dataset):
    """Paired source / (unlabelled) target samples for adversarial training."""

    def __init__(self, source_ds: Dataset, target_ds: Dataset, mix: bool = False,
                 seed: int = 42):
        self.source = source_ds
        self.target = target_ds
        self.mix = mix
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.source)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int):
        s_img, s_boxes, s_ids, _ = self.source[index]
        if self.mix:
            t_index = (index + self.epoch * 7919) % len(self.target)
        else:
            t_index = index % len(self.target)
        t_img, _, _, _ = self.target[t_index]
        return {
            "source_image": s_img,
            "target_image": t_img,
            "source_boxes": s_boxes,
            "source_labels": s_ids,
            "target_index": t_index,
        }


# ---------------------------------------------------------------------------
# Batching / loaders
# ---------------------------------------------------------------------------
def collate_detections(batch: Sequence[Any]):
    """Collate detection samples (tuple form or DomainAdaptation dicts)."""
    if isinstance(batch[0], dict):
        return {
            key: [sample[key] for sample in batch]
            for key in batch[0]
        }
    images, boxes, ids = [], [], []
    for image, box, cls, _ in batch:
        images.append(torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1))
        boxes.append(torch.from_numpy(box))
        ids.append(torch.from_numpy(cls))
    return images, boxes, ids


def make_loader(dataset: Dataset, batch_size: int = 4, shuffle: bool = True,
                num_workers: int = 0) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_detections,
        drop_last=False,
    )


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
DEFAULT_AUGMENTATION: Dict[str, Any] = {
    "RandomBrightnessContrast": {"brightness_limit": 0.2, "contrast_limit": 0.2, "p": 0.5},
    "GaussNoise": {"std_range": [0.02, 0.10], "p": 0.4},
    "RandomShadow": {"num_shadows_lower": 1, "num_shadows_upper": 2,
                     "shadow_roi": [0, 0.5, 1, 1], "p": 0.3},
    "RandomRain": {"slant_lower": -10, "slant_upper": 10, "drop_length": 20, "p": 0.2},
    "RandomFog": {"fog_coef_range": [0.1, 0.3], "p": 0.2},
}


def build_augmentation(albu_config: Dict[str, Any] | None = None, seed: int | None = 42):
    """Build an albumentations pipeline with bbox support."""
    cfg = albu_config if albu_config is not None else DEFAULT_AUGMENTATION
    transforms = []
    for name, params in cfg.items():
        cls = getattr(A, name, None)
        if cls is None:
            raise AttributeError(f"albumentations has no transform {name!r}")
        transforms.append(cls(**params))
    bbox_params = A.BboxParams(
        format="pascal_voc",
        label_fields=["class_ids"],
        min_area=1.0,
        min_visibility=0.0,
    )
    if seed is not None:
        try:
            return A.Compose(transforms, bbox_params=bbox_params, seed=seed)
        except TypeError:  # albumentations < 2 has no seed kwarg
            pass
    return A.Compose(transforms, bbox_params=bbox_params)


def augment_image(image: np.ndarray, boxes: np.ndarray,
                  albumentations_config: Dict[str, Any] | None = None,
                  class_ids: np.ndarray | None = None,
                  seed: int = 42):
    """Apply a standard driving-condition augmentation pipeline.

    Returns ``(image, boxes, class_ids)``.  Degenerate inputs (blank image,
    empty annotation list, 1-pixel boxes) are handled without raising.
    """
    if image.size == 0:
        raise ValueError("empty image")
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    h, w = image.shape[:2]
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if class_ids is None:
        class_ids = np.ones((len(boxes),), dtype=np.int64)
    class_ids = np.asarray(class_ids, dtype=np.int64).reshape(-1)
    if len(boxes) != len(class_ids):
        raise ValueError("boxes and class_ids must have the same length")

    boxes, class_ids = _filter_pairs(boxes, class_ids, w, h)
    # albumentations drops boxes with zero area: represent 1-pixel boxes with
    # a tiny positive area before the transform.
    degenerate = np.zeros((len(boxes),), dtype=bool)
    if len(boxes):
        degenerate = (boxes[:, 2] - boxes[:, 0] < 1.0) | (boxes[:, 3] - boxes[:, 1] < 1.0)
        boxes[degenerate, 2] = np.minimum(boxes[degenerate, 0] + 1.0, w)
        boxes[degenerate, 3] = np.minimum(boxes[degenerate, 1] + 1.0, h)

    pipeline = build_augmentation(albumentations_config, seed=seed)
    try:
        out = pipeline(image=image, bboxes=boxes.tolist(), class_ids=class_ids.tolist())
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"augmentation failed: {exc}") from exc

    out_img = np.ascontiguousarray(out["image"])
    out_boxes = np.asarray(out["bboxes"], dtype=np.float32).reshape(-1, 4)
    out_ids = np.asarray(out["class_ids"], dtype=np.int64).reshape(-1)
    out_boxes, out_ids = _filter_pairs(out_boxes, out_ids, out_img.shape[1], out_img.shape[0])
    return out_img, out_boxes, out_ids
