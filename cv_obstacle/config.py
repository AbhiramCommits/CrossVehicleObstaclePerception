"""Typed configuration objects with YAML (de)serialisation.

All tunable knobs used by the pipeline live here.  YAML files may be partial:
values they omit fall back to the defaults below.
"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type, TypeVar

import yaml

T = TypeVar("T")

OBSTACLE_CLASSES: List[str] = ["car", "pedestrian", "cyclist", "truck"]


@dataclass
class OpticalConfig:
    """Camera / input-side settings."""

    input_resolution: int = 320
    num_classes: int = 4
    obstacle_classes: List[str] = field(default_factory=lambda: list(OBSTACLE_CLASSES))

    def __post_init__(self) -> None:
        if len(self.obstacle_classes) != self.num_classes:
            raise ValueError(
                f"obstacle_classes has {len(self.obstacle_classes)} entries but "
                f"num_classes={self.num_classes}"
            )


@dataclass
class TrainingConfig:
    batch_size: int = 4
    epochs: int = 20
    learning_rate: float = 5e-4
    optimizer: str = "adamw"  # adamw | sgd
    weight_decay: float = 1e-4
    momentum: float = 0.9
    num_workers: int = 0
    seed: int = 42
    # torchvision Faster R-CNN geometry
    min_size: int = 320
    max_size: int = 320
    # transfer-learning control
    freeze_backbone: bool = True
    # kept so short CPU runs still learn a useful head
    lr_scheduler: str = "cosine"  # cosine | none
    grad_clip: float = 10.0


@dataclass
class AdaptationConfig:
    domain_adaptation_method: str = "dann"  # dann | none
    num_discriminator_layers: int = 3
    lambda_adv: float = 1.0
    epochs: int = 20
    learning_rate: float = 2e-4
    batch_size: int = 2
    # pseudo-label self-training phase (unsupervised on target images)
    self_training_epochs: int = 2
    pseudo_label_threshold: float = 0.70
    pseudo_label_iou: float = 0.5
    seed: int = 42


@dataclass
class DistillationConfig:
    temperature: float = 4.0
    alpha: float = 0.7
    loss_weight: float = 1.0
    epochs: int = 30
    learning_rate: float = 5e-4
    batch_size: int = 4
    max_teacher_proposals: int = 32
    feature_distill_weight: float = 0.0
    seed: int = 42


@dataclass
class BenchmarkConfig:
    num_runs: int = 200
    device: str = "cpu"
    input_size: Tuple[int, int, int] = (3, 384, 384)
    warmup: int = 5
    batch_size: int = 1


@dataclass
class Config:
    optical: OpticalConfig = field(default_factory=OpticalConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    seed: int = 42
    dataset_root: str = "datasets"
    results_dir: str = "results"
    figures_dir: str = "figures"
    # dataset sizes used by the CPU pipeline
    kitti_train_size: int = 240
    kitti_val_size: int = 80
    bdd_train_size: int = 240
    bdd_val_size: int = 80
    bdd_test_size: int = 80
    # camera domains
    source_camera: str = "sedan"
    target_camera: str = "truck"
    # exact camera-domain style transfer, documented in configs/truck_camera.yaml
    camera_style: Dict[str, float] = field(default_factory=lambda: {
        "brightness": 1.10,
        "contrast": 1.05,
        "saturation": 0.92,
        "hue_shift": 6,
        "zoom": 1.05,
        "gamma": 1.0,
    })

    # ------------------------------------------------------------------ YAML
    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Config":
        if not isinstance(raw, dict):
            raise TypeError(f"config must be a mapping, got {type(raw)!r}")
        fields = {f.name: f for f in dataclasses.fields(cls)}
        kwargs: Dict[str, Any] = {}
        for key, value in raw.items():
            if key not in fields:
                raise KeyError(f"unknown config key: {key!r}")
            ftype = fields[key].type
            sub = _resolve_dataclass(ftype)
            if sub is not None and isinstance(value, dict):
                kwargs[key] = _dataclass_from_dict(sub, value)
            else:
                kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        cfg = cls.from_dict(raw)
        env_root = os.environ.get("CV_DATASET_ROOT")
        if env_root:
            cfg.dataset_root = env_root
        return cfg

    @classmethod
    def load_many(cls, *paths: str | Path) -> "Config":
        merged: Dict[str, Any] = {}
        for path in paths:
            with Path(path).open("r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
            _deep_update(merged, raw)
        return cls.from_dict(merged)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False, indent=2)
        return path


def _resolve_dataclass(ftype: Any) -> Type[Any] | None:
    if isinstance(ftype, str):
        ftype = {
            "OpticalConfig": OpticalConfig,
            "TrainingConfig": TrainingConfig,
            "AdaptationConfig": AdaptationConfig,
            "DistillationConfig": DistillationConfig,
            "BenchmarkConfig": BenchmarkConfig,
        }.get(ftype)
    return ftype if isinstance(ftype, type) and dataclasses.is_dataclass(ftype) else None


def _dataclass_from_dict(dc: Type[T], raw: Dict[str, Any]) -> T:
    names = {f.name for f in dataclasses.fields(dc)}
    unknown = set(raw) - names
    if unknown:
        raise KeyError(f"unknown keys for {dc.__name__}: {sorted(unknown)}")
    kwargs: Dict[str, Any] = {}
    for key, value in raw.items():
        ftype = next(f.type for f in dataclasses.fields(dc) if f.name == key)
        sub = _resolve_dataclass(ftype)
        kwargs[key] = _dataclass_from_dict(sub, value) if sub and isinstance(value, dict) else value
    return dc(**kwargs)


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst
