"""Detector construction, checkpointing and feature access.

Two genuine torchvision Faster R-CNN foundation models are used:

* teacher : ``fasterrcnn_resnet50_fpn_v2`` pretrained on COCO (43.3M params)
* student : a slim Faster R-CNN with a ResNet-18 FPN backbone, 128-channel FPN,
            256-d box head (~14M params)

The COCO classifier is *surgically* remapped to the 4-class obstacle ontology so
transfer learning starts from real pretrained semantics instead of a randomly
initialised head.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from .data import CLASS_NAMES, NUM_CLASSES

# COCO category -> our obstacle class id (torchvision COCO label indices)
COCO_TO_OBSTACLE: Dict[int, int] = {3: 1, 1: 2, 2: 3, 8: 4}


@dataclass
class ModelSpec:
    kind: str = "resnet50_fpn_v2"  # resnet50_fpn_v2 | slim_resnet18_fpn
    num_classes: int = NUM_CLASSES
    min_size: int = 320
    max_size: int = 320
    fpn_out: int = 128
    box_head_dim: int = 256
    pretrained: bool = True


# ---------------------------------------------------------------------------
# Backbone construction
# ---------------------------------------------------------------------------
def _resnet_backbone(backbone_name: str, pretrained: bool, fpn_out: int = 128,
                     freeze: bool = True) -> BackboneWithFPN:
    weights = None
    if pretrained:
        weights = getattr(torchvision.models, f"{backbone_name}_weights", None)
        if weights is None:
            from torchvision.models import get_model_weights
            weights = get_model_weights(backbone_name).DEFAULT
    body = torchvision.models.get_model(backbone_name, weights=weights)
    in_channels = {
        "resnet18": [64, 128, 256, 512],
        "resnet34": [64, 128, 256, 512],
        "resnet50": [256, 512, 1024, 2048],
    }[backbone_name]
    return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
    backbone = BackboneWithFPN(
        body, return_layers, in_channels, fpn_out,
        extra_blocks=LastLevelMaxPool(), norm_layer=nn.BatchNorm2d,
    )
    if freeze:
        for name, param in backbone.body.named_parameters():
            param.requires_grad_(False)
    return backbone


def build_slim_detector(num_classes: int = NUM_CLASSES, min_size: int = 320,
                        max_size: int = 320, pretrained: bool = True,
                        fpn_out: int = 128, box_head_dim: int = 256,
                        rpn_post_nms: int = 150,
                        detections_per_img: int = 50) -> torch.nn.Module:
    """Lightweight student: ResNet-18 FPN, narrow head."""
    backbone = _resnet_backbone("resnet18", pretrained=pretrained, fpn_out=fpn_out, freeze=True)
    # four backbone levels + the FPN "pool" level (stride 64)
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
    model = torchvision.models.detection.FasterRCNN(
        backbone,
        num_classes=None,
        min_size=min_size,
        max_size=max_size,
        rpn_anchor_generator=anchor_generator,
        rpn_pre_nms_top_n_train=600,
        rpn_pre_nms_top_n_test=400,
        rpn_post_nms_top_n_train=rpn_post_nms,
        rpn_post_nms_top_n_test=rpn_post_nms,
        rpn_batch_size_per_image=128,
        box_batch_size_per_image=256,
        box_detections_per_img=detections_per_img,
        box_score_thresh=0.05,
        box_head=TwoMLPHead(fpn_out * 7 * 7, box_head_dim),
        box_predictor=FastRCNNPredictor(box_head_dim, num_classes + 1),
    )
    return model


def _coco_surgery(model: torch.nn.Module, num_classes: int,
                  weights_meta: Dict[str, Any]) -> None:
    """Copy COCO class rows into the new 4-class predictor."""
    cats: Sequence[str] = weights_meta["categories"]
    old = model.roi_heads.box_predictor
    in_channels = old.cls_score.in_features
    new = FastRCNNPredictor(in_channels, num_classes + 1)
    with torch.no_grad():
        new.cls_score.weight[0] = old.cls_score.weight[0]
        new.cls_score.bias[0] = old.cls_score.bias[0]
        new.bbox_pred.weight[0] = old.bbox_pred.weight[0]
        new.bbox_pred.bias[0] = old.bbox_pred.bias[0]
        for coco_idx, obstacle_id in COCO_TO_OBSTACLE.items():
            if coco_idx >= len(cats):
                continue
            new.cls_score.weight[obstacle_id] = old.cls_score.weight[coco_idx]
            new.cls_score.bias[obstacle_id] = old.cls_score.bias[coco_idx]
            start = coco_idx * 4
            new.bbox_pred.weight[obstacle_id * 4:(obstacle_id + 1) * 4] = \
                old.bbox_pred.weight[start:start + 4]
            new.bbox_pred.bias[obstacle_id * 4:(obstacle_id + 1) * 4] = \
                old.bbox_pred.bias[start:start + 4]
    model.roi_heads.box_predictor = new


def load_pretrained_detector(model_name: str = "yolov8m", num_classes: int = NUM_CLASSES,
                             pretrained: bool = True, min_size: int = 320,
                             max_size: int = 320) -> Tuple[torch.nn.Module, ModelSpec]:
    """Build a foundation detector and remap its classifier to obstacle classes.

    ``model_name`` accepts ``yolov8m``/``yolov8n`` aliases for the torchvision
    Faster R-CNN family (the implementation used in this repo), plus the
    explicit names ``fasterrcnn_resnet50_fpn_v2``, ``fasterrcnn_r50`` and
    ``student``.  YOLO weights are not required at runtime.
    """
    name = model_name.lower()
    if name in {"yolov8m", "fasterrcnn_resnet50_fpn_v2", "fasterrcnn_r50", "teacher"}:
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None
        model = fasterrcnn_resnet50_fpn_v2(
            weights=weights, weights_backbone=None, num_classes=91,
            min_size=min_size, max_size=max_size,
            rpn_pre_nms_top_n_train=800, rpn_pre_nms_top_n_test=500,
            rpn_post_nms_top_n_train=200, rpn_post_nms_top_n_test=200,
            rpn_batch_size_per_image=128,
            box_batch_size_per_image=256,
            box_detections_per_img=50,
        )
        if pretrained:
            _coco_surgery(model, num_classes, weights.meta)
        else:
            in_ch = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_ch, num_classes + 1)
        spec = ModelSpec("resnet50_fpn_v2", num_classes, min_size, max_size,
                         fpn_out=256, box_head_dim=1024, pretrained=pretrained)
        return model, spec
    if name in {"yolov8n", "student", "fasterrcnn_resnet18_fpn", "slim"}:
        model = build_slim_detector(num_classes=num_classes, min_size=min_size,
                                    max_size=max_size, pretrained=pretrained)
        return model, ModelSpec("slim_resnet18_fpn", num_classes, min_size, max_size,
                                fpn_out=128, box_head_dim=256, pretrained=pretrained)
    raise ValueError(f"unknown detector {model_name!r}")


def build_detector(spec: Dict[str, Any], pretrained: bool = False) -> torch.nn.Module:
    """Rebuild a detector from a checkpoint's recorded spec."""
    spec = dict(spec)
    spec["pretrained"] = pretrained
    if spec.get("kind", "resnet50_fpn_v2") == "slim_resnet18_fpn":
        return build_slim_detector(
            num_classes=spec.get("num_classes", NUM_CLASSES),
            min_size=spec.get("min_size", 320), max_size=spec.get("max_size", 320),
            pretrained=pretrained, fpn_out=spec.get("fpn_out", 128),
            box_head_dim=spec.get("box_head_dim", 256),
        )
    model, _ = load_pretrained_detector(
        "fasterrcnn_resnet50_fpn_v2", num_classes=spec.get("num_classes", NUM_CLASSES),
        pretrained=pretrained, min_size=spec.get("min_size", 320),
        max_size=spec.get("max_size", 320),
    )
    return model


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------
def split_transform_output(output):
    """Normalise torchvision's transform output across versions / tracing.

    Returns ``(tensor_batch, image_sizes)``.
    """
    image_list = output
    if isinstance(output, tuple) and not hasattr(output, "tensors"):
        first = output[0]
        if hasattr(first, "tensors") or torch.is_tensor(first):
            image_list = first
    if hasattr(image_list, "tensors"):
        return image_list.tensors, list(image_list.image_sizes)
    if torch.is_tensor(image_list):
        return image_list, None
    if isinstance(image_list, (tuple, list)) and len(image_list) == 2:
        return image_list[0], list(image_list[1])
    raise TypeError(f"unexpected transform output: {type(output)!r}")


class ObjectDetectionModel(nn.Module):
    """Uniform interface used by training / adaptation / distillation."""

    def __init__(self, model: torch.nn.Module, spec: ModelSpec, device: str = "cpu"):
        super().__init__()
        self.model = model
        self.spec = spec
        self.device = torch.device(device)
        self._captured: Optional[torch.Tensor] = None
        self._backbone_features: Optional[Dict[str, torch.Tensor]] = None
        self._image_sizes: Optional[List[Tuple[int, int]]] = None
        self._handles = [
            self.model.roi_heads.box_head.register_forward_hook(self._capture),
            self.model.backbone.register_forward_hook(self._capture_backbone),
            self.model.transform.register_forward_hook(self._capture_transform),
        ]

    def _capture(self, module, inputs, output):  # noqa: ANN001
        self._captured = output

    def _capture_backbone(self, module, inputs, output):  # noqa: ANN001
        self._backbone_features = output

    def _capture_transform(self, module, inputs, output):  # noqa: ANN001
        # In inference torchvision returns an ImageList; in training it returns
        # (ImageList, targets).  Tracing may downgrade ImageList to a plain
        # (tensor, sizes) tuple.  Be liberal about all three shapes.
        candidate = output
        if isinstance(output, tuple) and len(output) == 2 and not hasattr(output, "image_sizes") \
                and hasattr(output[0], "image_sizes"):
            candidate = output[0]
        sizes = getattr(candidate, "image_sizes", None)
        if sizes is None and isinstance(candidate, (tuple, list)) and len(candidate) == 2 \
                and isinstance(candidate[1], (list, tuple)):
            sizes = candidate[1]
        if sizes is not None:
            try:
                self._image_sizes = [tuple(int(v) for v in s) for s in sizes]
            except (TypeError, ValueError):
                self._image_sizes = None

    # -- lifecycle -------------------------------------------------------
    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self.clear_features()

    def clear_features(self) -> None:
        self._captured = None
        self._backbone_features = None
        self._image_sizes = None

    def forward(self, images: Sequence[torch.Tensor],
                targets: Optional[Sequence[Dict[str, torch.Tensor]]] = None):
        return self.model(images, targets)

    def predict(self, images: Sequence[torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            out = self.model(images)
        if was_training:
            self.model.train()
        return out

    def loss(self, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not isinstance(predictions, dict) or not predictions:
            raise ValueError("expected a dict of detection losses")
        return sum(v for v in predictions.values() if torch.is_tensor(v))

    @property
    def roi_features(self) -> Optional[torch.Tensor]:
        return self._captured

    @property
    def captured_image_sizes(self) -> Optional[List[Tuple[int, int]]]:
        return self._image_sizes

    def logits_at_captured_boxes(self, boxes_per_image: List[torch.Tensor]) -> List[torch.Tensor]:
        """ROI-head class logits for boxes, reusing the last forward's features."""
        if self._backbone_features is None or self._image_sizes is None:
            raise RuntimeError("no captured backbone features; run a forward pass first")
        out: List[torch.Tensor] = []
        for index, boxes in enumerate(boxes_per_image):
            if boxes.numel() == 0:
                out.append(torch.zeros((0, self.spec.num_classes + 1), device=boxes.device))
                continue
            feats = {k: v[index:index + 1] for k, v in self._backbone_features.items()}
            out.append(self.roi_class_logits(feats, boxes,
                                             image_shapes=[self._image_sizes[index]]))
        return out

    # -- transfer-learning controls -------------------------------------
    def freeze_backbone(self) -> None:
        for p in self.model.backbone.parameters():
            p.requires_grad_(False)

    def unfreeze_head(self) -> None:
        self.freeze_backbone()
        for module in (self.model.rpn, self.model.roi_heads):
            for p in module.parameters():
                p.requires_grad_(True)

    def unfreeze_all(self) -> None:
        for p in self.model.parameters():
            p.requires_grad_(True)

    def train_predictor_only(self) -> None:
        """Freeze backbone, RPN and box head; adapt only the class/box predictor.

        This is the safest transfer-learning regime when only a few hundred
        target frames are available: the COCO features and proposal quality are
        preserved while the 4-class classifier is re-calibrated.
        """
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in self.model.roi_heads.box_predictor.parameters():
            p.requires_grad_(True)

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]

    def parameter_counts(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {"total": int(total), "trainable": int(trainable),
                "frozen": int(total - trainable)}

    # -- feature access ---------------------------------------------------
    def backbone_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run the (frozen) FPN backbone on a batched tensor."""
        tensors, _ = split_transform_output(self.model.transform(images))
        return self.model.backbone(tensors)

    def roi_class_logits(self, features: Dict[str, torch.Tensor],
                         boxes: torch.Tensor,
                         image_shapes: Optional[Sequence[Tuple[int, int]]] = None) -> torch.Tensor:
        """Class logits for arbitrary boxes through this model's ROI head.

        ``features`` come from :meth:`backbone_features`.  When ``image_shapes``
        is omitted the input size is inferred from the stride-4 level.
        """
        if boxes.numel() == 0:
            return torch.zeros((0, self.spec.num_classes + 1), device=boxes.device)
        if image_shapes is None:
            h4, w4 = features["0"].shape[-2:]
            image_shapes = [(int(h4 * 4), int(w4 * 4))]
        pooled = self.model.roi_heads.box_roi_pool(features, [boxes], list(image_shapes))
        head = self.model.roi_heads.box_head(pooled)
        return self.model.roi_heads.box_predictor(head)[0]


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_model(model: ObjectDetectionModel | torch.nn.Module, path: str | Path,
               metadata: Optional[Dict[str, Any]] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(model, ObjectDetectionModel):
        payload = {
            "state_dict": model.model.state_dict(),
            "spec": vars(model.spec),
            "metadata": metadata or {},
        }
    else:
        payload = {"state_dict": model.state_dict(), "metadata": metadata or {}}
    torch.save(payload, path)
    return path


def load_model(checkpoint_path: str | Path, device: str = "cpu",
               wrapper: bool = True):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    spec = payload.get("spec", {"kind": "resnet50_fpn_v2", "num_classes": NUM_CLASSES})
    model = build_detector(spec, pretrained=False)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    if wrapper:
        return ObjectDetectionModel(model, ModelSpec(**spec), device=device)
    return model


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
