"""ONNX export, ONNX Runtime inference and latency / throughput measurement."""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import ObjectDetectionModel
from .utils import peak_memory_mb


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------
class OnnxDetectorWrapper(nn.Module):
    """Fixed-size ONNX-friendly output: padded boxes / scores / labels."""

    def __init__(self, model: torch.nn.Module, max_detections: int = 30):
        super().__init__()
        self.model = model
        self.max_detections = max_detections

    def forward(self, images: torch.Tensor):
        out = self.model([images[0]])[0]
        boxes, scores, labels = out["boxes"], out["scores"], out["labels"]
        k = self.max_detections
        # Concatenate k zero slots, then take the top-k by score.  This is
        # traceable for any number of detections (including zero) and always
        # produces a fixed-size tensor.
        boxes_p = torch.cat([boxes, boxes.new_zeros((k, 4))], dim=0)
        scores_p = torch.cat([scores, scores.new_zeros((k,))], dim=0)
        labels_p = torch.cat([labels, labels.new_zeros((k,))], dim=0)
        idx = torch.argsort(scores_p, descending=True)[:k]
        return boxes_p[idx].unsqueeze(0), scores_p[idx].unsqueeze(0), labels_p[idx].unsqueeze(0)


def torch_to_onnx(pytorch_model: ObjectDetectionModel | torch.nn.Module,
                  dummy_input: torch.Tensor, onnx_path: str | Path,
                  opset_version: int = 14, max_detections: int = 30) -> Path:
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    model = pytorch_model.model if isinstance(pytorch_model, ObjectDetectionModel) else pytorch_model
    model.eval()
    wrapper = OnnxDetectorWrapper(model, max_detections=max_detections).eval()
    kwargs: Dict[str, Any] = dict(
        input_names=["images"], output_names=["boxes", "scores", "labels"],
        opset_version=opset_version, do_constant_folding=True,
    )
    import inspect
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    with torch.no_grad():
        torch.onnx.export(wrapper, dummy_input, str(onnx_path), **kwargs)
    return onnx_path


class OnnxDetector:
    """ONNX Runtime adapter with the same call signature as the torch models."""

    def __init__(self, onnx_path: str | Path, device: str = "cpu"):
        import onnxruntime as ort
        providers = ["CPUExecutionProvider"]
        if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider"]
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_outputs()[0].shape
        self.max_detections = int(shape[1]) if len(shape) > 1 and isinstance(shape[1], int) else 30
        self.device = device
        self.onnx_path = str(onnx_path)

    def __call__(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        array = image.detach().cpu().numpy().astype(np.float32)
        if array.ndim == 3:
            array = array[None]
        boxes, scores, labels = self.session.run(None, {self.input_name: array})
        return {
            "boxes": torch.from_numpy(boxes[0]),
            "scores": torch.from_numpy(scores[0]),
            "labels": torch.from_numpy(labels[0]).long(),
        }


def load_onnx_model(onnx_path: str | Path, device: str = "cpu") -> OnnxDetector:
    return OnnxDetector(onnx_path, device=device)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def _make_input(input_size: Sequence[int], device: str) -> torch.Tensor:
    shape = tuple(int(v) for v in input_size)
    if len(shape) == 2:
        shape = (3, *shape)
    return torch.rand(*shape, device=device)


def _infer(model, tensor: torch.Tensor):
    if isinstance(model, OnnxDetector):
        return model(tensor)
    if isinstance(model, ObjectDetectionModel):
        return model.predict([tensor])
    if callable(model) and not isinstance(model, nn.Module):
        return model(tensor)
    model.eval()
    with torch.no_grad():
        return model([tensor])


def benchmark_latency(model, input_size: Sequence[int] = (3, 384, 384),
                      num_runs: int = 100, device: str = "cpu", warmup: int = 5,
                      seed: int = 42) -> Dict[str, Any]:
    if num_runs < 1:
        raise ValueError("num_runs must be positive")
    torch.manual_seed(seed)
    tensor = _make_input(input_size, device)
    times: List[float] = []
    is_torch = not isinstance(model, OnnxDetector)
    for _ in range(max(warmup, 1)):
        _infer(model, tensor)
    for _ in range(num_runs):
        if is_torch and device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        _infer(model, tensor)
        if is_torch and device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    arr = np.asarray(times, dtype=np.float64)
    return {
        "mean_ms": round(float(arr.mean()), 3),
        "median_ms": round(float(np.median(arr)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "min_ms": round(float(arr.min()), 3),
        "max_ms": round(float(arr.max()), 3),
        "std_ms": round(float(arr.std()), 3),
        "num_runs": int(num_runs),
        "warmup": int(warmup),
        "device": device,
        "input_size": list(input_size),
        "throughput_ips": round(1000.0 / max(float(arr.mean()), 1e-9), 2),
        "peak_memory_mb": peak_memory_mb(),
    }


def benchmark_throughput(model, batch_size: int = 1,
                         input_size: Sequence[int] = (3, 384, 384),
                         num_runs: int = 50, device: str = "cpu") -> Dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    tensors = [_make_input(input_size, device) for _ in range(batch_size)]
    is_torch = not isinstance(model, OnnxDetector)
    for _ in range(3):
        if is_torch:
            if isinstance(model, ObjectDetectionModel):
                model.predict(tensors)
            else:
                model(tensors)
        else:
            for t in tensors:
                model(t)
    if is_torch and device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        if is_torch:
            if isinstance(model, ObjectDetectionModel):
                model.predict(tensors)
            else:
                model(tensors)
        else:
            for t in tensors:
                model(t)
    if is_torch and device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    images = num_runs * batch_size
    return {
        "batch_size": int(batch_size),
        "num_runs": int(num_runs),
        "elapsed_s": round(elapsed, 3),
        "images_per_second": round(images / max(elapsed, 1e-9), 2),
        "mean_latency_ms": round(elapsed * 1000.0 / num_runs, 3),
        "device": device,
    }


# ---------------------------------------------------------------------------
# Quantisation
# ---------------------------------------------------------------------------
def quantize_int8(pytorch_model, calibration_loader=None, device: str = "cpu"):
    """Dynamic int8 quantisation of the detector's linear layers.

    Dynamic quantisation needs no calibration to run; when a loader is supplied
    it is executed first so the reported latency includes a warmed model.
    """
    engines = torch.backends.quantized.supported_engines
    if not engines:
        raise RuntimeError("this PyTorch build has no quantized CPU engine")
    if torch.backends.quantized.engine == "none":
        torch.backends.quantized.engine = "qnnpack" if "qnnpack" in engines else engines[0]
    model = pytorch_model.model if isinstance(pytorch_model, ObjectDetectionModel) else pytorch_model
    model = model.cpu().eval()
    quantized = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    if calibration_loader is not None:
        with torch.no_grad():
            for i, batch in enumerate(calibration_loader):
                images = batch[0] if isinstance(batch, (tuple, list)) else batch["images"]
                images = [img.float() / 255.0 if img.dtype == torch.uint8 else img.float()
                          for img in images]
                quantized(images)
                if i >= 2:
                    break
    quantized.eval()
    return quantized
