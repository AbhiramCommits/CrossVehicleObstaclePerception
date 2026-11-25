"""Tests for ONNX export, ONNX Runtime parity and latency/throughput."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from cv_obstacle.inference import (benchmark_latency, benchmark_throughput,
                                   load_onnx_model, quantize_int8, torch_to_onnx)
from cv_obstacle.models import ObjectDetectionModel, load_pretrained_detector
from cv_obstacle.utils import seed_everything


@pytest.fixture(scope="module")
def student():
    seed_everything(3)
    model, spec = load_pretrained_detector("yolov8n", num_classes=4, pretrained=False,
                                           min_size=128, max_size=128)
    return ObjectDetectionModel(model, spec)


def test_onnx_export_and_parity(student, tmp_path):
    path = tmp_path / "student.onnx"
    dummy = torch.rand(1, 3, 128, 128)
    torch_to_onnx(student, dummy, path, opset_version=14)
    assert path.exists() and path.stat().st_size > 0
    onnx_model = load_onnx_model(path)

    student.model.eval()
    tensor = torch.rand(3, 128, 128)
    with torch.no_grad():
        reference = student.predict([tensor])[0]
        exported = onnx_model(tensor)

    k = min(5, len(reference["scores"]))
    if k > 0:
        ref_scores = reference["scores"][:k].numpy()
        onx_scores = exported["scores"][:k].numpy()
        scale = max(float(np.abs(ref_scores).max()), 1e-6)
        assert np.abs(ref_scores - onx_scores).max() / scale < 0.01
        ref_boxes = reference["boxes"][:k].numpy()
        onx_boxes = exported["boxes"][:k].numpy()
        assert np.abs(ref_boxes - onx_boxes).max() / 128.0 < 0.01


def test_latency_and_throughput(student, tmp_path):
    path = tmp_path / "student.onnx"
    torch_to_onnx(student, torch.rand(1, 3, 128, 128), path)
    onnx_model = load_onnx_model(path)
    for model in (student, onnx_model):
        latency = benchmark_latency(model, (3, 128, 128), num_runs=5, warmup=1)
        assert 0 < latency["mean_ms"] < 1000
        assert latency["p95_ms"] >= latency["mean_ms"] * 0.5
        assert latency["throughput_ips"] > 0
        throughput = benchmark_throughput(model, batch_size=1, input_size=(3, 128, 128),
                                          num_runs=3)
        assert throughput["images_per_second"] > 0


def test_int8_quantisation_runs(student):
    quantized = quantize_int8(student)
    with torch.no_grad():
        out = quantized([torch.rand(3, 128, 128)])
    assert len(out) == 1
    assert out[0]["boxes"].shape[1] == 4
