"""Full latency / throughput / size benchmark across model variants."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch

from .inference import (OnnxDetector, benchmark_latency, benchmark_throughput,
                        load_onnx_model, quantize_int8, torch_to_onnx)
from .models import ObjectDetectionModel, load_model, count_parameters
from .utils import environment_info, peak_memory_mb, save_json


def _safe_size_mb(path: str | Path) -> float:
    return round(Path(path).stat().st_size / 1e6, 2)


def full_benchmark(teacher_checkpoint: str | Path, student_checkpoint: str | Path,
                   num_runs: int = 200, device: str = "cpu",
                   input_size: Sequence[int] = (3, 384, 384),
                   out: Optional[str | Path] = None,
                   onnx_path: Optional[str | Path] = None,
                   warmup: int = 5) -> Dict[str, Any]:
    started = time.perf_counter()
    teacher = load_model(teacher_checkpoint, device=device)
    student = load_model(student_checkpoint, device=device)
    # bench at the requested resolution regardless of the training resolution
    for wrapped in (teacher, student):
        wrapped.model.transform.min_size = (int(input_size[-2]),)
        wrapped.model.transform.max_size = int(input_size[-1])
    teacher.model.eval()
    student.model.eval()

    if onnx_path is None:
        out_dir = Path(out).parent if out is not None else Path("results")
        onnx_path = out_dir / "student.onnx"
    dummy = torch.rand(1, *input_size, device=device)
    t0 = time.perf_counter()
    torch_to_onnx(student, dummy, onnx_path, opset_version=14)
    export_seconds = round(time.perf_counter() - t0, 2)
    onnx_model = load_onnx_model(onnx_path, device=device)

    results: Dict[str, Any] = {
        "device": device,
        "input_size": list(input_size),
        "num_runs": int(num_runs),
        "warmup": int(warmup),
        "teacher": {
            "params": count_parameters(teacher.model),
            "checkpoint_mb": _safe_size_mb(teacher_checkpoint),
            "latency": benchmark_latency(teacher, input_size, num_runs, device, warmup),
        },
        "student": {
            "params": count_parameters(student.model),
            "checkpoint_mb": _safe_size_mb(student_checkpoint),
            "latency": benchmark_latency(student, input_size, num_runs, device, warmup),
        },
        "student_onnx": {
            "path": str(onnx_path),
            "size_mb": _safe_size_mb(onnx_path),
            "export_seconds": export_seconds,
            "latency": benchmark_latency(onnx_model, input_size, num_runs, device, warmup),
        },
        "throughput": {},
        "param_reduction_pct": None,
        "speedup": {},
        "environment": environment_info(),
        "peak_memory_mb": peak_memory_mb(),
    }
    results["student_onnx"]["latency"].setdefault(
        "throughput_ips", round(1000.0 / max(results["student_onnx"]["latency"]["mean_ms"], 1e-9), 2))

    # int8 dynamic quantisation (optional but supported on all CPU builds)
    try:
        quantized = quantize_int8(student)
        q_latency = benchmark_latency(quantized, input_size, max(5, num_runs // 4), device, warmup)
        results["student_int8"] = {
            "latency": q_latency,
            "size_mb": round(sum(p.numel() * p.element_size() for p in quantized.parameters()) / 1e6, 2),
        }
    except Exception as exc:  # noqa: BLE001
        results["student_int8"] = {"error": f"{type(exc).__name__}: {exc}"}

    # throughput at batch size 1 (measured on the actual model objects)
    results["throughput"]["teacher"] = benchmark_throughput(
        teacher, batch_size=1, input_size=input_size, num_runs=max(5, num_runs // 4), device=device)
    results["throughput"]["student"] = benchmark_throughput(
        student, batch_size=1, input_size=input_size, num_runs=max(5, num_runs // 4), device=device)
    results["throughput"]["student_onnx"] = benchmark_throughput(
        onnx_model, batch_size=1, input_size=input_size, num_runs=max(5, num_runs // 4), device=device)

    t_ms = results["teacher"]["latency"]["mean_ms"]
    s_ms = results["student"]["latency"]["mean_ms"]
    o_ms = results["student_onnx"]["latency"]["mean_ms"]
    results["speedup"] = {
        "student_vs_teacher": round(t_ms / max(s_ms, 1e-9), 2),
        "student_onnx_vs_teacher": round(t_ms / max(o_ms, 1e-9), 2),
        "student_onnx_vs_student": round(s_ms / max(o_ms, 1e-9), 2),
    }
    if "error" not in results["student_int8"]:
        q_ms = results["student_int8"]["latency"]["mean_ms"]
        results["speedup"]["student_int8_vs_teacher"] = round(t_ms / max(q_ms, 1e-9), 2)
    t_params = results["teacher"]["params"]
    s_params = results["student"]["params"]
    results["param_reduction_pct"] = round(100.0 * (1 - s_params / t_params), 2)
    results["elapsed_s"] = round(time.perf_counter() - started, 2)

    teacher.close()
    student.close()
    if out is not None:
        save_json(results, out)
    return results
