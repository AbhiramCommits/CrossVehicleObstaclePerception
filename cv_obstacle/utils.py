"""Shared utilities: seeding, timing, JSON I/O, memory measurement."""
from __future__ import annotations

import json
import os
import platform
import random
import resource
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def configure_threads(num_threads: Optional[int] = None) -> int:
    """Use all available cores (torch defaults to a conservative count)."""
    if num_threads is None:
        num_threads = os.cpu_count() or 4
    torch.set_num_threads(int(num_threads))
    try:
        torch.set_num_interop_threads(max(1, int(num_threads) // 2))
    except RuntimeError:
        pass
    return int(num_threads)


def get_device(preferred: str = "auto") -> torch.device:
    if preferred == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preferred == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(preferred)


class Timer:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.laps: Dict[str, float] = {}

    def lap(self, name: str) -> float:
        elapsed = time.perf_counter() - self.t0
        self.laps[name] = round(elapsed, 3)
        self.t0 = time.perf_counter()
        return elapsed

    @property
    def elapsed(self) -> float:
        return round(sum(self.laps.values()), 3)


@contextmanager
def stage(timer: Timer, name: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        timer.laps[name] = round(time.perf_counter() - start, 3)


def peak_memory_mb() -> float:
    """Resident-set high-water mark for this process (portable fallback)."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return round(usage / (1024 * 1024), 1)  # bytes on macOS
    return round(usage / 1024.0, 1)  # KiB on Linux


def environment_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_hours": 0.0,
    }
    try:
        import torchvision
        info["torchvision"] = torchvision.__version__
    except ImportError:
        pass
    try:
        import cv2
        info["opencv"] = cv2.__version__
    except ImportError:
        pass
    try:
        import onnxruntime
        info["onnxruntime"] = onnxruntime.__version__
    except ImportError:
        pass
    try:
        import tensorrt  # noqa: F401
        info["tensorrt"] = getattr(tensorrt, "__version__", "installed")
    except Exception:  # noqa: BLE001
        info["tensorrt"] = "unavailable"
    try:
        info["cpu_model"] = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:  # noqa: BLE001
        pass
    return info


def save_json(payload: Dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(_jsonable(payload), fh, indent=2)
    return path


def load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value
