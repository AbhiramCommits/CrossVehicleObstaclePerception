#!/usr/bin/env python
"""Run the complete pipeline, record every command and its wall-clock time."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cv_obstacle.config import Config  # noqa: E402
from cv_obstacle.utils import environment_info, load_json, save_json  # noqa: E402


def run(cmd: list[str], env: dict) -> float:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    start = time.perf_counter()
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    return round(time.perf_counter() - start, 2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline_cpu.yaml")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--stages", default="all")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--dataset-root", default=None)
    args = parser.parse_args()

    cfg = Config.load(ROOT / args.config)
    if args.dataset_root:
        cfg.dataset_root = args.dataset_root
    python = sys.executable
    env = dict(os.environ)
    env["CV_DATASET_ROOT"] = cfg.dataset_root
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", str(os.cpu_count() or 4))

    train_epochs = 2 if args.quick else cfg.training.epochs
    scratch_epochs = 1
    adapt_epochs = 1 if args.quick else cfg.adaptation.epochs
    self_epochs = 1 if args.quick else cfg.adaptation.self_training_epochs
    distill_epochs = 2 if args.quick else cfg.distillation.epochs
    runs = 8 if args.quick else cfg.benchmark.num_runs
    iterations = 12 if args.quick else 30
    draws = 30 if args.quick else 80
    val_eval = 12 if args.quick else 24

    stages: list[tuple[str, list[str]]] = []
    if not args.skip_download:
        stages.append(("download", [python, "scripts/download_data.py"]))
    stages += [
        ("train", [python, "-m", "cv_obstacle.cli", "train", "--config", args.config,
                   "--dataset", "kitti", "--epochs", str(train_epochs),
                   "--scratch-epochs", str(scratch_epochs),
                   "--scratch-images", "24",
                   "--val-eval-images", str(val_eval),
                   "--out", "results/baseline_checkpoint.pt"]),
        ("adapt", [python, "-m", "cv_obstacle.cli", "adapt", "--config", args.config,
                   "--checkpoint", "results/baseline_checkpoint.pt",
                   "--epochs", str(adapt_epochs), "--max-images", "16",
                   "--self-training-epochs", str(self_epochs),
                   "--out", "results/adapted_checkpoint.pt"]),
        ("distill", [python, "-m", "cv_obstacle.cli", "distill", "--config", args.config,
                     "--teacher", "results/adapted_checkpoint.pt",
                     "--student", "yolov8n", "--epochs", str(distill_epochs),
                     "--out", "results/student_checkpoint.pt"]),
        ("benchmark", [python, "-m", "cv_obstacle.cli", "benchmark", "--config", args.config,
                       "--teacher", "results/adapted_checkpoint.pt",
                       "--student", "results/student_checkpoint.pt",
                       "--runs", str(runs),
                       "--out", "results/benchmark_report.json"]),
        ("optimize", [python, "-m", "cv_obstacle.cli", "optimize", "--config", args.config,
                      "--checkpoint", "results/student_checkpoint.pt",
                      "--dataset", "test_bdd100k", "--iterations", str(iterations),
                      "--out", "results/optimization_report.json"]),
        ("yield", [python, "-m", "cv_obstacle.cli", "yield", "--config", args.config,
                   "--checkpoint", "results/student_checkpoint.pt",
                   "--dataset", "test_bdd100k", "--draws", str(draws),
                   "--out", "results/monte_carlo_yield.json"]),
        ("plots", [python, "-m", "cv_obstacle.cli", "plot", "--config", args.config]),
        ("summary", [python, "-m", "cv_obstacle.cli", "summary", "--config", args.config]),
    ]
    selected = None if args.stages in {"all", ""} else set(args.stages.split(","))
    if selected:
        stages = [s for s in stages if s[0] in selected]

    manifest: dict = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "quick": args.quick,
        "config": args.config,
        "environment": environment_info(),
        "commands": {},
        "stage_seconds": {},
        "pipelines": {},
    }
    started = time.perf_counter()
    try:
        for name, cmd in stages:
            if name == "download":
                manifest["commands"][name] = " ".join(cmd[1:])
                manifest["stage_seconds"][name] = run(cmd, env)
                continue
            manifest["commands"][name] = " ".join(cmd[1:])
            manifest["stage_seconds"][name] = run(cmd, env)
    except subprocess.CalledProcessError as exc:
        manifest["error"] = f"stage failed: {exc}"
        manifest["runtime"] = {"total_seconds": round(time.perf_counter() - started, 2)}
        save_json(manifest, ROOT / cfg.results_dir / "run_manifest.json")
        print(f"pipeline failed: {exc}", file=sys.stderr)
        return 1

    total = round(time.perf_counter() - started, 2)
    manifest["runtime"] = {
        "total_seconds": total,
        "total_minutes": round(total / 60, 2),
        "stages": manifest["stage_seconds"],
        "gpu_hours": 0.0 if not (environment_info().get("cuda_available")) else round(total / 3600, 4),
        "cpu_hours": round(total / 3600, 4),
        "cpu_count": os.cpu_count(),
    }
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    # attach dataset provenance if available
    dataset_manifest = ROOT / cfg.dataset_root / "MANIFEST.json"
    if dataset_manifest.exists():
        manifest["dataset_manifest"] = load_json(dataset_manifest)
    save_json(manifest, ROOT / cfg.results_dir / "run_manifest.json")
    # regenerate summary so it includes total runtime
    subprocess.run([python, "-m", "cv_obstacle.cli", "summary", "--config", args.config],
                   cwd=ROOT, env=env, check=False)
    print(f"\nTOTAL {total:.1f}s ({total / 60:.2f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
