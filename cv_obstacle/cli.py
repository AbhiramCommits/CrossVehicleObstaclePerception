"""Command line interface for every pipeline stage."""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Subset

from .adaptation import (DomainDiscriminator, adapt_model, build_discriminator,
                         self_train)
from .benchmarking import full_benchmark
from .config import Config
from .data import (BDD100KDataset, CameraStyle, KittiDataset, DomainAdaptationDataset,
                   make_loader)
from .distillation import create_student_model, train_with_distillation
from .inference import (benchmark_latency, load_onnx_model, quantize_int8,
                        torch_to_onnx)
from .mc_yield import run_monte_carlo_yield
from .models import (ObjectDetectionModel, load_model, load_pretrained_detector,
                     save_model)
from .optimization import (CONDITIONS, RobustnessConfig, optimize_robustness,
                           verify_gradients, evaluate_with_preprocessor_conditions)
from .training import evaluate, make_optimizer, train_transfer
from .utils import (configure_threads, environment_info, get_device, load_json,
                    save_json, seed_everything)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else _repo_root() / path


def _command() -> str:
    return "python -m cv_obstacle.cli " + " ".join(sys.argv[1:])


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=_repo_root(), text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def _base_metadata(cfg: Config, stage: str) -> Dict[str, Any]:
    return {
        "stage": stage,
        "command": _command(),
        "git_commit": _git_commit(),
        "seed": cfg.seed,
        "environment": environment_info(),
        "config": cfg.to_dict(),
    }


def build_datasets(cfg: Config, camera_target: str = "truck") -> Dict[str, Any]:
    root = _resolve(cfg.dataset_root)
    kitti_root = root / "kitti"
    bdd_root = root / "bdd100k"
    style = CameraStyle.from_dict(cfg.camera_style)
    return {
        "kitti_train": KittiDataset(kitti_root, "train", limit=cfg.kitti_train_size),
        "kitti_val": KittiDataset(kitti_root, "val", limit=cfg.kitti_val_size),
        "bdd_train_sedan": BDD100KDataset(bdd_root, "train", camera="sedan",
                                         limit=cfg.bdd_train_size),
        "bdd_train": BDD100KDataset(bdd_root, "train", camera=camera_target,
                                    style=style, limit=cfg.bdd_train_size),
        "bdd_val": BDD100KDataset(bdd_root, "val", camera=camera_target,
                                  style=style, limit=cfg.bdd_val_size),
        "bdd_test": BDD100KDataset(bdd_root, "test", camera=camera_target,
                                   style=style, limit=cfg.bdd_test_size),
    }


def _loaders(cfg: Config, splits: Tuple[str, ...] = ("kitti_train", "kitti_val")):
    datasets = build_datasets(cfg)
    return {name: make_loader(datasets[name], batch_size=cfg.training.batch_size,
                              shuffle=(name.endswith("train"))) for name in splits}


# ---------------------------------------------------------------------------
# Stage 1: transfer learning
# ---------------------------------------------------------------------------
def cmd_train(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = Config.load(args.config)
    cfg.training.epochs = args.epochs or cfg.training.epochs
    cfg.training.batch_size = args.batch_size or cfg.training.batch_size
    if args.pretrained == "false":
        cfg.training.freeze_backbone = False
    device = get_device(args.device)
    seed_everything(cfg.seed)
    datasets = build_datasets(cfg)
    train_loader = make_loader(datasets["kitti_train"], batch_size=cfg.training.batch_size,
                               shuffle=True)
    val_loader = make_loader(datasets["kitti_val"], batch_size=cfg.training.batch_size,
                             shuffle=False)
    probe_loader = make_loader(Subset(datasets["kitti_train"], range(min(24, len(datasets["kitti_train"])))),
                               batch_size=4, shuffle=False)

    started = time.perf_counter()
    model, spec = load_pretrained_detector("yolov8m", num_classes=cfg.optical.num_classes,
                                           pretrained=args.pretrained == "true",
                                           min_size=cfg.training.min_size,
                                           max_size=cfg.training.max_size)
    model = ObjectDetectionModel(model, spec, device=str(device)).to(device)
    if args.train_scope == "predictor":
        model.train_predictor_only()
    elif args.freeze_backbone:
        model.unfreeze_head()
    else:
        model.unfreeze_all()
    out_path = _resolve(args.out)

    result = train_transfer(model, train_loader, val_loader, cfg.training,
                            cfg.training.epochs, device, checkpoint_path=out_path,
                            probe_loader=probe_loader, val_eval_images=args.val_eval_images)
    final = evaluate(model, val_loader, device)
    payload = _base_metadata(cfg, "baseline_transfer")
    payload.update({
        "dataset": args.dataset,
        "pretrained": args.pretrained == "true",
        "frozen_backbone": bool(args.freeze_backbone),
        "params": model.parameter_counts(),
        "training": result,
        "val": final,
        "checkpoint": str(out_path),
        "runtime_seconds": round(time.perf_counter() - started, 2),
    })
    metrics_path = _resolve(args.metrics_out or "results/baseline_metrics.json")
    save_json(payload, metrics_path)
    save_model(model, out_path, metadata={"stage": "transfer", "val_map50": final["map50"]})

    if args.scratch_comparison:
        print("[scratch] training comparison model from random initialisation")
        scratch_epochs = args.scratch_epochs
        scratch_limit = min(args.scratch_images, len(datasets["kitti_train"]))
        scratch_train = make_loader(Subset(datasets["kitti_train"], range(scratch_limit)),
                                    batch_size=cfg.training.batch_size, shuffle=True)
        scratch_cfg = Config.load(args.config).training
        scratch_cfg.epochs = scratch_epochs
        scratch_cfg.learning_rate = max(cfg.training.learning_rate, 1e-3)
        scratch_cfg.freeze_backbone = False
        scratch_model, scratch_spec = load_pretrained_detector(
            "yolov8m", num_classes=cfg.optical.num_classes, pretrained=False,
            min_size=cfg.training.min_size, max_size=cfg.training.max_size)
        scratch_model = ObjectDetectionModel(scratch_model, scratch_spec, device=str(device)).to(device)
        scratch_model.unfreeze_all()
        scratch_path = _resolve("results/scratch_checkpoint.pt")
        scratch_result = train_transfer(scratch_model, scratch_train, val_loader, scratch_cfg,
                                        scratch_epochs, device, checkpoint_path=scratch_path,
                                        probe_loader=None, val_eval_images=args.val_eval_images)
        scratch_final = evaluate(scratch_model, val_loader, device)
        payload["scratch"] = {
            "training": scratch_result,
            "val": scratch_final,
            "epochs": scratch_epochs,
            "train_images": scratch_limit,
            "params": scratch_model.parameter_counts(),
            "checkpoint": str(scratch_path),
        }
        payload["comparison"] = {
            "transfer_map50": final["map50"],
            "scratch_map50": scratch_final["map50"],
            "absolute_gain": round(final["map50"] - scratch_final["map50"], 4),
            "relative_gain_pct": (round(100.0 * (final["map50"] - scratch_final["map50"]) /
                                        scratch_final["map50"], 2)
                                  if scratch_final["map50"] > 1e-6 else None),
            "transfer_epochs": cfg.training.epochs,
            "scratch_epochs": scratch_epochs,
        }
        save_json(payload, metrics_path)
        scratch_model.close()
    print(json.dumps({"val_map50": final["map50"], "val_precision": final["precision"],
                      "val_recall": final["recall"],
                      "per_class": {k: v["ap50"] for k, v in final["per_class"].items()}},
                     indent=2))
    model.close()
    return payload


# ---------------------------------------------------------------------------
# Stage 2: domain adaptation
# ---------------------------------------------------------------------------
def cmd_adapt(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = Config.load(args.config)
    cfg.adaptation.epochs = args.epochs or cfg.adaptation.epochs
    cfg.adaptation.self_training_epochs = args.self_training_epochs \
        if args.self_training_epochs is not None else cfg.adaptation.self_training_epochs
    device = get_device(args.device)
    seed_everything(cfg.seed)
    datasets = build_datasets(cfg, camera_target=cfg.target_camera)
    max_images = min(args.max_images, len(datasets["kitti_train"]), len(datasets["bdd_train"]))
    source_sub = Subset(datasets["kitti_train"], range(max_images))
    target_sub = Subset(datasets["bdd_train"], range(max_images))
    source_loader = make_loader(source_sub, batch_size=cfg.adaptation.batch_size, shuffle=True)
    target_loader = make_loader(target_sub, batch_size=cfg.adaptation.batch_size, shuffle=True)
    target_val_loader = make_loader(datasets["bdd_val"], batch_size=4, shuffle=False)
    source_val_loader = make_loader(datasets["kitti_val"], batch_size=4, shuffle=False)

    started = time.perf_counter()
    checkpoint = _resolve(args.checkpoint)
    model = load_model(checkpoint, device=str(device))
    before_target = evaluate(model, target_val_loader, device)
    before_source = evaluate(model, source_val_loader, device)
    baseline_state = copy.deepcopy(model.model.state_dict())
    best_state = copy.deepcopy(baseline_state)
    best_target = before_target["map50"]
    selected_phase = "baseline"

    # ---- adversarial phase --------------------------------------------
    batch = next(iter(target_loader))
    images = [img.to(device).float() / 255.0 for img in batch[0]]
    model.clear_features()
    with torch.no_grad():
        model(images, None)
    feats = model.roi_features
    discriminator = build_discriminator(feats, cfg.adaptation).to(device)
    model.clear_features()

    adv = adapt_model(model, source_loader, target_loader, discriminator,
                      cfg.adaptation, cfg.adaptation.epochs, device,
                      target_eval_loader=None)
    after_adv = evaluate(model, target_val_loader, device)
    if after_adv["map50"] > best_target:
        best_target = after_adv["map50"]
        best_state = copy.deepcopy(model.model.state_dict())
        selected_phase = "adversarial"
    else:
        model.model.load_state_dict(baseline_state)

    # ---- self-training phase ------------------------------------------
    st: Dict[str, Any] = {"history": [], "self_training_seconds": 0.0}
    after_st = evaluate(model, target_val_loader, device)
    if cfg.adaptation.self_training_epochs > 0:
        st_cfg = copy.deepcopy(cfg.adaptation)
        st_cfg.learning_rate = max(cfg.adaptation.learning_rate, 1e-3)
        st = self_train(model, target_loader, st_cfg,
                        cfg.adaptation.self_training_epochs, device,
                        target_eval_loader=None, scope="predictor")
        after_st = evaluate(model, target_val_loader, device)
        if after_st["map50"] > best_target:
            best_target = after_st["map50"]
            best_state = copy.deepcopy(model.model.state_dict())
            selected_phase = "self_training"
        else:
            model.model.load_state_dict(best_state)

    after_target = evaluate(model, target_val_loader, device)
    after_source = evaluate(model, source_val_loader, device)
    out_path = _resolve(args.out)
    save_model(model, out_path, metadata={"stage": "adapted", "target_map50": after_target["map50"],
                                          "selected_phase": selected_phase})

    source_map = after_source["map50"]
    payload = _base_metadata(cfg, "domain_adaptation")
    payload.update({
        "source_domain": "kitti (sedan camera)",
        "target_domain": f"bdd100k ({cfg.target_camera}-camera style transfer)",
        "checkpoint_in": str(checkpoint),
        "checkpoint": str(out_path),
        "target_before": before_target,
        "target_after": after_target,
        "target_after_adversarial": after_adv,
        "target_after_self_training": after_st,
        "source_before": before_source,
        "source_after": after_source,
        "selected_phase": selected_phase,
        "protocol": "no-harm checkpoint selection on the labelled target validation "
                    "split (candidate phases) with the target test split held out",
        "improvement": {
            "absolute": round(after_target["map50"] - before_target["map50"], 4),
            "percent": round(100.0 * (after_target["map50"] - before_target["map50"]) /
                             max(before_target["map50"], 1e-9), 2),
            "target_over_source_pct": round(100.0 * after_target["map50"] /
                                            max(source_map, 1e-9), 2),
            "source_map_after": source_map,
        },
        "adversarial": adv,
        "self_training": st,
        "adaptation_images": max_images,
        "runtime_seconds": round(time.perf_counter() - started, 2),
    })
    save_json(payload, _resolve(args.metrics_out or "results/adaptation_metrics.json"))
    model.close()
    print(json.dumps({
        "target_before_map50": before_target["map50"],
        "target_after_map50": after_target["map50"],
        "source_map50": source_map,
        "selected_phase": selected_phase,
        "improvement": payload["improvement"],
    }, indent=2))
    return payload


# ---------------------------------------------------------------------------
# Stage 3: distillation
# ---------------------------------------------------------------------------
def cmd_distill(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = Config.load(args.config)
    cfg.distillation.epochs = args.epochs or cfg.distillation.epochs
    device = get_device(args.device)
    seed_everything(cfg.seed)
    datasets = build_datasets(cfg, camera_target=cfg.target_camera)
    train_loader = make_loader(datasets["bdd_train"], batch_size=cfg.distillation.batch_size,
                               shuffle=True)
    val_loader = make_loader(datasets["bdd_val"], batch_size=4, shuffle=False)

    started = time.perf_counter()
    teacher_path = _resolve(args.teacher)
    teacher = load_model(teacher_path, device=str(device))
    teacher.model.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher_map = evaluate(teacher, val_loader, device)

    # ---- student without KD (control) ----------------------------------
    seed_everything(cfg.seed)
    student_plain = create_student_model(args.student, cfg.optical.num_classes,
                                         pretrained=True,
                                         min_size=cfg.training.min_size,
                                         max_size=cfg.training.max_size).to(device)
    plain_result = train_with_distillation(
        student_plain, None, train_loader, cfg.distillation, cfg.distillation.epochs,
        device, use_kd=False, val_loader=val_loader,
        checkpoint_path=_resolve("results/student_no_kd.pt"))
    plain_map = evaluate(student_plain, val_loader, device)

    # ---- student with KD ------------------------------------------------
    seed_everything(cfg.seed)
    student = create_student_model(args.student, cfg.optical.num_classes,
                                   pretrained=True, min_size=cfg.training.min_size,
                                   max_size=cfg.training.max_size).to(device)
    out_path = _resolve(args.out)
    kd_result = train_with_distillation(
        student, teacher, train_loader, cfg.distillation, cfg.distillation.epochs,
        device, use_kd=True, val_loader=val_loader, checkpoint_path=out_path)
    kd_map = evaluate(student, val_loader, device)

    teacher_params = sum(p.numel() for p in teacher.model.parameters())
    student_params = sum(p.numel() for p in student.model.parameters())
    payload = _base_metadata(cfg, "distillation")
    payload.update({
        "teacher_checkpoint": str(teacher_path),
        "student_checkpoint": str(out_path),
        "student_model": args.student,
        "temperature": cfg.distillation.temperature,
        "alpha": cfg.distillation.alpha,
        "teacher_map": teacher_map,
        "student_no_kd_map": plain_map,
        "student_kd_map": kd_map,
        "params": {
            "teacher": int(teacher_params),
            "student": int(student_params),
            "reduction_pct": round(100.0 * (1 - student_params / teacher_params), 2),
        },
        "retention": {
            "kd_over_teacher_pct": round(100.0 * kd_map["map50"] / max(teacher_map["map50"], 1e-9), 2),
            "no_kd_over_teacher_pct": round(100.0 * plain_map["map50"] / max(teacher_map["map50"], 1e-9), 2),
            "kd_minus_no_kd": round(kd_map["map50"] - plain_map["map50"], 4),
        },
        "training": kd_result,
        "no_kd_training": plain_result,
        "runtime_seconds": round(time.perf_counter() - started, 2),
    })
    save_json(payload, _resolve(args.metrics_out or "results/distillation_metrics.json"))
    print(json.dumps({"teacher_map50": teacher_map["map50"],
                      "student_no_kd_map50": plain_map["map50"],
                      "student_kd_map50": kd_map["map50"],
                      "retention_pct": payload["retention"]["kd_over_teacher_pct"]}, indent=2))
    teacher.close()
    student.close()
    student_plain.close()
    return payload


# ---------------------------------------------------------------------------
# Stage 4: benchmark
# ---------------------------------------------------------------------------
def cmd_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = Config.load(args.config)
    device = get_device(args.device)
    result = full_benchmark(_resolve(args.teacher), _resolve(args.student),
                            num_runs=args.runs, device=str(device),
                            input_size=tuple(cfg.benchmark.input_size),
                            out=_resolve(args.out),
                            warmup=cfg.benchmark.warmup)
    save_json({"stage": "benchmark", "command": _command(),
               "git_commit": _git_commit(), **result}, _resolve(args.out))
    print(json.dumps({"teacher_ms": result["teacher"]["latency"]["mean_ms"],
                      "student_ms": result["student"]["latency"]["mean_ms"],
                      "speedup": result["speedup"],
                      "param_reduction_pct": result["param_reduction_pct"]}, indent=2))
    return result


# ---------------------------------------------------------------------------
# Stage 5: differentiable robustness optimisation
# ---------------------------------------------------------------------------
def cmd_optimize(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = Config.load(args.config)
    device = get_device(args.device)
    seed_everything(cfg.seed)
    datasets = build_datasets(cfg, camera_target=cfg.target_camera)
    if args.dataset in {"val_bdd100k", "bdd100k_val"}:
        dataset = datasets["bdd_val"]
    else:
        dataset = datasets["bdd_test"]
    eval_loader = make_loader(dataset, batch_size=4, shuffle=False)
    model = load_model(_resolve(args.checkpoint), device=str(device))

    started = time.perf_counter()
    grad_cfg = RobustnessConfig(seed=cfg.seed)
    images, targets = [], []
    for i in range(min(4, len(dataset))):
        image, boxes, labels, _ = dataset[i]
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
        tensor = torch.nn.functional.interpolate(tensor[None], size=(192, 192),
                                                 mode="bilinear", align_corners=False)[0]
        h, w = image.shape[:2]
        boxes = torch.from_numpy(boxes).float() * torch.tensor([192 / w, 192 / h, 192 / w, 192 / h])
        images.append(tensor)
        targets.append({"boxes": boxes, "labels": torch.from_numpy(labels).long()})
    from .optimization import RobustPreprocessor
    pre = RobustPreprocessor(grad_cfg).to(device)
    gradient_check = verify_gradients(pre, model, torch.stack(images).to(device), targets,
                                      CONDITIONS[:3], seed=cfg.seed)

    opt_cfg = RobustnessConfig(
        iterations=args.iterations,
        seed=cfg.seed,
        images_per_step=4,
        conditions_per_step=3,
        optim_resolution=192,
    )
    optimization = optimize_robustness(model, dataset, device, opt_cfg,
                                       eval_loader=eval_loader)
    payload = _base_metadata(cfg, "robustness_optimization")
    payload.update({
        "checkpoint": str(_resolve(args.checkpoint)),
        "dataset": args.dataset,
        "gradient_check": gradient_check,
        "optimized_parameters": optimization["optimized_parameters"],
        "initial_parameters": optimization["initial_parameters"],
        "history": optimization["history"],
        "evaluation": optimization["evaluation"],
        "optimization_seconds": optimization["optimization_seconds"],
        "runtime_seconds": round(time.perf_counter() - started, 2),
    })
    save_json(payload, _resolve(args.out))
    print(json.dumps({"gradient_check_passed": gradient_check["passed"],
                      "max_relative_error": gradient_check["max_relative_error"],
                      "optimized_parameters": payload["optimized_parameters"],
                      "improvement": payload["evaluation"]["improvement"]}, indent=2))
    model.close()
    return payload


# ---------------------------------------------------------------------------
# Stage 6: Monte Carlo yield
# ---------------------------------------------------------------------------
def cmd_yield(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = Config.load(args.config)
    device = get_device(args.device)
    datasets = build_datasets(cfg, camera_target=cfg.target_camera)
    dataset = datasets["bdd_test"] if args.dataset != "val_bdd100k" else datasets["bdd_val"]
    model = load_model(_resolve(args.checkpoint), device=str(device))
    started = time.perf_counter()
    result = run_monte_carlo_yield(model, dataset, device, num_draws=args.draws,
                                   seed=cfg.seed, iou_threshold=0.5,
                                   score_threshold=args.score_threshold)
    payload = _base_metadata(cfg, "monte_carlo_yield")
    payload.update({"checkpoint": str(_resolve(args.checkpoint)), **result,
                    "runtime_seconds": round(time.perf_counter() - started, 2)})
    save_json(payload, _resolve(args.out))
    print(json.dumps({"yield": result["yield"], "ci95": result["yield_ci95"],
                      "mean_recall": result["mean_recall"],
                      "map50_mean": result["map50_mean"]}, indent=2))
    model.close()
    return payload


# ---------------------------------------------------------------------------
# Stage 7: plots + summary
# ---------------------------------------------------------------------------
def cmd_plot(args: argparse.Namespace) -> Dict[str, Any]:
    from . import plots as P
    cfg = Config.load(args.config)
    results_dir = _resolve(cfg.results_dir)
    figures_dir = _resolve(cfg.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated: Dict[str, str] = {}
    baseline = P._maybe(results_dir, "baseline_metrics.json")
    if baseline:
        p = P.plot_transfer_curves(baseline, figures_dir / "transfer_learning_curves.png")
        if p: generated["transfer_learning_curves"] = p
    adaptation = P._maybe(results_dir, "adaptation_metrics.json")
    if adaptation:
        p = P.plot_adaptation(adaptation, figures_dir / "adaptation_per_class.png")
        if p: generated["adaptation_per_class"] = p
    distillation = P._maybe(results_dir, "distillation_metrics.json")
    if distillation:
        p = P.plot_distillation(distillation, figures_dir / "distillation_comparison.png")
        if p: generated["distillation_comparison"] = p
    benchmark = P._maybe(results_dir, "benchmark_report.json")
    if benchmark:
        p = P.plot_latency(benchmark, figures_dir / "latency_throughput.png")
        if p: generated["latency_throughput"] = p
    optimization = P._maybe(results_dir, "optimization_report.json")
    if optimization:
        p = P.plot_optimization(optimization, figures_dir / "optimization_convergence.png")
        if p: generated["optimization_convergence"] = p
        p = P.plot_robustness_heatmap(optimization, figures_dir / "robustness_heatmap.png")
        if p: generated["robustness_heatmap"] = p
    mc = P._maybe(results_dir, "monte_carlo_yield.json")
    if mc:
        p = P.plot_monte_carlo(mc, figures_dir / "monte_carlo_yield.png")
        if p: generated["monte_carlo_yield"] = p
    datasets = build_datasets(cfg, camera_target=cfg.target_camera) \
        if (_resolve(cfg.dataset_root) / "kitti").exists() else None
    p = P.plot_domain_shift(datasets["kitti_val"] if datasets else None,
                            datasets["bdd_val"] if datasets else None,
                            figures_dir / "domain_shift_examples.png")
    if p: generated["domain_shift_examples"] = p
    student_ckpt = _resolve("results/student_checkpoint.pt")
    if student_ckpt.exists():
        model = load_model(student_ckpt, device=str(get_device(args.device)))
        p = P.plot_detection_examples(model, {"kitti val": datasets["kitti_val"],
                                              "bdd100k truck": datasets["bdd_val"]},
                                      figures_dir / "detection_examples.png",
                                      device=str(get_device(args.device)))
        if p: generated["detection_examples"] = p
        model.close()
    save_json({"stage": "plots", "command": _command(), "figures": generated},
              results_dir / "plots_manifest.json")
    print(json.dumps(generated, indent=2))
    return generated


def cmd_summary(args: argparse.Namespace) -> Dict[str, Any]:
    from .summary import build_summary
    cfg = Config.load(args.config)
    summary = build_summary(_resolve(cfg.results_dir), _resolve(cfg.figures_dir))
    save_json(summary, _resolve(cfg.results_dir) / "summary.json")
    print(json.dumps({k: summary[k] for k in summary if k in
                      ("baseline_map50", "adapted_target_map50", "student_kd_map50",
                       "speedup", "monte_carlo_yield")}, indent=2))
    return summary


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cv_obstacle.cli",
        description="Cross-vehicle obstacle detection pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--threads", type=int, default=0, help="torch CPU threads (0=all)")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--config", default="configs/training_base.yaml")
        p.add_argument("--device", default="cpu")
        return p

    p = common(sub.add_parser("train", help="fine-tune a pretrained detector on KITTI"))
    p.add_argument("--dataset", default="kitti")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--out", default="results/baseline_checkpoint.pt")
    p.add_argument("--pretrained", default="true", choices=["true", "false"])
    p.add_argument("--train-scope", default="head", choices=["head", "predictor", "all"],
                   help="which parameters to fine-tune (head=default, predictor=classifier only)")
    p.add_argument("--freeze-backbone", action="store_true", default=True)
    p.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false")
    p.add_argument("--scratch-comparison", action="store_true", default=True)
    p.add_argument("--no-scratch-comparison", dest="scratch_comparison",
                   action="store_false")
    p.add_argument("--scratch-epochs", type=int, default=2)
    p.add_argument("--scratch-images", type=int, default=96)
    p.add_argument("--val-eval-images", type=int, default=48)
    p.add_argument("--metrics-out", default=None, help="override the metrics JSON path")
    p.set_defaults(func=cmd_train)

    p = common(sub.add_parser("adapt", help="adversarial domain adaptation"))
    p.add_argument("--source-dataset", default="kitti")
    p.add_argument("--target-dataset", default="bdd100k_augmented")
    p.add_argument("--checkpoint", default="results/baseline_checkpoint.pt")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--self-training-epochs", type=int, default=None)
    p.add_argument("--max-images", type=int, default=64)
    p.add_argument("--metrics-out", default=None)
    p.add_argument("--out", default="results/adapted_checkpoint.pt")
    p.set_defaults(func=cmd_adapt)

    p = common(sub.add_parser("distill", help="distill teacher into lightweight student"))
    p.add_argument("--teacher", default="results/adapted_checkpoint.pt")
    p.add_argument("--student", default="yolov8n")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--metrics-out", default=None)
    p.add_argument("--out", default="results/student_checkpoint.pt")
    p.set_defaults(func=cmd_distill)

    p = common(sub.add_parser("benchmark", help="latency / throughput benchmark"))
    p.add_argument("--teacher", default="results/adapted_checkpoint.pt")
    p.add_argument("--student", default="results/student_checkpoint.pt")
    p.add_argument("--runs", type=int, default=200)
    p.add_argument("--out", default="results/benchmark_report.json")
    p.set_defaults(func=cmd_benchmark)

    p = common(sub.add_parser("optimize", help="differentiable robustness optimisation"))
    p.add_argument("--checkpoint", default="results/student_checkpoint.pt")
    p.add_argument("--dataset", default="test_bdd100k")
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--out", default="results/optimization_report.json")
    p.set_defaults(func=cmd_optimize)

    p = common(sub.add_parser("yield", help="Monte Carlo detection yield"))
    p.add_argument("--checkpoint", default="results/student_checkpoint.pt")
    p.add_argument("--dataset", default="test_bdd100k")
    p.add_argument("--draws", type=int, default=200)
    p.add_argument("--score-threshold", type=float, default=0.5)
    p.add_argument("--out", default="results/monte_carlo_yield.json")
    p.set_defaults(func=cmd_yield)

    p = common(sub.add_parser("plot", help="generate all figures"))
    p.set_defaults(func=cmd_plot)

    p = common(sub.add_parser("summary", help="aggregate results/summary.json"))
    p.set_defaults(func=cmd_summary)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_threads(args.threads or None)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
