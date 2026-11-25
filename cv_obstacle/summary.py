"""Aggregate every measured result into summary.json / summary.md."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .utils import load_json


def _get(path: Path) -> Optional[Dict[str, Any]]:
    return load_json(path) if path.exists() else None


def build_summary(results_dir: Path, figures_dir: Path) -> Dict[str, Any]:
    baseline = _get(results_dir / "baseline_metrics.json")
    adaptation = _get(results_dir / "adaptation_metrics.json")
    distillation = _get(results_dir / "distillation_metrics.json")
    benchmark = _get(results_dir / "benchmark_report.json")
    optimization = _get(results_dir / "optimization_report.json")
    mc = _get(results_dir / "monte_carlo_yield.json")
    manifest = _get(results_dir / "run_manifest.json")

    summary: Dict[str, Any] = {"stage": "summary", "available": {}}
    summary["available"] = {
        "baseline": baseline is not None,
        "adaptation": adaptation is not None,
        "distillation": distillation is not None,
        "benchmark": benchmark is not None,
        "optimization": optimization is not None,
        "monte_carlo_yield": mc is not None,
    }

    if baseline:
        summary["baseline_map50"] = baseline["val"]["map50"]
        summary["baseline_map75"] = baseline["val"]["map75"]
        summary["baseline_precision"] = baseline["val"]["precision"]
        summary["baseline_recall"] = baseline["val"]["recall"]
        summary["baseline_per_class"] = baseline["val"]["per_class"]
        summary["baseline_epochs"] = baseline["training"].get("best_epoch")
        summary["baseline_train_seconds"] = baseline["training"].get("train_seconds")
        summary["transfer_params"] = baseline.get("params")
        if "comparison" in baseline:
            summary["scratch_comparison"] = baseline["comparison"]
            summary["scratch_map50"] = baseline["scratch"]["val"]["map50"]

    if adaptation:
        summary["adapted_target_map50"] = adaptation["target_after"]["map50"]
        summary["adapted_source_map50"] = adaptation["source_after"]["map50"]
        summary["target_before_map50"] = adaptation["target_before"]["map50"]
        summary["adaptation_gain"] = adaptation["improvement"]
        summary["adaptation_target_per_class"] = adaptation["target_after"]["per_class"]
        summary["adaptation_runtime_seconds"] = adaptation.get("runtime_seconds")

    if distillation:
        summary["teacher_map50"] = distillation["teacher_map"]["map50"]
        summary["student_no_kd_map50"] = distillation["student_no_kd_map"]["map50"]
        summary["student_kd_map50"] = distillation["student_kd_map"]["map50"]
        summary["student_per_class"] = distillation["student_kd_map"]["per_class"]
        summary["kd_retention_pct"] = distillation["retention"]["kd_over_teacher_pct"]
        summary["parameter_reduction_pct"] = distillation["params"]["reduction_pct"]
        summary["distillation_runtime_seconds"] = distillation.get("runtime_seconds")

    if benchmark:
        summary["latency"] = {
            "teacher_ms": benchmark["teacher"]["latency"]["mean_ms"],
            "student_ms": benchmark["student"]["latency"]["mean_ms"],
            "student_onnx_ms": benchmark["student_onnx"]["latency"]["mean_ms"],
            "student_int8_ms": benchmark.get("student_int8", {}).get("latency", {}).get("mean_ms"),
            "teacher_p95_ms": benchmark["teacher"]["latency"]["p95_ms"],
            "student_p95_ms": benchmark["student"]["latency"]["p95_ms"],
        }
        summary["throughput"] = {
            key: benchmark["throughput"][key]["images_per_second"]
            for key in benchmark.get("throughput", {})
        }
        summary["speedup"] = benchmark["speedup"]
        summary["benchmark_param_reduction_pct"] = benchmark["param_reduction_pct"]
        summary["benchmark_runs"] = benchmark.get("num_runs")
        summary["benchmark_input_size"] = benchmark.get("input_size")
        summary["benchmark_device"] = benchmark.get("device")

    if optimization:
        summary["gradient_check"] = {
            "passed": optimization["gradient_check"]["passed"],
            "max_relative_error": optimization["gradient_check"]["max_relative_error"],
        }
        summary["optimized_parameters"] = optimization["optimized_parameters"]
        evaluation = optimization.get("evaluation", {})
        summary["robustness"] = {
            "baseline_mean_map50": evaluation.get("baseline", {}).get("mean_map50"),
            "baseline_worst_map50": evaluation.get("baseline", {}).get("worst_map50"),
            "optimized_mean_map50": evaluation.get("optimized", {}).get("mean_map50"),
            "optimized_worst_map50": evaluation.get("optimized", {}).get("worst_map50"),
            "improvement": evaluation.get("improvement"),
        }
        summary["optimization_seconds"] = optimization.get("optimization_seconds")

    if mc:
        summary["monte_carlo_yield"] = {
            "yield": mc["yield"],
            "ci95": mc["yield_ci95"],
            "num_draws": mc["num_draws"],
            "mean_recall": mc["mean_recall"],
            "map50_mean": mc["map50_mean"],
            "map50_std": mc["map50_std"],
            "by_weather": mc["stratified"]["weather"],
        }

    if manifest:
        summary["runtime"] = manifest.get("runtime", {})
        summary["commands"] = manifest.get("commands", {})
        summary["pipeline_started"] = manifest.get("started_at")
        summary["quick"] = manifest.get("quick", False)

    # per-stage runtimes always exposed
    per_stage = {}
    for name, payload in (("baseline", baseline), ("adaptation", adaptation),
                          ("distillation", distillation),
                          ("benchmark", benchmark), ("optimization", optimization),
                          ("monte_carlo_yield", mc)):
        if payload and payload.get("runtime_seconds") is not None:
            per_stage[name] = payload["runtime_seconds"]
    summary["stage_runtimes_seconds"] = per_stage

    for path in sorted(figures_dir.glob("*.png")):
        summary.setdefault("figures", []).append(str(path))
    summary["environment"] = (baseline or {}).get("environment")
    _write_markdown(summary, results_dir / "summary.md")
    return summary


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "not measured"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_markdown(summary: Dict[str, Any], path: Path) -> None:
    lines = ["# Measured results", "",
             "Every value below is read from the JSON artefacts produced by the run "
             "recorded in `results/run_manifest.json`. No value is hand-entered.", "",
             "## Reproduce in ~5 min", "",
             "```bash",
             "# one-time: fetch the real reduced dataset (~173 MB)",
             "python scripts/download_data.py --kitti 900 --bdd 900",
             "",
             "# fast smoke profile: 1-2 epochs per stage, tiny budgets",
             "python scripts/run_all.py --quick --skip-download",
             "```",
             "",
             "`--quick` keeps the identical code path but shortens every stage; the "
             "numbers above come from the full CPU profile (`make all`).", ""]
    if "baseline_map50" in summary:
        lines += ["## Detection quality", "",
                  "| Stage | mAP@0.5 | mAP@0.75 | Precision | Recall |",
                  "|---|---|---|---|---|",
                  f"| Baseline transfer (KITTI) | {_fmt(summary.get('baseline_map50'))} | "
                  f"{_fmt(summary.get('baseline_map75'))} | {_fmt(summary.get('baseline_precision'))} | "
                  f"{_fmt(summary.get('baseline_recall'))} |"]
        if "teacher_map50" in summary:
            lines.append(f"| Adapted teacher (BDD100K truck) | {_fmt(summary.get('teacher_map50'))} | "
                         f"{_fmt(None)} | {_fmt(None)} | {_fmt(None)} |")
            lines.append(f"| Distilled student (BDD100K truck) | {_fmt(summary.get('student_kd_map50'))} | "
                         f"{_fmt(None)} | {_fmt(None)} | {_fmt(None)} |")
        lines.append("")
    if "speedup" in summary:
        latency = summary["latency"]
        lines += ["## Latency and throughput", "",
                  f"Measured on {summary.get('benchmark_device')} at input "
                  f"{summary.get('benchmark_input_size')} over {summary.get('benchmark_runs')} runs.", "",
                  "| Model | Mean latency (ms) | p95 (ms) | Images/s |",
                  "|---|---|---|---|"]
        lines.append(f"| Teacher | {_fmt(latency.get('teacher_ms'))} | {_fmt(latency.get('teacher_p95_ms'))} | "
                     f"{_fmt(summary.get('throughput', {}).get('teacher'))} |")
        lines.append(f"| Student | {_fmt(latency.get('student_ms'))} | {_fmt(latency.get('student_p95_ms'))} | "
                     f"{_fmt(summary.get('throughput', {}).get('student'))} |")
        lines.append(f"| Student ONNX | {_fmt(latency.get('student_onnx_ms'))} | {_fmt(None)} | "
                     f"{_fmt(summary.get('throughput', {}).get('student_onnx'))} |")
        if latency.get("student_int8_ms"):
            lines.append(f"| Student int8 | {_fmt(latency.get('student_int8_ms'))} | {_fmt(None)} | "
                         f"{_fmt(None)} |")
        lines += ["", f"Student speedup: **{_fmt(summary['speedup'].get('student_vs_teacher'), 2)}x** "
                      f"(ONNX {_fmt(summary['speedup'].get('student_onnx_vs_teacher'), 2)}x).", ""]
    if "adapted_target_map50" in summary:
        lines += ["## Domain adaptation", "",
                  f"- Target mAP before adaptation: {_fmt(summary.get('target_before_map50'))}",
                  f"- Target mAP after adaptation: {_fmt(summary.get('adapted_target_map50'))}",
                  f"- Source mAP after adaptation: {_fmt(summary.get('adapted_source_map50'))}",
                  f"- Target / source: {_fmt(summary.get('adaptation_gain', {}).get('target_over_source_pct'), 1)} %", ""]
    if "kd_retention_pct" in summary:
        lines += ["## Knowledge distillation", "",
                  f"- Teacher mAP: {_fmt(summary.get('teacher_map50'))}",
                  f"- Student without KD: {_fmt(summary.get('student_no_kd_map50'))}",
                  f"- Student with KD: {_fmt(summary.get('student_kd_map50'))}",
                  f"- KD retention: {_fmt(summary.get('kd_retention_pct'), 1)} % of teacher",
                  f"- Parameter reduction: {_fmt(summary.get('parameter_reduction_pct'), 1)} %", ""]
    if "robustness" in summary:
        rob = summary["robustness"]
        lines += ["## Robustness optimisation", "",
                  f"- Optimised preprocessing: `{summary.get('optimized_parameters')}`",
                  f"- Baseline mean / worst mAP: {_fmt(rob.get('baseline_mean_map50'))} / "
                  f"{_fmt(rob.get('baseline_worst_map50'))}",
                  f"- Optimised mean / worst mAP: {_fmt(rob.get('optimized_mean_map50'))} / "
                  f"{_fmt(rob.get('optimized_worst_map50'))}",
                  f"- Gradient check: passed={summary.get('gradient_check', {}).get('passed')}, "
                  f"max relative error={summary.get('gradient_check', {}).get('max_relative_error'):.2e}"
                  if summary.get("gradient_check", {}).get("max_relative_error") is not None else "", ""]
    if "monte_carlo_yield" in summary:
        mc = summary["monte_carlo_yield"]
        lines += ["## Monte Carlo detection yield", "",
                  f"- Yield over {mc.get('num_draws')} randomised draws: **{_fmt(mc.get('yield'), 4)}** "
                  f"(95 % CI {mc.get('ci95')})",
                  f"- Mean obstacle recall: {_fmt(mc.get('mean_recall'))}",
                  f"- mAP@0.5 over draw batches: {_fmt(mc.get('map50_mean'))} ± {_fmt(mc.get('map50_std'))}", ""]
    if summary.get("stage_runtimes_seconds"):
        lines += ["## Runtime", "", "| Stage | Seconds |", "|---|---|"]
        for stage, seconds in summary["stage_runtimes_seconds"].items():
            lines.append(f"| {stage} | {_fmt(seconds, 1)} |")
        if summary.get("runtime", {}).get("total_seconds"):
            lines.append(f"| **total** | **{_fmt(summary['runtime']['total_seconds'], 1)}** |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")
