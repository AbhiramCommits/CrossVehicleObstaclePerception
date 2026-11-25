"""End-to-end artefact checks for a completed pipeline run."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

REQUIRED = {
    "baseline_metrics.json": ["val", "training", "comparison", "command"],
    "adaptation_metrics.json": ["target_before", "target_after", "improvement", "command"],
    "distillation_metrics.json": ["teacher_map", "student_no_kd_map", "student_kd_map",
                                  "retention", "command"],
    "benchmark_report.json": ["teacher", "student", "student_onnx", "speedup"],
    "optimization_report.json": ["gradient_check", "optimized_parameters", "evaluation",
                                 "command"],
    "monte_carlo_yield.json": ["yield", "yield_ci95", "stratified", "num_draws"],
    "summary.json": ["baseline_map50", "student_kd_map50", "latency", "runtime"],
}

FIGURES_EXPECTED = [
    "transfer_learning_curves.png",
    "adaptation_per_class.png",
    "distillation_comparison.png",
    "latency_throughput.png",
    "optimization_convergence.png",
    "robustness_heatmap.png",
    "monte_carlo_yield.png",
    "detection_examples.png",
    "domain_shift_examples.png",
]


def _have_run() -> bool:
    return (RESULTS / "summary.json").exists()


pytestmark = pytest.mark.skipif(not _have_run(), reason="pipeline has not been run yet")


@pytest.mark.parametrize("name,keys", REQUIRED.items())
def test_results_json_valid(name, keys):
    path = RESULTS / name
    assert path.exists(), f"missing {path}"
    payload = json.loads(path.read_text())
    for key in keys:
        assert key in payload, f"{name} missing {key}"


@pytest.mark.parametrize("figure", FIGURES_EXPECTED)
def test_figures_exist_and_are_nonempty(figure):
    path = FIGURES / figure
    assert path.exists(), f"missing {path}"
    assert path.stat().st_size > 1000


def test_summary_numbers_come_from_stage_files():
    summary = json.loads((RESULTS / "summary.json").read_text())
    baseline = json.loads((RESULTS / "baseline_metrics.json").read_text())
    assert summary["baseline_map50"] == baseline["val"]["map50"]
    adaptation = json.loads((RESULTS / "adaptation_metrics.json").read_text())
    assert summary["adapted_target_map50"] == adaptation["target_after"]["map50"]
    distillation = json.loads((RESULTS / "distillation_metrics.json").read_text())
    assert summary["student_kd_map50"] == distillation["student_kd_map"]["map50"]


def test_commands_are_recorded():
    manifest = json.loads((RESULTS / "run_manifest.json").read_text())
    assert manifest["commands"]
    for stage, command in manifest["commands"].items():
        if stage == "download":
            assert "download_data.py" in command, (stage, command)
        else:
            assert "cv_obstacle.cli" in command, (stage, command)
    assert manifest["runtime"]["total_seconds"] > 0
