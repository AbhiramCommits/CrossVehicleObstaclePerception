PY := .venv/bin/python
CONFIG ?= configs/pipeline_cpu.yaml

.PHONY: help venv data all train adapt distill benchmark optimize yield plots summary test smoke clean-results

help:
	@echo "make venv      create .venv and install requirements"
	@echo "make data      download the reduced real KITTI + BDD100K dataset"
	@echo "make all       run the entire pipeline (data cached) and write results/ + figures/"
	@echo "make test      run the pytest suite"
	@echo "make smoke     tiny fixture end-to-end smoke run"

venv:
	python3.11 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) scripts/download_data.py --kitti 900 --bdd 900

all: data
	$(PY) scripts/run_all.py --config $(CONFIG)

train:
	$(PY) -m cv_obstacle.cli train --config $(CONFIG) --dataset kitti --epochs 5 \
		--out results/baseline_checkpoint.pt

adapt:
	$(PY) -m cv_obstacle.cli adapt --config $(CONFIG) \
		--checkpoint results/baseline_checkpoint.pt --epochs 1 --max-images 16 \
		--out results/adapted_checkpoint.pt

distill:
	$(PY) -m cv_obstacle.cli distill --config $(CONFIG) \
		--teacher results/adapted_checkpoint.pt --student yolov8n --epochs 4 \
		--out results/student_checkpoint.pt

benchmark:
	$(PY) -m cv_obstacle.cli benchmark --config $(CONFIG) \
		--teacher results/adapted_checkpoint.pt --student results/student_checkpoint.pt \
		--runs 12 --out results/benchmark_report.json

optimize:
	$(PY) -m cv_obstacle.cli optimize --config $(CONFIG) \
		--checkpoint results/student_checkpoint.pt --iterations 30 \
		--out results/optimization_report.json

yield:
	$(PY) -m cv_obstacle.cli yield --config $(CONFIG) \
		--checkpoint results/student_checkpoint.pt --draws 80 \
		--out results/monte_carlo_yield.json

plots:
	$(PY) -m cv_obstacle.cli plot --config $(CONFIG)

summary:
	$(PY) -m cv_obstacle.cli summary --config $(CONFIG)

test:
	$(PY) -m pytest -q

smoke:
	$(PY) scripts/make_fixture.py --out datasets_fixture --count 12
	CV_DATASET_ROOT=datasets_fixture $(PY) -m cv_obstacle.cli train --config $(CONFIG) \
		--epochs 1 --no-scratch-comparison --val-eval-images 4 \
		--out results/smoke_baseline.pt --metrics-out results/smoke_baseline_metrics.json
	CV_DATASET_ROOT=datasets_fixture $(PY) -m cv_obstacle.cli distill --config $(CONFIG) \
		--teacher results/smoke_baseline.pt --epochs 1 \
		--out results/smoke_student.pt --metrics-out results/smoke_distillation_metrics.json
	CV_DATASET_ROOT=datasets_fixture $(PY) -m cv_obstacle.cli benchmark --config $(CONFIG) \
		--teacher results/smoke_baseline.pt --student results/smoke_student.pt \
		--runs 3 --out results/smoke_benchmark.json

clean-results:
	rm -f results/*.pt results/*.onnx results/smoke_*.json
