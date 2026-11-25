# Measured results

Every value below is read from the JSON artefacts produced by the run recorded in `results/run_manifest.json`. No value is hand-entered.

## Reproduce in ~5 min

```bash
# one-time: fetch the real reduced dataset (~173 MB)
python scripts/download_data.py --kitti 900 --bdd 900

# fast smoke profile: 1-2 epochs per stage, tiny budgets
python scripts/run_all.py --quick --skip-download
```

`--quick` keeps the identical code path but shortens every stage; the numbers above come from the full CPU profile (`make all`).

## Detection quality

| Stage | mAP@0.5 | mAP@0.75 | Precision | Recall |
|---|---|---|---|---|
| Baseline transfer (KITTI) | 0.269 | 0.131 | 0.443 | 0.303 |
| Adapted teacher (BDD100K truck) | 0.104 | not measured | not measured | not measured |
| Distilled student (BDD100K truck) | 0.006 | not measured | not measured | not measured |

## Latency and throughput

Measured on cpu at input [3, 384, 384] over 12 runs.

| Model | Mean latency (ms) | p95 (ms) | Images/s |
|---|---|---|---|
| Teacher | 510.193 | 558.160 | 2.330 |
| Student | 81.367 | 104.685 | 14.020 |
| Student ONNX | 141.097 | not measured | 7.470 |
| Student int8 | 69.027 | not measured | not measured |

Student speedup: **6.27x** (ONNX 3.62x).

## Domain adaptation

- Target mAP before adaptation: 0.101
- Target mAP after adaptation: 0.104
- Source mAP after adaptation: 0.204
- Target / source: 51.1 %

## Knowledge distillation

- Teacher mAP: 0.104
- Student without KD: 0.007
- Student with KD: 0.006
- KD retention: 5.8 % of teacher
- Parameter reduction: 68.3 %

## Robustness optimisation

- Optimised preprocessing: `{'brightness_scale': 1.2065, 'contrast_scale': 1.2051, 'gamma': 1.1233, 'channel_gains': [0.9338, 1.0807, 1.0813]}`
- Baseline mean / worst mAP: 0.001 / 0.000
- Optimised mean / worst mAP: 0.001 / 0.000
- Gradient check: passed=True, max relative error=3.99e-04

## Monte Carlo detection yield

- Yield over 80 randomised draws: **0.0000** (95 % CI [0.0, 0.0458])
- Mean obstacle recall: 0.013
- mAP@0.5 over draw batches: 0.004 ± 0.004

## Runtime

| Stage | Seconds |
|---|---|
| baseline | 106.6 |
| adaptation | 174.4 |
| distillation | 143.5 |
| optimization | 36.5 |
| monte_carlo_yield | 7.3 |
| **total** | **594.4** |

