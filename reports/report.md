# Predictive maintenance — model report

10,000 rows, 198 failures (2.0%). Failure modes: TWF 57, HDF 54, PWF 23, OSF 61, RNF 8.

## Model comparison (5-fold stratified CV on the training split)

| Model | PR-AUC | ± | ROC-AUC | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| random_forest **(selected)** | 0.763 | 0.051 | 0.964 | 0.550 | 0.681 |
| hist_gradient_boosting | 0.723 | 0.040 | 0.942 | 0.624 | 0.676 |
| logistic_regression | 0.286 | 0.038 | 0.862 | 0.732 | 0.117 |

## Feature ablation

PR-AUC with raw sensor columns only: **0.643**; with power, ΔT and strain added: **0.763** (+0.120).

## Held-out evaluation

2,500 rows, 49 failures. Threshold 0.072 chosen for ≥ 90 % recall.

| PR-AUC | ROC-AUC | Precision | Recall | F1 |
| ---: | ---: | ---: | ---: | ---: |
| 0.756 | 0.965 | 0.281 | 0.918 | 0.431 |

Confusion matrix (rows = actual, columns = predicted; no failure / failure):

```
[2336, 115]
[4, 45]
```

## Permutation importance (Δ PR-AUC when the column is shuffled)

| Feature | Importance |
| --- | ---: |
| temp_delta_K | 0.413 |
| Rotational speed [rpm] | 0.259 |
| Tool wear [min] | 0.127 |
| strain_minNm | 0.107 |
| Torque [Nm] | 0.063 |
| Type | 0.054 |
| Air temperature [K] | 0.033 |
| power_W | 0.017 |
| Process temperature [K] | 0.009 |

## Artifacts

- `model`: reports/model.joblib
- `pr_curve`: reports/pr_curve.png
- `confusion`: reports/confusion_matrix.png
- `importance`: reports/feature_importance.png
