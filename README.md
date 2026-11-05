# predictive-maintenance-ml

**Will this machine fail? A small, honest ML pipeline on the AI4I 2020 schema, built
by someone who has stood next to the equipment.**

The AI4I 2020 dataset (UCI, Matzka 2020) models a milling process: air and process
temperature, spindle speed, torque and tool wear, with five failure modes — tool wear,
heat dissipation, power, overstrain and random. Failures are ~2–3 % of rows, so the
interesting problems are class imbalance, choosing the right metric, and giving the
model features that correspond to the physics instead of hoping a forest discovers them.

```
$ python -m pdm simulate --n 10000 --out data/ai4i_synthetic.csv
$ python -m pdm train --data data/ai4i_synthetic.csv --out reports
best model: random_forest
  random_forest            CV PR-AUC 0.763 ± 0.051   ROC-AUC 0.964
  hist_gradient_boosting   CV PR-AUC 0.723 ± 0.040   ROC-AUC 0.942
  logistic_regression      CV PR-AUC 0.286 ± 0.038   ROC-AUC 0.862
ablation: raw 0.643 → raw+engineered 0.763
held-out: PR-AUC 0.756  ROC-AUC 0.965  precision 0.281  recall 0.918  (threshold 0.072)
```

On the held-out 2,500 rows the selected model catches **45 of 49 failures** at a
threshold tuned for ≥ 90 % recall, with 115 false alarms — the trade-off you want on a
maintenance line, where a missed failure costs far more than an unnecessary inspection.
The full report with plots is in [`reports/`](reports/report.md).

## What's in it

**`pdm/data.py` — load or simulate.** `load()` reads the real UCI csv (`pdm download`
fetches it). `simulate()` re-implements the dataset's *documented* generating rules —
heat-dissipation failure when ΔT < 8.6 K and speed < 1380 rpm, power failure outside
3500–9000 W, overstrain when wear · torque exceeds 11/12/13 kN·m·min by product type,
tool-wear failure at a 200–240 min replacement point, 0.1 % random failures — so the
pipeline is reproducible offline and every failure flag is verifiable from the stored
columns (the tests check exactly that). The committed `data/ai4i_synthetic.csv` is
simulated; it is *not* the UCI file.

**`pdm/features.py` — physics-based features.** Three derived columns, one per
mechanism: mechanical power `τ·ω` (W), temperature delta (K), and strain
`wear · torque` (min·Nm). The ablation above shows they lift PR-AUC by **+0.12** over
the raw sensors with the same model.

**`pdm/train.py` — compare, ablate, evaluate.** Three estimators with class weighting
(logistic regression, random forest, histogram gradient boosting) under 5-fold
stratified CV; selection on **average precision**, not accuracy; threshold chosen for a
target recall; held-out confusion matrix, PR curve, and **permutation importance** on
the exact columns the pipeline reads. Everything lands in `reports/` as
`metrics.json`, `report.md`, three PNGs and the serialised pipeline.

**`pdm/cli.py`** — `simulate` / `download` / `train` / `predict`:

```bash
python -m pdm predict --model reports/model.joblib --json \
  '{"Type":"L","Air temperature [K]":300,"Process temperature [K]":310,"Rotational speed [rpm]":1250,"Torque [Nm]":70,"Tool wear [min]":230}'
{"failure_probability": 0.6504, "threshold": 0.0716, "flag": true}
```

## Install & run

```bash
git clone https://github.com/Priyal2012/predictive-maintenance-ml
cd predictive-maintenance-ml
pip install -e ".[dev]"
pytest                       # 10 tests incl. a full training run (~45 s)
python -m pdm train          # uses the committed simulated csv
python -m pdm download && python -m pdm train --data data/ai4i2020.csv   # real data
```

## Things worth knowing

- **Why PR-AUC?** With 2 % positives, ROC-AUC looks flattering (0.96) while precision
  at useful recall is modest (0.28). Average precision tracks the number that matters.
- **Why permutation importance on `X_test[model columns]`?** Permuting the whole frame
  would also shuffle the id and label columns the pipeline ignores and mis-align the
  importances — a bug I hit and now test for.
- **What the model learned:** ΔT and speed dominate (the heat-dissipation rule), then
  tool wear and strain (overstrain); power matters less here because power failures are
  the rarest mode in the simulated data.
- **Limits:** the simulator reproduces the rules and roughly the marginals of the
  original (failure rate ~2 % vs 3.4 %); it is a stand-in for the real file, not a
  replacement. Train on the UCI csv for publishable numbers.

MIT licensed.
