import json

import numpy as np
import pytest

from pdm.cli import main
from pdm.data import simulate
from pdm.train import threshold_for_recall, train


def test_threshold_for_recall_prefers_highest_precision_at_target():
    y = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    s = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.7, 0.65, 0.4])
    t = threshold_for_recall(y, s, target_recall=0.75)  # 3 of 4 positives
    assert t == pytest.approx(0.7)
    assert threshold_for_recall(y, s, target_recall=1.0) == pytest.approx(0.4)


@pytest.mark.slow
def test_train_end_to_end(tmp_path):
    df = simulate(4000, seed=3)
    result = train(df, out_dir=tmp_path, seed=0, folds=3, make_plots=True)

    assert result.best_model in {"logistic_regression", "random_forest", "hist_gradient_boosting"}
    assert set(result.cv) == {"logistic_regression", "random_forest", "hist_gradient_boosting"}
    # The failure rules are deterministic functions of the sensors, so a good model should find them.
    assert result.test["roc_auc"] > 0.9
    assert result.test["recall"] >= 0.8
    assert result.ablation["raw_plus_engineered"] >= result.ablation["raw_features"] - 0.05
    assert sum(result.importances.values()) > 0

    for name in (
        "model.joblib",
        "metrics.json",
        "report.md",
        "pr_curve.png",
        "confusion_matrix.png",
        "feature_importance.png",
    ):
        assert (tmp_path / name).exists(), name
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["best_model"] == result.best_model
    assert "## Feature ablation" in (tmp_path / "report.md").read_text()


@pytest.mark.slow
def test_cli_simulate_train_predict(tmp_path, capsys):
    csv = tmp_path / "sim.csv"
    assert main(["simulate", "--n", "5000", "--seed", "5", "--out", str(csv)]) == 0
    out = tmp_path / "reports"
    assert main(["train", "--data", str(csv), "--out", str(out), "--no-plots"]) == 0
    record = {  # overstrain (230 min * 70 Nm > 11 000) and power (> 9 000 W): an unambiguous failure
        "Type": "L",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1250,
        "Torque [Nm]": 70.0,
        "Tool wear [min]": 230,
    }
    assert main(["predict", "--model", str(out / "model.joblib"), "--json", json.dumps(record)]) == 0
    pred = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert pred["flag"] is True and pred["failure_probability"] > 0.5
    assert main(["predict", "--model", str(out / "model.joblib"), "--json", json.dumps({"Type": "L"})]) == 2
