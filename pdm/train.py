"""Model comparison, ablation and held-out evaluation for machine-failure prediction.

Failures are rare (a few percent), so accuracy is meaningless here: a model that
always says "no failure" scores 97 %. We select on **average precision (PR-AUC)**
and also report ROC-AUC, recall and F1 at a threshold chosen for a target recall,
because a missed failure costs far more than a false alarm on a maintenance line.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pdm.data import TARGET, describe
from pdm.features import add_engineered, feature_columns


def make_models(seed: int = 0) -> dict[str, object]:
    """Candidate estimators. Class weights compensate for the ~3 % positive rate."""
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=seed
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31, class_weight="balanced", random_state=seed
        ),
    }


def make_pipeline(estimator, *, engineered: bool = True) -> Pipeline:
    numeric, categorical = feature_columns(engineered=engineered)
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    return Pipeline([("pre", pre), ("clf", estimator)])


def threshold_for_recall(y_true: np.ndarray, scores: np.ndarray, target_recall: float = 0.90) -> float:
    """Smallest threshold whose recall is still ≥ ``target_recall`` (maximises precision at that recall)."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # precision_recall_curve returns one more (p, r) point than thresholds; drop the last.
    ok = np.where(recall[:-1] >= target_recall)[0]
    if len(ok) == 0:
        return 0.5
    return float(thresholds[ok[-1]])


@dataclass
class TrainResult:
    best_model: str
    cv: dict[str, dict[str, float]]
    ablation: dict[str, float]
    test: dict[str, float]
    confusion: list[list[int]]
    threshold: float
    importances: dict[str, float]
    data_summary: dict
    engineered: bool = True
    artifacts: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__


def train(
    df: pd.DataFrame,
    *,
    out_dir: Path | str = "reports",
    seed: int = 0,
    test_size: float = 0.25,
    folds: int = 5,
    target_recall: float = 0.90,
    make_plots: bool = True,
) -> TrainResult:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = add_engineered(df)
    y = data[TARGET].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=test_size, stratify=y, random_state=seed)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scoring = {"pr_auc": "average_precision", "roc_auc": "roc_auc", "recall": "recall", "f1": "f1"}

    # ---- 1. compare models with the full feature set --------------------------------
    cv_results: dict[str, dict[str, float]] = {}
    for name, est in make_models(seed).items():
        pipe = make_pipeline(est, engineered=True)
        cv = cross_validate(pipe, X_train, y_train, cv=skf, scoring=scoring, n_jobs=1)
        cv_results[name] = {k: round(float(np.mean(cv[f"test_{k}"])), 4) for k in scoring}
        cv_results[name]["pr_auc_std"] = round(float(np.std(cv["test_pr_auc"])), 4)
    best_name = max(cv_results, key=lambda n: cv_results[n]["pr_auc"])

    # ---- 2. ablation: does the physics-based feature engineering help? ------------------
    ablation: dict[str, float] = {}
    for engineered in (False, True):
        pipe = make_pipeline(make_models(seed)[best_name], engineered=engineered)
        cv = cross_validate(pipe, X_train, y_train, cv=skf, scoring="average_precision", n_jobs=1)
        ablation["raw_features" if not engineered else "raw_plus_engineered"] = round(
            float(np.mean(cv["test_score"])), 4
        )

    # ---- 3. fit the winner, evaluate on the held-out split ---------------------------------
    model = make_pipeline(make_models(seed)[best_name], engineered=True).fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]
    threshold = threshold_for_recall(y_test, scores, target_recall)
    pred = (scores >= threshold).astype(int)
    test = {
        "pr_auc": round(float(average_precision_score(y_test, scores)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, scores)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred)), 4),
        "f1": round(float(f1_score(y_test, pred)), 4),
        "positives": int(y_test.sum()),
        "n": int(len(y_test)),
    }
    cm = confusion_matrix(y_test, pred).tolist()

    numeric, categorical = feature_columns(engineered=True)
    model_inputs = X_test[numeric + categorical]  # only the columns the pipeline actually reads
    perm = permutation_importance(
        model, model_inputs, y_test, scoring="average_precision", n_repeats=5, random_state=seed
    )
    importances = {
        col: round(float(v), 4)
        for col, v in sorted(zip(model_inputs.columns, perm.importances_mean, strict=True), key=lambda kv: -kv[1])
    }

    result = TrainResult(
        best_model=best_name,
        cv=cv_results,
        ablation=ablation,
        test=test,
        confusion=cm,
        threshold=round(threshold, 4),
        importances=importances,
        data_summary=describe(df),
    )

    # ---- 4. artifacts -------------------------------------------------------------------------------
    model_path = out_dir / "model.joblib"
    joblib.dump({"pipeline": model, "threshold": threshold, "engineered": True}, model_path)
    result.artifacts["model"] = str(model_path)
    if make_plots:
        result.artifacts.update(_plots(out_dir, y_test, scores, cm, importances))
    (out_dir / "metrics.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(render_report(result), encoding="utf-8")
    result.artifacts["report"] = str(out_dir / "report.md")
    return result


def _plots(out_dir: Path, y_test, scores, cm, importances) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = {}

    precision, recall, _ = precision_recall_curve(y_test, scores)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision–recall (AP = {average_precision_score(y_test, scores):.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    paths["pr_curve"] = str(out_dir / "pr_curve.png")
    fig.savefig(paths["pr_curve"], dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(np.array(cm)):
        ax.text(j, i, str(v), ha="center", va="center", color="black" if v < max(map(max, cm)) / 2 else "white")
    ax.set_xticks([0, 1], ["no failure", "failure"])
    ax.set_yticks([0, 1], ["no failure", "failure"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (held-out)")
    fig.tight_layout()
    paths["confusion"] = str(out_dir / "confusion_matrix.png")
    fig.savefig(paths["confusion"], dpi=130)
    plt.close(fig)

    names = list(importances)[::-1]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.barh(names, [importances[n] for n in names])
    ax.set_xlabel("Permutation importance (Δ average precision)")
    ax.set_title("What the model relies on")
    fig.tight_layout()
    paths["importance"] = str(out_dir / "feature_importance.png")
    fig.savefig(paths["importance"], dpi=130)
    plt.close(fig)
    return paths


def render_report(r: TrainResult) -> str:
    d = r.data_summary
    lines = [
        "# Predictive maintenance — model report",
        "",
        f"{d['rows']:,} rows, {d['failures']} failures ({d['failure_rate']:.1%}). "
        f"Failure modes: " + ", ".join(f"{m} {n}" for m, n in d["by_mode"].items()) + ".",
        "",
        "## Model comparison (5-fold stratified CV on the training split)",
        "",
        "| Model | PR-AUC | ± | ROC-AUC | Recall | F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, m in sorted(r.cv.items(), key=lambda kv: -kv[1]["pr_auc"]):
        flag = " **(selected)**" if name == r.best_model else ""
        cells = [f"{m[k]:.3f}" for k in ("pr_auc", "pr_auc_std", "roc_auc", "recall", "f1")]
        lines.append(f"| {name}{flag} | " + " | ".join(cells) + " |")
    a = r.ablation
    delta = a["raw_plus_engineered"] - a["raw_features"]
    lines += [
        "",
        "## Feature ablation",
        "",
        f"PR-AUC with raw sensor columns only: **{a['raw_features']:.3f}**; "
        f"with power, ΔT and strain added: **{a['raw_plus_engineered']:.3f}** ({delta:+.3f}).",
        "",
        "## Held-out evaluation",
        "",
        f"{r.test['n']:,} rows, {r.test['positives']} failures. Threshold {r.threshold:.3f} chosen for ≥ 90 % recall.",
        "",
        "| PR-AUC | ROC-AUC | Precision | Recall | F1 |",
        "| ---: | ---: | ---: | ---: | ---: |",
        "| " + " | ".join(f"{r.test[k]:.3f}" for k in ("pr_auc", "roc_auc", "precision", "recall", "f1")) + " |",
        "",
        "Confusion matrix (rows = actual, columns = predicted; no failure / failure):",
        "",
        "```",
        f"{r.confusion[0]}",
        f"{r.confusion[1]}",
        "```",
        "",
        "## Permutation importance (Δ PR-AUC when the column is shuffled)",
        "",
        "| Feature | Importance |",
        "| --- | ---: |",
    ]
    lines += [f"| {k} | {v:.3f} |" for k, v in r.importances.items()]
    if r.artifacts:
        lines += ["", "## Artifacts", ""]
        lines += [f"- `{k}`: {v}" for k, v in r.artifacts.items()]
    return "\n".join(lines) + "\n"
