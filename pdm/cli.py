"""Command-line interface.

python -m pdm simulate --n 10000 --out data/ai4i_synthetic.csv
python -m pdm download --out data/ai4i2020.csv          # real UCI data (needs network)
python -m pdm train --data data/ai4i_synthetic.csv --out reports
python -m pdm predict --model reports/model.joblib --json '{"Type":"L","Air temperature [K]":298.1,...}'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from pdm.data import CATEGORICAL, NUMERIC, describe, download, load, simulate
from pdm.features import add_engineered
from pdm.train import train


def cmd_simulate(args) -> int:
    df = simulate(args.n, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df):,} rows to {args.out}: {json.dumps(describe(df))}")
    return 0


def cmd_download(args) -> int:
    try:
        path = download(args.out)
    except Exception as exc:  # noqa: BLE001 - report any network/zip problem plainly
        print(f"download failed ({exc}); use `pdm simulate` for offline data", file=sys.stderr)
        return 1
    print(f"saved {path}")
    return 0


def cmd_train(args) -> int:
    df = load(args.data)
    result = train(df, out_dir=args.out, seed=args.seed, target_recall=args.target_recall, make_plots=not args.no_plots)
    print(f"best model: {result.best_model}")
    for name, m in sorted(result.cv.items(), key=lambda kv: -kv[1]["pr_auc"]):
        print(f"  {name:<24} CV PR-AUC {m['pr_auc']:.3f} ± {m['pr_auc_std']:.3f}   ROC-AUC {m['roc_auc']:.3f}")
    a = result.ablation
    print(f"ablation: raw {a['raw_features']:.3f} → raw+engineered {a['raw_plus_engineered']:.3f}")
    t = result.test
    print(
        f"held-out: PR-AUC {t['pr_auc']:.3f}  ROC-AUC {t['roc_auc']:.3f}  "
        f"precision {t['precision']:.3f}  recall {t['recall']:.3f}  (threshold {result.threshold:.3f})"
    )
    print(f"report: {result.artifacts['report']}")
    return 0


def cmd_predict(args) -> int:
    import joblib

    bundle = joblib.load(args.model)
    record = json.loads(args.json)
    missing = [c for c in NUMERIC + CATEGORICAL if c not in record]
    if missing:
        print(f"missing fields: {missing}", file=sys.stderr)
        return 2
    row = add_engineered(pd.DataFrame([record]))
    prob = float(bundle["pipeline"].predict_proba(row)[0, 1])
    flag = prob >= bundle["threshold"]
    print(
        json.dumps(
            {"failure_probability": round(prob, 4), "threshold": round(bundle["threshold"], 4), "flag": bool(flag)}
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pdm", description="Predictive maintenance on AI4I-style sensor data.")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("simulate", help="generate data from the documented failure rules")
    s.add_argument("--n", type=int, default=10_000)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out", default="data/ai4i_synthetic.csv")
    s.set_defaults(fn=cmd_simulate)

    d = sub.add_parser("download", help="fetch the real UCI AI4I 2020 csv")
    d.add_argument("--out", default="data/ai4i2020.csv")
    d.set_defaults(fn=cmd_download)

    t = sub.add_parser("train", help="compare models, run the ablation, evaluate, write a report")
    t.add_argument("--data", default="data/ai4i_synthetic.csv")
    t.add_argument("--out", default="reports")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--target-recall", type=float, default=0.90)
    t.add_argument("--no-plots", action="store_true")
    t.set_defaults(fn=cmd_train)

    pr = sub.add_parser("predict", help="score one sensor reading with a trained model")
    pr.add_argument("--model", default="reports/model.joblib")
    pr.add_argument("--json", required=True, help="one record as JSON with the raw sensor fields")
    pr.set_defaults(fn=cmd_predict)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)
