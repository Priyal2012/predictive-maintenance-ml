"""Load or simulate AI4I 2020-style predictive-maintenance data.

The real dataset (Matzka, 2020; UCI id 601) is itself synthetic, generated from a
documented set of rules. ``simulate()`` re-implements those rules so the pipeline
is reproducible offline, with the same column names and failure modes:

TWF  tool wear failure      tool wear reaches a replacement time drawn from 200–240 min
HDF  heat dissipation       (process − air) temperature < 8.6 K  and  speed < 1380 rpm
PWF  power failure          power = torque · ω  outside 3500–9000 W
OSF  overstrain failure     tool wear · torque > 11 000 (L) / 12 000 (M) / 13 000 (H) min·Nm
RNF  random failure         0.1 % of rows regardless of parameters

``Machine failure`` is 1 if any mode fired. Rows: ~3–4 % failures, like the original.

Use ``load()`` on the real CSV when you have it (``pdm download`` tries to fetch it).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"

TARGET = "Machine failure"
FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
NUMERIC = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
CATEGORICAL = ["Type"]
COLUMNS = ["UDI", "Product ID", *CATEGORICAL, *NUMERIC, TARGET, *FAILURE_MODES]

OSF_THRESHOLD = {"L": 11_000, "M": 12_000, "H": 13_000}
TYPE_SHARE = {"L": 0.5, "M": 0.3, "H": 0.2}


def power_watts(torque_nm: np.ndarray | float, rpm: np.ndarray | float) -> np.ndarray | float:
    """Mechanical power P = τ·ω with ω in rad/s."""
    return torque_nm * rpm * 2 * np.pi / 60


def _random_walk(rng: np.random.Generator, n: int, *, mean: float, std: float) -> np.ndarray:
    """A smoothed random walk re-normalised to the requested mean and standard deviation."""
    steps = rng.normal(0, 1, n).cumsum()
    smooth = pd.Series(steps).rolling(50, min_periods=1).mean().to_numpy()
    smooth = (smooth - smooth.mean()) / (smooth.std() + 1e-9)
    return mean + std * smooth


def simulate(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Generate ``n`` rows following the AI4I 2020 failure rules."""
    rng = np.random.default_rng(seed)

    types = rng.choice(list(TYPE_SHARE), size=n, p=list(TYPE_SHARE.values()))
    air = _random_walk(rng, n, mean=300.0, std=2.0)
    process = air + 10.0 + _random_walk(rng, n, mean=0.0, std=1.0)

    # Torque ~ N(40, 10) Nm, never below 3.8. Speed follows from a nominal ~6.45 kW
    # operating point (so torque and speed are negatively correlated, as in the
    # original data) plus noise, clipped to the original's observed range.
    torque = np.clip(rng.normal(40.0, 10.0, n), 3.8, None)
    nominal_speed = 6450.0 / (torque * 2 * np.pi / 60)
    speed = np.clip(nominal_speed + rng.normal(0, 120.0, n), 1168, 2886)

    # Tool wear accumulates in runs: each product type adds 2/3/5 minutes per step.
    wear_step = np.vectorize({"L": 2, "M": 3, "H": 5}.get)(types)
    replacement_at = rng.integers(200, 241, n)  # minutes at which the tool is replaced
    wear = np.zeros(n)
    current = 0.0
    for i in range(n):
        wear[i] = current
        current += wear_step[i]
        if current >= replacement_at[i]:
            current = 0.0

    # Round to the precision of the published data *before* applying the rules, so
    # the flags are exactly reproducible from the stored columns.
    air, process = np.round(air, 1), np.round(process, 1)
    speed, torque, wear = np.round(speed).astype(int), np.round(torque, 1), wear.astype(int)

    twf = (wear + wear_step >= replacement_at) & (rng.random(n) < 0.42)  # ~42% of replacements fail, as in the original
    hdf = ((process - air) < 8.6) & (speed < 1380)
    pw = power_watts(torque, speed)
    pwf = (pw < 3500) | (pw > 9000)
    osf_limit = np.vectorize(OSF_THRESHOLD.get)(types)
    osf = (wear * torque) > osf_limit
    rnf = rng.random(n) < 0.001

    df = pd.DataFrame(
        {
            "UDI": np.arange(1, n + 1),
            "Product ID": [f"{t}{14860 + i}" for i, t in enumerate(types)],
            "Type": types,
            "Air temperature [K]": air,
            "Process temperature [K]": process,
            "Rotational speed [rpm]": speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": wear,
            "TWF": twf.astype(int),
            "HDF": hdf.astype(int),
            "PWF": pwf.astype(int),
            "OSF": osf.astype(int),
            "RNF": rnf.astype(int),
        }
    )
    df[TARGET] = df[FAILURE_MODES].max(axis=1)
    return df[COLUMNS]


def load(path: Path | str) -> pd.DataFrame:
    """Load a CSV in the AI4I schema (the UCI file or one written by ``simulate``)."""
    df = pd.read_csv(path)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")
    return df[COLUMNS]


def download(dest: Path | str, *, url: str = UCI_URL, timeout: int = 60) -> Path:
    """Fetch the real UCI dataset zip and extract ``ai4i2020.csv`` to ``dest``."""
    import io
    import zipfile

    import requests

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        dest.write_bytes(zf.read(name))
    return dest


def describe(df: pd.DataFrame) -> dict:
    """Quick summary used by the report."""
    return {
        "rows": int(len(df)),
        "failure_rate": round(float(df[TARGET].mean()), 4),
        "failures": int(df[TARGET].sum()),
        "by_mode": {m: int(df[m].sum()) for m in FAILURE_MODES},
        "by_type": {t: int(c) for t, c in df["Type"].value_counts().items()},
    }
