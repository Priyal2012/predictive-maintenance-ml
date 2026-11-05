"""Engineered features grounded in the failure physics.

The raw sensor columns are kept; three derived quantities are added because each
corresponds directly to one failure mechanism:

* ``power_W``        τ·ω — power failures happen outside 3500–9000 W
* ``temp_delta_K``   process − air — heat-dissipation failures need ΔT < 8.6 K
* ``strain_minNm``   tool wear · torque — overstrain failures exceed a per-type limit

A tree model can approximate these from the raw columns, but giving it the right
axes makes the decision boundaries simpler, the model smaller and the explanations
honest.
"""

from __future__ import annotations

import pandas as pd

from pdm.data import CATEGORICAL, NUMERIC, power_watts

ENGINEERED = ["power_W", "temp_delta_K", "strain_minNm"]


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["power_W"] = power_watts(out["Torque [Nm]"], out["Rotational speed [rpm]"]).round(1)
    out["temp_delta_K"] = (out["Process temperature [K]"] - out["Air temperature [K]"]).round(2)
    out["strain_minNm"] = (out["Tool wear [min]"] * out["Torque [Nm]"]).round(1)
    return out


def feature_columns(*, engineered: bool = True) -> tuple[list[str], list[str]]:
    """(numeric, categorical) column names for a feature set."""
    numeric = list(NUMERIC) + (ENGINEERED if engineered else [])
    return numeric, list(CATEGORICAL)
