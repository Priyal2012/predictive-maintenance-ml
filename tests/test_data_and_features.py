import numpy as np
import pytest

from pdm.data import COLUMNS, FAILURE_MODES, OSF_THRESHOLD, TARGET, describe, load, power_watts, simulate
from pdm.features import ENGINEERED, add_engineered, feature_columns


@pytest.fixture(scope="module")
def df():
    return simulate(6000, seed=1)


def test_schema_and_determinism(df):
    assert list(df.columns) == COLUMNS
    assert len(df) == 6000
    again = simulate(6000, seed=1)
    assert df.equals(again)
    assert not df.equals(simulate(6000, seed=2))


def test_target_is_union_of_modes(df):
    assert (df[TARGET] == df[FAILURE_MODES].max(axis=1)).all()
    rate = df[TARGET].mean()
    assert 0.005 < rate < 0.08, f"failure rate {rate:.3f} is implausible"


def test_failure_rules_hold(df):
    air, proc = df["Air temperature [K]"], df["Process temperature [K]"]
    speed, torque, wear = df["Rotational speed [rpm]"], df["Torque [Nm]"], df["Tool wear [min]"]

    hdf_rule = ((proc - air) < 8.6) & (speed < 1380)
    assert (df["HDF"] == hdf_rule.astype(int)).all()

    power = power_watts(torque, speed)
    pwf_rule = (power < 3500) | (power > 9000)
    assert (df["PWF"] == pwf_rule.astype(int)).all()

    limit = df["Type"].map(OSF_THRESHOLD)
    osf_rule = (wear * torque) > limit
    assert (df["OSF"] == osf_rule.astype(int)).all()

    assert df.loc[df["TWF"] == 1, "Tool wear [min]"].min() >= 195  # fails only near replacement time
    assert df["RNF"].mean() < 0.005


def test_marginals_resemble_original(df):
    assert abs(df["Air temperature [K]"].mean() - 300) < 1
    assert abs((df["Process temperature [K]"] - df["Air temperature [K]"]).mean() - 10) < 0.5
    assert abs(df["Torque [Nm]"].mean() - 40) < 1.5
    assert df["Rotational speed [rpm]"].between(1168, 2886).all()
    assert df["Tool wear [min]"].between(0, 240).all()
    assert df["Torque [Nm]"].corr(df["Rotational speed [rpm]"]) < -0.5
    share = df["Type"].value_counts(normalize=True)
    assert share["L"] > share["M"] > share["H"]


def test_engineered_features(df):
    out = add_engineered(df.head(5))
    assert ENGINEERED[0] in out and len(out.columns) == len(COLUMNS) + 3
    row = out.iloc[0]
    expected_power = row["Torque [Nm]"] * row["Rotational speed [rpm]"] * 2 * np.pi / 60
    assert row["power_W"] == pytest.approx(expected_power, abs=0.1)
    assert row["temp_delta_K"] == pytest.approx(row["Process temperature [K]"] - row["Air temperature [K]"], abs=0.01)
    assert row["strain_minNm"] == pytest.approx(row["Tool wear [min]"] * row["Torque [Nm]"], abs=0.1)
    assert df.shape[1] == len(COLUMNS), "input must not be modified"
    numeric, categorical = feature_columns(engineered=False)
    assert "power_W" not in numeric and categorical == ["Type"]


def test_load_round_trip_and_validation(tmp_path, df):
    path = tmp_path / "x.csv"
    df.to_csv(path, index=False)
    back = load(path)
    assert list(back.columns) == COLUMNS and len(back) == len(df)
    df.drop(columns=["Torque [Nm]"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Torque"):
        load(path)


def test_describe(df):
    d = describe(df)
    assert d["rows"] == 6000 and set(d["by_mode"]) == set(FAILURE_MODES)
    assert d["failures"] == df[TARGET].sum()
