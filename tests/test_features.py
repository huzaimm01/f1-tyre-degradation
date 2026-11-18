import pandas as pd
import pytest

from tyremodel import features


def make_laps(n=12):
    rows = []
    for i in range(n):
        rows.append({
            "Driver": "VER" if i < n // 2 else "NOR",
            "LapNumber": i + 1,
            "LapTime": pd.Timedelta(seconds=80 + 0.05 * i),
            "Stint": 1 if i % 6 < 3 else 2,
            "Compound": "MEDIUM" if i % 2 else "HARD",
            "TyreLife": float(i % 6 + 1),
            "PitInTime": pd.NaT,
            "PitOutTime": pd.NaT,
            "TrackStatus": "1",
            "IsAccurate": True,
        })
    laps = pd.DataFrame(rows)
    # FastF1 stores pit times as timedeltas, so match that dtype.
    for col in ("PitInTime", "PitOutTime"):
        laps[col] = pd.Series(pd.NaT, index=laps.index, dtype="timedelta64[ns]")
    return laps


def test_clean_laps_drops_pit_laps_and_non_green_laps():
    laps = make_laps()
    laps.loc[0, "PitInTime"] = pd.Timedelta(seconds=1)
    laps.loc[1, "PitOutTime"] = pd.Timedelta(seconds=1)
    laps.loc[2, "TrackStatus"] = "12"
    laps.loc[3, "IsAccurate"] = False
    laps.loc[4, "LapTime"] = pd.NaT
    laps.loc[5, "Compound"] = "WET"
    cleaned = features.clean_laps(laps)
    assert len(cleaned) == len(laps) - 6
    assert (cleaned["Compound"].isin(features.COMPOUNDS)).all()


def test_clean_laps_adds_seconds_and_stint_id():
    cleaned = features.clean_laps(make_laps())
    assert cleaned.loc[0, "lap_time_s"] == pytest.approx(80.0)
    assert cleaned.loc[0, "stint_id"] == "VER-1"


def test_fuel_correction_removes_the_trend_from_a_pure_fuel_effect():
    # Lap times that drop exactly 0.03 s per lap should become constant.
    laps = make_laps(10)
    laps["LapTime"] = [pd.Timedelta(seconds=90 - 0.03 * n) for n in range(10)]
    cleaned = features.clean_laps(laps)
    corrected = features.fuel_correct(cleaned, total_laps=10)
    assert corrected["corrected_s"].std() < 1e-6


def test_make_xy_shapes_and_columns():
    cleaned = features.fuel_correct(features.clean_laps(make_laps()), total_laps=12)
    X, y = features.make_xy(cleaned)
    assert list(X.columns) == ["tyre_life", "soft", "medium", "hard"]
    assert len(X) == len(y)
    assert (X[["soft", "medium", "hard"]].sum(axis=1) == 1).all()

    X_raw, _ = features.make_xy(cleaned, target="raw")
    assert "lap_number" in X_raw.columns


def test_split_keeps_each_stint_on_one_side():
    cleaned = features.fuel_correct(features.clean_laps(make_laps(24)), total_laps=24)
    train, test = features.split_by_stint(cleaned, test_size=0.4, seed=1)
    assert set(train["stint_id"]).isdisjoint(set(test["stint_id"]))
    assert len(train) + len(test) == len(cleaned)
