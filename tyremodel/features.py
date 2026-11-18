"""Lap cleaning, fuel correction, features and train/test splitting."""
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]

# A car gets quicker as fuel burns off. Roughly 0.03 s per lap is a commonly
# quoted figure; it varies by circuit, so treat it as an assumption.
FUEL_SECONDS_PER_LAP = 0.03


def clean_laps(laps):
    """Keep green-flag racing laps on dry tyres with a usable lap time."""
    df = laps.copy()
    df = df[df["PitInTime"].isna() & df["PitOutTime"].isna()]
    df = df[df["TrackStatus"].astype(str) == "1"]
    df = df[df["IsAccurate"].fillna(False).astype(bool)]
    df = df.dropna(subset=["LapTime", "Compound", "TyreLife", "Stint"])
    df = df[df["Compound"].isin(COMPOUNDS)].copy()
    df["lap_time_s"] = df["LapTime"].dt.total_seconds()
    df["stint_id"] = df["Driver"] + "-" + df["Stint"].astype(int).astype(str)
    return df.reset_index(drop=True)


def fuel_correct(df, total_laps, seconds_per_lap=FUEL_SECONDS_PER_LAP):
    """Add `corrected_s`: lap time with the fuel-burn speed-up removed.

    The car is lighter on later laps, so we subtract the time gained per lap of
    fuel still to burn. The result is only meaningful up to a constant, which is
    fine because we compare laps within one race.
    """
    out = df.copy()
    out["corrected_s"] = out["lap_time_s"] - seconds_per_lap * (total_laps - out["LapNumber"])
    return out


def make_xy(df, target="corrected"):
    """Build the feature matrix and target vector.

    With the fuel-corrected target only tyre age and compound are used. With the
    raw target the lap number is added so the model can pick up fuel burn itself.
    """
    X = pd.DataFrame({"tyre_life": df["TyreLife"].astype(float)})
    for compound in COMPOUNDS:
        X[compound.lower()] = (df["Compound"] == compound).astype(float)
    if target == "raw":
        X["lap_number"] = df["LapNumber"].astype(float)
        y = df["lap_time_s"]
    else:
        y = df["corrected_s"]
    return X, y


def split_by_stint(df, test_size=0.25, seed=0):
    """Split so that all laps of a stint land on the same side.

    Laps within a stint are strongly correlated, so a random row split would
    leak information from the test set into training.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=df["stint_id"]))
    return df.iloc[train_idx], df.iloc[test_idx]
