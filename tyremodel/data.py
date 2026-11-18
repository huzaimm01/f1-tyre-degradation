"""Download race laps with tif1.

tif1 reads timing data that was generated with FastF1 and published as
static files on a public CDN, so no F1TV account or sign-in is needed.
Its session/laps API mirrors FastF1's, including the lap columns used here.
"""
from pathlib import Path

import tif1

LAP_COLUMNS = [
    "Driver", "LapNumber", "LapTime", "Stint", "Compound", "TyreLife",
    "PitInTime", "PitOutTime", "TrackStatus", "IsAccurate",
]


def load_laps(year, event, cache_dir="cache"):
    """Return every lap of a race as a DataFrame, using a local cache."""
    # tif1 loads lazily and caches in ~/.cache/tif1 by default; cache_dir is
    # kept for compatibility with the old FastF1 loader and is not used.
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    session = tif1.get_session(year, event, "R")
    laps = session.laps[LAP_COLUMNS].copy()
    # tif1 returns Driver/Compound/TrackStatus as categoricals; FastF1
    # returned plain strings, which the cleaning code expects.
    for col in ("Driver", "Compound", "TrackStatus"):
        laps[col] = laps[col].astype(str)
    return laps
