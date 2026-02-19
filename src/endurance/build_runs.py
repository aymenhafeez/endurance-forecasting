import json
import pandas as pd
import numpy as np
from .db import connect
from .config import CFG


def zone_from_avg_hr(avg_hr: float | None) -> int | None:
    if avg_hr is None or np.isnan(avg_hr):
        return None

    hrmax = CFG.hr_max

    if avg_hr < 0.7 * hrmax:
        return 1
    if avg_hr < 0.8 * hrmax:
        return 2
    if avg_hr < 0.87 * hrmax:
        return 3

    return 4


def trimp_proxy(moving_time_min: float, avg_hr: float | None) -> float:
    if avg_hr is None or np.isnan(avg_hr):
        return 0.0

    intensity = float(avg_hr) / float(CFG.hr_max)

    return float(moving_time_min) * (intensity**2)


def build_runs():
    with connect() as con:
        raw = pd.read_sql_query(
            "SELECT activity_id, raw_json FROM activities_raw WHERE type='Run'", con
        )

    if raw.empty:
        print("No run activities found in activities_raw")
        return

    rows = []
    for _, r in raw.iterrows():
        a = json.loads(r["raw_json"])
        dist = float(a.get("distance", 0.0))
        moving = int(a.get("moving_time", 0))
        elapsed = (
            int(a.get("elapsed_time", 0)) if a.get("elapsed_time") is not None else None
        )
        elev = (
            float(a.get("total_elevation_gain", 0.0))
            if a.get("total_elevation_gain") is not None
            else 0.0
        )
        avg_hr = a.get("average_heartrate")
        max_hr = a.get("max_heartrate")
        avg_speed = a.get("average_speed")

        start_local = a.get("start_date_local", "")

        # normalise date time format
        dt = pd.to_datetime(start_local, utc=True, errors="coerce")
        date = dt.date().isoformat() if pd.notna(dt) else ""
        time = dt.time().isoformat() if pd.notna(dt) else ""

        # pace in seconds / km
        pace = (moving / (dist / 1000.0)) if dist > 0 else np.nan
        moving_min = moving / 60.0

        z = zone_from_avg_hr(avg_hr if avg_hr is not None else np.nan)
        tr = trimp_proxy(moving_min, avg_hr if avg_hr is not None else np.nan)

        rows.append(
            {
                "activity_id": int(a["id"]),
                "start_date_local": date,
                "start_time_local": time,
                "distance_m": dist,
                "moving_time_s": moving,
                "elapsed_time_s": elapsed,
                "elev_gain_m": elev,
                "avg_hr": float(avg_hr) if avg_hr is not None else np.nan,
                "max_hr": float(max_hr) if max_hr is not None else np.nan,
                "avg_speed:mps": float(avg_speed) if avg_speed is not None else np.nan,
                "pace_s_per_km": float(pace) if pd.notna(pace) else np.nan,
                "trimp": float(tr),
                "effort_zone": int(z) if z is not None else None,
            }
        )

    df = pd.DataFrame(rows)
    df = df[df["start_date_local"] != ""].copy()

    with connect() as con:
        # replace table entries
        # not clean, maybe rebuild runs table from scratch
        df.to_sql("runs", con, if_exists="append", index=False)

    print(f"Inserted {len(df)} runs")


if __name__ == "__main__":
    build_runs()
