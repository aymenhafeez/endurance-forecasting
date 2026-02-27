import numpy as np
import pandas as pd

from .db import connect


def week_start_monday(date_series: pd.Series) -> pd.Series:
    d = pd.to_datetime(date_series)
    return (d - pd.to_timedelta(d.dt.weekday, unit="D")).dt.date.astype(str)


def build_daily_and_weekly():
    with connect() as con:
        runs = pd.read_sql_query("SELECT * FROM runs", con)

    if runs.empty:
        print("No runs found. Build runs first.")
        return

    runs["date"] = pd.to_datetime(runs["start_date_local"])
    runs["distance_km"] = runs["distance_m"] / 1000.0
    runs["time_min"] = runs["moving_time_s"] / 60.0
    runs["elev_m"] = runs["elev_gain_m"].fillna(0.0)

    # easy/hard split
    runs["is_hard"] = runs["effort_zone"].fillna(0).astype(int) >= 3
    runs["easy_distance_km"] = np.where(runs["is_hard"], 0.0, runs["distance_km"])
    runs["hard_distance_km"] = np.where(runs["is_hard"], runs["distance_km"], 0.0)
    runs["easy_time_min"] = np.where(runs["is_hard"], 0.0, runs["time_min"])
    runs["hard_time_min"] = np.where(runs["is_hard"], runs["time_min"], 0.0)

    daily = (
        runs.groupby(runs["date"].dt.date)
        .agg(
            run_count=("activity_id", "count"),
            distance_km=("distance_km", "sum"),
            time_min=("time_min", "sum"),
            elev_m=("elev_m", "sum"),
            avg_hr_mean=("avg_hr", "mean"),
            trimp=("trimp", "sum"),
            easy_distance_km=("easy_distance_km", "sum"),
            hard_distance_km=("hard_distance_km", "sum"),
            easy_time_min=("easy_time_min", "sum"),
            hard_time_min=("hard_time_min", "sum"),
        )
        .reset_index()
        .rename(columns={"date": "date"})
    )
    daily["date"] = daily["date"].astype(str)

    # build weekly totals from daily data
    daily_dt = pd.to_datetime(daily["date"])
    daily["week_start"] = (
        daily_dt - pd.to_timedelta(daily_dt.dt.weekday, unit="D")
    ).dt.date.astype(str)

    weekly = (
        daily.groupby("week_start")
        .agg(
            distance_km=("distance_km", "sum"),
            time_min=("time_min", "sum"),
            elev_m=("elev_m", "sum"),
            trimp=("trimp", "sum"),
            easy_distance_km=("easy_distance_km", "sum"),
            hard_distance_km=("hard_distance_km", "sum"),
            hard_sessions=("hard_distance_km", lambda x: int((x > 0).sum())),
        )
        .reset_index()
    )

    # better to get rolling windows rather than just weekly
    daily_sorted = daily.sort_values("date").copy()
    daily_sorted["date_dt"] = pd.to_datetime(daily_sorted["date"])
    daily_sorted = daily_sorted.set_index("date_dt")

    # make sure index is continuous for rolling stats
    full_idx = pd.date_range(
        daily_sorted.index.min(), daily_sorted.index.max(), freq="D"
    )
    daily_full = daily_sorted.reindex(full_idx).fillna(
        {
            "run_count": 0,
            "distance_km": 0.0,
            "time_min": 0.0,
            "elev_m": 0.0,
            "trimp": 0.0,
            "easy_distance_km": 0.0,
            "hard_distance_km": 0.0,
            "easy_time_min": 0.0,
            "hard_time_min": 0.0,
        }
    )

    daily_full["trimp_7d"] = daily_full["trimp"].rolling(7).sum()
    daily_full["trimp_28d"] = daily_full["trimp"].rolling(28).sum()
    daily_full["dist_7d"] = daily_full["distance_km"].rolling(7).sum()
    daily_full["dist_28d"] = daily_full["distance_km"].rolling(28).sum()

    trimp_mean = daily_full["trimp"].rolling(7).mean()
    trimp_std = daily_full["trimp"].rolling(7).std().replace(0, np.nan)
    daily_full["monotony_7d"] = (trimp_mean / trimp_std).fillna(0.0)
    daily_full["strain_7d"] = daily_full["monotony_7d"] * daily_full["trimp_7d"]

    daily_full["acwr"] = daily_full["trimp_7d"] / (
        daily_full["trimp_28d"] / 4.0
    ).replace(0, np.nan)
    daily_full["acwr"] = (
        daily_full["acwr"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )

    # map rolling features to each week_start using the last day of that week
    daily_full = daily_full.reset_index().rename(columns={"index": "date_dt"})
    daily_full["week_start"] = (
        daily_full["date_dt"]
        - pd.to_timedelta(daily_full["date_dt"].dt.weekday, unit="D")
    ).dt.date.astype(str)
    week_roll = daily_full.groupby("week_start").tail(1)[
        [
            "week_start",
            "dist_7d",
            "dist_28d",
            "trimp_7d",
            "trimp_28d",
            "acwr",
            "monotony_7d",
            "strain_7d",
        ]
    ]

    weekly = weekly.merge(week_roll, on="week_start", how="left")
    weekly["easy_pct"] = np.where(
        weekly["distance_km"] > 0,
        weekly["easy_distance_km"] / weekly["distance_km"],
        0.0,
    )

    daily_to_store = daily.drop(columns=["week_start"], errors="ignore")

    # maybe not necessary but explicitly define tthe columns to add
    daily_cols = [
        "date",
        "run_count",
        "distance_km",
        "time_min",
        "elev_m",
        "avg_hr_mean",
        "trimp",
        "easy_distance_km",
        "hard_distance_km",
        "easy_time_min",
        "hard_time_min",
    ]
    weekly_cols = [
        "week_start",
        "distance_km",
        "time_min",
        "elev_m",
        "trimp",
        "easy_distance_km",
        "hard_distance_km",
        "hard_sessions",
        "dist_7d",
        "dist_28d",
        "trimp_7d",
        "trimp_28d",
        "acwr",
        "monotony_7d",
        "strain_7d",
        "easy_pct",
    ]

    daily_to_store = daily_to_store[daily_cols]
    weekly = weekly[weekly_cols]

    with connect() as con:
        con.execute("DELETE FROM daily_features;")
        con.execute("DELETE FROM weekly_features;")
        daily_to_store.to_sql("daily_features", con, if_exists="append", index=False)
        weekly.to_sql("weekly_features", con, if_exists="append", index=False)

    print(f"Wrote {len(daily)} daily rows and {len(weekly)} weekly rows.")


if __name__ == "__main__":
    build_daily_and_weekly()
