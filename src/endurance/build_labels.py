import pandas as pd
from .db import connect


def build_weekly_labels():
    with connect() as con:
        weekly = pd.read_sql_query(
            "SELECT week_start, distance_km FROM weekly_features ORDER BY week_start",
            con,
        )

    if weekly.empty:
        print("No weekly_features found")
        return

    weekly["week_start_dt"] = pd.to_datetime(weekly["week_start"])
    weekly["y_next_week_distance_km"] = weekly["distance_km"].shift(-1)

    labels = weekly.dropna(subset=["y_next_week_distance_km"])[
        ["week_start", "y_next_week_distance_km"]
    ].copy()

    with connect() as con:
        con.execute("DELETE FROM lebels_weekly;")
        labels.to_sql("labels_weekly", con, if_exists="append", index=False)

    print(f"Wrote {len(labels)} weekly labels")


if __name__ == "__main__":
    build_weekly_labels()
