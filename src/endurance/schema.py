from .db import connect

DDL = [
    """
CREATE TABLE IF NOT EXISTS activities_raw (
  activity_id INTEGER PRIMARY KEY,
  start_date_utc TEXT NOT NULL,
  type TEXT NOT NULL,
  raw_json TEXT NOT NULL,
  ingested_at TEXT NOT NULL
);
""",
    """
CREATE TABLE IF NOT EXISTS runs (
  activity_id INTEGER PRIMARY KEY,
  start_date_local TEXT NOT NULL,
  start_time_local TEXT,
  distance_m REAL NOT NULL,
  moving_time_s INTEGER NOT NULL,
  elapsed_time_s INTEGER,
  elev_gain_m REAL,
  avg_hr REAL,
  max_hr REAL,
  avg_speed_mps REAL,
  pace_s_per_km REAL NOT NULL,
  trimp REAL,
  effort_zone INTEGER
);
""",
    """
CREATE INDEX IF NOT EXISTS idx_runs_date ON runs(start_date_local);
""",
    """
CREATE TABLE IF NOT EXISTS daily_features (
  date TEXT PRIMARY KEY,
  run_count INTEGER NOT NULL,
  distance_km REAL NOT NULL,
  time_min REAL NOT NULL,
  elev_m REAL NOT NULL,
  avg_hr_mean REAL,
  trimp REAL NOT NULL,
  easy_distance_km REAL NOT NULL,
  hard_distance_km REAL NOT NULL,
  easy_time_min REAL NOT NULL,
  hard_time_min REAL NOT NULL
);
""",
    """
CREATE TABLE IF NOT EXISTS weekly_features (
  week_start TEXT PRIMARY KEY,
  distance_km REAL NOT NULL,
  time_min REAL NOT NULL,
  elev_m REAL NOT NULL,
  trimp REAL NOT NULL,
  easy_distance_km REAL NOT NULL,
  hard_distance_km REAL NOT NULL,
  hard_sessions INTEGER NOT NULL,
  dist_7d REAL,
  dist_28d REAL,
  trimp_7d REAL,
  trimp_28d REAL,
  acwr REAL,
  monotony_7d REAL,
  strain_7d REAL,
  easy_pct REAL
);
""",
    """
CREATE TABLE IF NOT EXISTS labels_weekly (
  week_start TEXT PRIMARY KEY,
  y_next_week_distance_km REAL NOT NULL
);
""",
]


def init_db():
    with connect() as con:
        for stmt in DDL:
            con.execute(stmt)


if __name__ == "__main__":
    init_db()
    print("DB initialised.")
