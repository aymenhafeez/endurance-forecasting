import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .db import connect

FEATURES = [
    "distance_km",
    "time_min",
    "elev_m",
    "trimp",
    "easy_pct",
    "hard_sessions",
    "dist_7d",
    "dist_28d",
    "trimp_7d",
    "trimp_28d",
    "acwr",
    "monotony_7d",
    "strain_7d",
    "lag1_distance_km",
    "lag2_distance_km",
    "lag3_distance_km",
    "lag1_trimp",
    "lag2_trimp",
]


def train_eval():
    with connect() as con:
        X = pd.read_sql_query(
            """
            SELECT wf.*, lb.y_next_week_distance_km
            FROM weekly_features wf
            JOIN labels_weekly lb USING(week_start)
            ORDER BY wf.week_start
        """,
            con,
        )

    X = X.dropna(subset=["y_next_week_distance_km"]).copy()
    if len(X) < 20:
        print("Not enough weeks to train robustly yet.")
        return

    # use max value between splitting at the last 8 weeks vs the last 20%
    # to make sure the test set isn't too small
    split = max(len(X) - 8, int(len(X) * 0.8))
    train = X.iloc[:split].copy()
    test = X.iloc[split:].copy()

    y_train = train["y_next_week_distance_km"].values
    y_test = test["y_next_week_distance_km"].values

    # baseline predictions
    # last week distance baseline predictions
    last_week_pred = test["distance_km"].values

    # for each test week get the last 4 weeks distance
    # full series in time order
    all_dist = X["distance_km"].values
    rolling4_pred = []
    for idx in range(split, len(X)):
        start = max(0, idx - 4)
        hist = all_dist[start:idx]
        # if there's fewer than 4 weeks available use whatever exists
        rolling4_pred.append(
            float(np.mean(hist)) if len(hist) else float(all_dist[idx - 1])
        )
    rolling4_pred = np.array(rolling4_pred, dtype=float)

    def report(name, y_pred):
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        within_10 = float(np.mean(np.abs(y_test - y_pred) <= 10.0))

        return name, mae, rmse, within_10

    results = []
    results.append(report("baseline_last_week", last_week_pred))
    results.append(report("baseline_4wk_mean", rolling4_pred))

    # train and fit Ridge regression model
    X_train = train[FEATURES].fillna(0.0).values
    X_test = test[FEATURES].fillna(0.0).values

    model = Ridge(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results.append(report("ridge", y_pred))

    res_df = pd.DataFrame(
        results, columns=["model", "MAE_km", "RMSE_km", "within_10km"]
    )

    # make sure df gets printed in a readeable format
    print(res_df.to_string(index=False))


if __name__ == "__main__":
    train_eval()
