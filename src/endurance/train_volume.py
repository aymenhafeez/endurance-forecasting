import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
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

    X = X.dropna(subset=["y_next_week_distance_km"])
    if len(X) < 20:
        print("Not enough weeks to train robustly yet.")
        return

    # 80/20 data train/test split
    split = int(len(X) * 0.8)
    train = X.iloc[:split]
    test = X.iloc[split:]

    y_train = train["y_next_week_distance_km"].values
    y_test = test["y_next_week_distance_km"].values

    # baseline predictions
    last_week_pred = test["distance_km"].values
    rolling4_pred = train["distance_km"].rolling(4).mean().iloc[-1]
    rolling4_pred = np.full_like(y_test, fill_value=float(rolling4_pred))

    def report(name, y_pred):
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred)
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
