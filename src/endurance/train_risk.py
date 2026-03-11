import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import TimeSeriesSplit

from .db import connect

FEATURES = [
    "lag1_distance_km",
    "lag1_trimp",
    "dist_7d",
    "dist_28d",
    "acwr",
    "monotony_7d",
    "strain_7d",
]


def train_eval():
    with connect() as con:
        df = pd.read_sql_query(
            """
            SELECT wf.*, lb.y_risk
            FROM weekly_features wf
            JOIN labels_weekly lb USING(week_start)
            ORDER BY wf.week_start
            """,
            con,
        )

    df = df.dropna(subset=["y_risk"]).copy()

    print("Class balance (y_risk):")
    print(df["y_risk"].value_counts())

    split = max(len(df) - 8, int(len(df) * 0.8))
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()

    X_train = train[FEATURES].fillna(0.0).values
    y_train = train["y_risk"].values
    X_test = test[FEATURES].fillna(0.0).values
    y_test = test["y_risk"].values

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )

    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    if len(np.unique(y_test)) > 1:
        print("\nROC-AUC:", roc_auc_score(y_test, probs))
        print("PR-AUC:", average_precision_score(y_test, probs))
    else:
        print("\nTest set has only one class; ROC-AUC undefined.")

    print("\nConfusion matrix:\n", confusion_matrix(y_test, preds))
    print("\nClassification report:\n", classification_report(y_test, preds, digits=3))


if __name__ == "__main__":
    train_eval()
