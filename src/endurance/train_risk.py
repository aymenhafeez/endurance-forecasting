import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .db import connect

FEATURES = [
    "lag1_distance_km",
    "lag1_trimp",
    # "dist_7d",
    # "dist_28d",
    "acwr",
    # "monotony_7d",
    # "strain_7d",
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

    print("\nFinal holdout size:", len(test))

    X_train = train[FEATURES].fillna(0.0).values
    y_train = train["y_risk"].values
    X_test = test[FEATURES].fillna(0.0).values
    y_test = test["y_risk"].values

    # TimeSeries CV on training block
    tscv = TimeSeriesSplit(n_splits=5)

    cv_aucs = []
    cv_prs = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ])

        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_val)[:, 1]

        if len(np.unique(y_val)) > 1:
            cv_aucs.append(roc_auc_score(y_val, probs))
            cv_prs.append(average_precision_score(y_val, probs))

    print("\nTimeSeries CV Results (training block)")
    print("AUC mean:", np.mean(cv_aucs))
    print("AUC std :", np.std(cv_aucs))
    print("PR mean :", np.mean(cv_prs))
    print("PR std  :", np.std(cv_prs))

    # final model trained on full training block
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])

    model.fit(X_train, y_train)

    probs_test = model.predict_proba(X_test)[:, 1]
    preds_test = (probs_test >= 0.5).astype(int)

    print("\nFinal Holdout Performance")
    if len(np.unique(y_test)) > 1:
        print("ROC-AUC:", roc_auc_score(y_test, probs_test))
        print("PR-AUC :", average_precision_score(y_test, probs_test))
    else:
        print("Only one class in test; AUC undefined.")

    print("\nConfusion matrix:\n", confusion_matrix(y_test, preds_test))
    print(
        "\nClassification report:\n",
        classification_report(y_test, preds_test, digits=3),
    )


if __name__ == "__main__":
    train_eval()
