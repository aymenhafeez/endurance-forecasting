import json
from pathlib import Path

import joblib
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

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "lag1_distance_km",
    "lag1_trimp",
    "acwr",
]

THRESHOLD = 0.5


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
    final_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])

    final_model.fit(X_train, y_train)

    probs_test = final_model.predict_proba(X_test)[:, 1]
    preds_test = (probs_test >= THRESHOLD).astype(int)

    holdout_metrics = {}
    if len(np.unique(y_test)) > 1:
        holdout_metrics["roc_auc"] = float(roc_auc_score(y_test, probs_test))
        holdout_metrics["pr_auc"] = float(average_precision_score(y_test, probs_test))
        print("\nFinal holdout preformance")
        print("ROC-AUC:", holdout_metrics["roc_auc"])
        print("PR-AUC:", holdout_metrics["pr_auc"])
    else:
        holdout_metrics["roc_auc"] = None
        holdout_metrics["pr_auc"] = None
        print("\nOnly one classe in test. AUC undefined")

    conf_mat = confusion_matrix(y_test, preds_test)
    report = classification_report(y_test, preds_test, digits=3, zero_division=0)

    print("\nConfusion matrix:\n", conf_mat)
    print("\nClassification report:\n", report)

    # save artifacts
    joblib.dump(final_model, ARTIFACTS_DIR / "risk_model.joblib")

    clf = final_model.named_steps["clf"]
    coefficients = dict(zip(FEATURES, clf.coef_[0].tolist(), strict=False))

    metrics = {
        "features": FEATURES,
        "threshold": THRESHOLD,
        "cv": {
            "auc_mean": float(np.mean(cv_aucs)) if cv_aucs else None,
            "auc_std": float(np.std(cv_aucs)) if cv_aucs else None,
            "pr_mean": float(np.mean(cv_prs)) if cv_prs else None,
            "pr_std": float(np.std(cv_prs)) if cv_prs else None,
        },
        "holdout": holdout_metrics,
        "confusion_matrix": conf_mat.tolist(),
        "classification_report": report,
        "coefficients": coefficients,
        "train_size": int(len(train)),
        "test_size": int(len(test)),
    }

    with open(ARTIFACTS_DIR / "rist_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    holdout_predictions = test[["week_start", "y_risk"]].copy()
    holdout_predictions["predicted_probability"] = probs_test
    holdout_predictions["predicted_class"] = preds_test
    holdout_predictions.to_csv(ARTIFACTS_DIR / "holdout_predictions.csv", index=False)

    print(f"\nSaved artifacts to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    train_eval()
