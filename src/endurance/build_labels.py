import pandas as pd
import numpy as np
from .db import connect

ACWR_THR = 1.5
SPIKE_THR = 0.30
MONO_THR = 2.0


def build_weekly_labels():
    with connect() as con:
        wf = pd.read_sql_query(
            """
            SELECT week_start, distance_km, acwr, monotony_7d, strain_7d
            FROM weekly_features
            ORDER BY week_start
            """,
            con,
        )

    wf = wf.sort_values("week_start").reset_index(drop=True)

    # label uses future next week label
    wf["next_distance_km"] = wf["distance_km"].shift(-1)
    wf["next_acwr"] = wf["acwr"].shift(-1)
    wf["next_monotony"] = wf["monotony_7d"].shift(-1)
    wf["next_strain"] = wf["strain_7d"].shift(-1)

    # Sub-labels
    wf["y_acwr_high"] = (wf["next_acwr"] > ACWR_THR).astype(int)

    # define spike = (next - this) / this, guard div0
    denom = wf["distance_km"].replace(0, np.nan)
    wf["next_spike"] = (wf["next_distance_km"] - wf["distance_km"]) / denom
    wf["y_spike_high"] = (wf["next_spike"] > SPIKE_THR).fillna(0).astype(int)

    # monotony + strain -> choose strain threshold from data distribution
    # start with p90
    strain_thr = float(wf["next_strain"].quantile(0.90))
    wf["y_mono_strain_high"] = (
        (wf["next_monotony"] > MONO_THR) & (wf["next_strain"] > strain_thr)
    ).astype(int)

    # composite risk
    wf["y_risk"] = (
        (wf["y_acwr_high"] == 1)
        | (wf["y_spike_high"] == 1)
        | (wf["y_mono_strain_high"] == 1)
    ).astype(int)

    labels = wf[
        ["week_start", "y_risk", "y_acwr_high", "y_spike_high", "y_mono_strain_high"]
    ].dropna()

    labels = labels.iloc[:-1].copy()

    with connect() as con:
        con.execute("DELETE FROM labels_weekly;")
        labels.to_sql("labels_weekly", con, if_exists="append", index=False)

    print("Risk label counts:")
    print(labels[["y_risk", "y_acwr_high", "y_spike_high", "y_mono_strain_high"]].sum())
    print(f"Wrote {len(labels)} weekly labels")
