import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Config
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = REPO_ROOT / "data" / "endurance.db"

st.set_page_config(
    page_title="Endurance Forecasting",
    page_icon="🏃",
    layout="wide",
)


# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def run_query(query: str) -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(query, con)
    finally:
        con.close()
    return df


@st.cache_data
def load_weekly() -> pd.DataFrame:
    return run_query(
        """
        SELECT
            wf.week_start,
            wf.distance_km,
            wf.time_min,
            wf.elev_m,
            wf.trimp,
            wf.acwr,
            wf.monotony_7d,
            wf.strain_7d,
            wf.easy_pct,
            wf.lag1_distance_km,
            wf.lag1_trimp,
            lb.y_risk,
            lb.y_acwr_high,
            lb.y_spike_high,
            lb.y_mono_strain_high
        FROM weekly_features wf
        LEFT JOIN labels_weekly lb USING(week_start)
        ORDER BY wf.week_start
        """
    )


@st.cache_data
def load_daily() -> pd.DataFrame:
    return run_query(
        """
        SELECT
            date,
            run_count,
            distance_km,
            time_min,
            elev_m,
            avg_hr_mean,
            trimp,
            easy_distance_km,
            hard_distance_km,
            easy_time_min,
            hard_time_min
        FROM daily_features
        ORDER BY date
        """
    )


def compute_risk_probability(row: pd.Series) -> float:
    """
    Temporary heuristic proxy for the app until you save/load the trained model.
    This is not the model itself. It gives the UI something sensible to display.
    """
    # Conservative, interpretable weighted score
    acwr = 0.0 if pd.isna(row["acwr"]) else float(row["acwr"])
    lag1_trimp = 0.0 if pd.isna(row["lag1_trimp"]) else float(row["lag1_trimp"])
    lag1_distance = (
        0.0 if pd.isna(row["lag1_distance_km"]) else float(row["lag1_distance_km"])
    )

    # Simple normalisations to keep score stable
    acwr_component = min(acwr / 1.5, 2.0)
    trimp_component = min(lag1_trimp / 400.0, 2.0)
    dist_component = min(lag1_distance / 100.0, 2.0)

    score = -2.2 + 1.5 * acwr_component + 0.8 * trimp_component + 0.5 * dist_component
    prob = 1.0 / (1.0 + np.exp(-score))
    return float(prob)


def style_risk(prob: float, threshold: float) -> str:
    return "High" if prob >= threshold else "Lower"


def format_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


# -----------------------------
# Load data
# -----------------------------
if not DB_PATH.exists():
    st.error(f"Database not found at {DB_PATH}. Run the pipeline first.")
    st.stop()

weekly = load_weekly()
daily = load_daily()

if weekly.empty:
    st.error("weekly_features is empty. Run the pipeline first.")
    st.stop()

weekly["week_start"] = pd.to_datetime(weekly["week_start"])
daily["date"] = pd.to_datetime(daily["date"])

# Use all rows except the final one if y_risk is null there
weekly_valid = weekly.copy()

# Compute temporary app-side risk probabilities
weekly_valid["risk_probability"] = weekly_valid.apply(compute_risk_probability, axis=1)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Endurance Forecasting")
st.sidebar.caption("Training load and proxy injury-risk monitoring")

threshold = st.sidebar.slider(
    "Risk threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.30,
    step=0.05,
)

min_date = weekly_valid["week_start"].min().date()
max_date = weekly_valid["week_start"].max().date()

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

show_only_risk = st.sidebar.checkbox("Highlight labelled risk weeks", value=True)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
else:
    start_date, end_date = (
        weekly_valid["week_start"].min(),
        weekly_valid["week_start"].max(),
    )

weekly_view = weekly_valid[
    (weekly_valid["week_start"] >= start_date)
    & (weekly_valid["week_start"] <= end_date)
].copy()

# -----------------------------
# Header
# -----------------------------
st.title("Endurance Forecasting")
st.caption(
    "Proxy injury-risk modelling from Strava-derived workload features. "
    "Risk is defined from future workload-regime signals, not clinical injury outcomes."
)

tabs = st.tabs(["Overview", "Training Load", "Risk Model", "Data Explorer"])

# -----------------------------
# Overview
# -----------------------------
with tabs[0]:
    latest = weekly_valid.iloc[-1]
    current_prob = float(latest["risk_probability"])
    current_label = style_risk(current_prob, threshold)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current risk probability", format_pct(current_prob))
    c2.metric("Risk status", current_label)
    c3.metric(
        "Current ACWR", f"{latest['acwr']:.2f}" if pd.notna(latest["acwr"]) else "N/A"
    )
    c4.metric(
        "Last week distance",
        f"{latest['distance_km']:.1f} km" if pd.notna(latest["distance_km"]) else "N/A",
    )

    st.subheader("Weekly distance")
    distance_chart = weekly_view[["week_start", "distance_km"]].set_index("week_start")
    st.line_chart(distance_chart)

    if show_only_risk:
        risk_weeks = weekly_view[weekly_view["y_risk"] == 1][
            ["week_start", "distance_km"]
        ]
        if not risk_weeks.empty:
            st.caption("Weeks labelled as future high-risk regime")
            st.dataframe(risk_weeks, use_container_width=True, hide_index=True)

    st.subheader("Latest model inputs")
    latest_inputs = pd.DataFrame({
        "feature": ["lag1_distance_km", "lag1_trimp", "acwr"],
        "value": [
            latest.get("lag1_distance_km"),
            latest.get("lag1_trimp"),
            latest.get("acwr"),
        ],
    })
    st.dataframe(latest_inputs, use_container_width=True, hide_index=True)

    with st.expander("What this app is showing"):
        st.write(
            "This app tracks training load and estimates the probability of entering a "
            "high-risk workload regime in the following week. The current MVP uses SQLite-backed "
            "features from your pipeline and a temporary app-side probability score for display."
        )

# -----------------------------
# Training Load
# -----------------------------
with tabs[1]:
    st.subheader("Weekly training load")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Weekly distance (km)**")
        st.line_chart(weekly_view.set_index("week_start")[["distance_km"]])

        st.markdown("**Weekly TRIMP proxy**")
        st.line_chart(weekly_view.set_index("week_start")[["trimp"]])

    with col2:
        st.markdown("**ACWR**")
        st.line_chart(weekly_view.set_index("week_start")[["acwr"]])
        st.caption(
            "Reference threshold: 1.5 often used as a high-risk workload marker."
        )

        st.markdown("**Monotony and strain**")
        st.line_chart(weekly_view.set_index("week_start")[["monotony_7d", "strain_7d"]])

    st.subheader("Easy vs hard split")
    easy_hard = weekly_view.copy()
    easy_hard["hard_pct"] = 1.0 - easy_hard["easy_pct"].fillna(0.0)
    display_df = easy_hard[["week_start", "easy_pct", "hard_pct"]].set_index(
        "week_start"
    )
    st.line_chart(display_df)

# -----------------------------
# Risk Model
# -----------------------------
with tabs[2]:
    st.subheader("Risk model view")

    latest_probs = weekly_view[["week_start", "risk_probability"]].set_index(
        "week_start"
    )
    st.markdown("**Predicted risk probability over time**")
    st.line_chart(latest_probs)

    st.markdown("**Current classification at chosen threshold**")
    risk_df = weekly_view[["week_start", "risk_probability"]].copy()
    risk_df["predicted_risk"] = (risk_df["risk_probability"] >= threshold).astype(int)
    st.dataframe(risk_df.tail(12), use_container_width=True, hide_index=True)

    st.markdown("**Label breakdown in dataset**")
    label_counts = pd.DataFrame({
        "label": ["y_risk", "y_acwr_high", "y_spike_high", "y_mono_strain_high"],
        "count": [
            int(weekly["y_risk"].fillna(0).sum()),
            int(weekly["y_acwr_high"].fillna(0).sum()),
            int(weekly["y_spike_high"].fillna(0).sum()),
            int(weekly["y_mono_strain_high"].fillna(0).sum()),
        ],
    })
    st.dataframe(label_counts, use_container_width=True, hide_index=True)

    st.markdown("**Model notes**")
    st.write(
        "The current training pipeline uses a simple, interpretable feature set: "
        "`lag1_distance_km`, `lag1_trimp`, and `acwr`. "
        "This app currently shows a temporary probability proxy for presentation. "
        "Next step: load the trained logistic regression pipeline and saved metrics directly."
    )

# -----------------------------
# Data Explorer
# -----------------------------
with tabs[3]:
    st.subheader("Weekly feature table")
    st.dataframe(weekly_view, use_container_width=True, hide_index=True)

    st.subheader("Daily feature table")
    daily_view = daily[
        (daily["date"] >= start_date) & (daily["date"] <= end_date)
    ].copy()
    st.dataframe(daily_view, use_container_width=True, hide_index=True)
