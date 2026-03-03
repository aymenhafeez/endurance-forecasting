import sqlite3
import pandas as pd
import streamlit as st

DB_PATH = "data/endurance.db"


@st.cache_data
def load_table(query: str) -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(query, con)
    con.close()

    return df


st.title("Endurance Forecasting")

tab1, tab2, tab3 = st.tabs(["Training Load", "Forecast", "Model Performance"])

with tab1:
    weekly = load_table(
        "SELECT week_start, distance_km, trimp, acwr, monotony_7d, strain_7d, easy_pct, dist_7d, trimp_7d FROM weekly_features ORDER BY week_start"
    )
    st.subheader("Weekly volume")
    st.line_chart(weekly.set_index("week_start")["distance_km"])

    st.subheader("Weekly volume vs training load")
    st.scatter_chart(
        weekly.set_index("dist_7d")["trimp_7d"],
        x_label="Weekly volume (km)",
        y_label="TRIMP",
    )

    st.subheader("Training load (TRIMP proxy)")
    st.line_chart(weekly.set_index("week_start")["trimp"])

    st.subheader("ACWR")
    st.line_chart(weekly.set_index("week_start")["acwr"])

with tab2:
    # show last known week and naive forecast
    weekly = load_table(
        "SELECT week_start, distance_km FROM weekly_features ORDER BY week_start"
    )
    last_week = weekly.iloc[-1]
    st.metric("Last week distance (km)", f"{last_week.distance_km:.1f}")

    # TODO: Add mode forecasts predictions weekly table

with tab3:
    # TODO: show baseline vs ridge metrics here from training output or a saved metrics table
    st.write(
        "TODO: show baseline vs ridge metrics here from training output or a saved metrics table"
    )
