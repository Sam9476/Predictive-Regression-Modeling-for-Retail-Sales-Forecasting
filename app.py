import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ── 1) FIRST Streamlit command ─────────────────────────────────────────────────
st.set_page_config(page_title="📈 7‑Day Rossmann Forecast", layout="wide")

# ── 2) Load model once ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# ── 3) Feature builder (single row) ────────────────────────────────────────────
def make_features(row):
    # Numeric
    d = {
        "Store": row["Store"],
        "DayOfWeek": row["Date"].weekday() + 1,
        "Day": row["Date"].day,
        "Month": row["Date"].month,
        "CompetitionDistance": row["CompetitionDistance"],
        # months since competition opened
        "CompetitionMonths": max(0, (row["Date"].year - row["CompYear"]) * 12 + (row["Date"].month - row["CompMonth"])),
        "Promo": row["Promo"],
        "Promo2": row["Promo2"],
        "Promo2Months": max(0, (row["Date"].year - row["P2Year"]) * 12 + (row["Date"].month - row["P2Month"])),
        "PromoInMonth": int(row["Date"].strftime("%b") in row["PromoIntervalList"])
    }

    # One‑hots template
    for col in (
        [f"StoreType_{x}" for x in ["a","b","c","d"]] +
        [f"Assortment_{x}" for x in ["a","b","c"]] +
        [f"StateHoliday_{x}" for x in ["0","a","b","c"]] +
        [f"SchoolHoliday_{x}" for x in [0,1]]
    ):
        d[col] = 0

    # Fill actual one‑hots
    d[f"StoreType_{row['StoreType']}"]   = 1
    d[f"Assortment_{row['Assortment']}"] = 1
    d[f"StateHoliday_{row['StateHoliday']}"] = 1
    d[f"SchoolHoliday_{row['SchoolHoliday']}"] = 1

    return pd.DataFrame([d])


# ── 4) Sidebar inputs ─────────────────────────────────────────────────────────
st.sidebar.header("Store & Promo Details")

store = st.sidebar.number_input("Store ID", 1, 1115, value=1)
store_type = st.sidebar.selectbox("Store Type", ["a","b","c","d"])
assortment = st.sidebar.selectbox("Assortment", ["a","b","c"])
competition_distance = st.sidebar.number_input("Competition Distance (m)", 0.0, 1e5, 5000.0, step=100.0)
comp_month = st.sidebar.slider("Competition Open Since Month", 1, 12, 1)
comp_year  = st.sidebar.number_input("Competition Open Since Year", 1900, pd.Timestamp.today().year, 2010)

promo    = st.sidebar.selectbox("Promo (0/1)", [0,1])
promo2   = st.sidebar.selectbox("Promo2 (0/1)", [0,1])
p2_week  = st.sidebar.slider("Promo2 Since Week", 1, 52, 1)
p2_year  = st.sidebar.number_input("Promo2 Since Year", 1900, pd.Timestamp.today().year, 2010)
p_interval = st.sidebar.text_input("PromoInterval (e.g. Jan,Apr...)", "")

# Precompute list for feature builder
promo_interval_list = [m.strip() for m in p_interval.split(",") if m.strip()]

st.sidebar.header("Forecast Settings")
start_date = st.sidebar.date_input("Start Date", pd.Timestamp.today().date())

# 7-day window
dates = pd.date_range(start_date, periods=7, freq="D")
date_strs = [d.strftime("%Y-%m-%d") for d in dates]

# Let user pick which of these dates are holidays
school_hols = st.sidebar.multiselect(
    "Select School Holiday Dates",
    options=date_strs
)
state_hols = st.sidebar.multiselect(
    "Select State Holiday Dates",
    options=date_strs
)

if st.sidebar.button("🔮 Run 7‑Day Forecast"):
    # Build features for each date
    feats = []
    for d in dates:
        row = {
            "Store": store,
            "StoreType": store_type,
            "Assortment": assortment,
            "CompetitionDistance": competition_distance,
            "CompMonth": comp_month,
            "CompYear": comp_year,
            "Promo": promo,
            "Promo2": promo2,
            "P2Month": pd.Timestamp.today().week if promo2 else 0,  # fallback if needed
            "P2Week": p2_week,
            "P2Year": p2_year,
            "PromoIntervalList": promo_interval_list,
            "Date": d,
            # dynamically assign holiday flags
            "StateHoliday": "b" if d.strftime("%Y-%m-%d") in state_hols else "0",
            "SchoolHoliday": 1 if d.strftime("%Y-%m-%d") in school_hols else 0
        }
        feats.append(make_features(row))

    X = pd.concat(feats, ignore_index=True)
    # … after you’ve built X (the full feature DataFrame for your 7‑day window) …

# ── DEBUGGING: inspect feature alignment ───────────────────────────────────────
st.write("🔍 **DEBUG: Feature matrix (first 5 rows):**")
st.write(X.head())

st.write("🔍 **DEBUG: Columns in feature matrix:**")
st.write(list(X.columns))

st.write("🔍 **DEBUG: Model expected feature names:**")
st.write(list(model.feature_names_in_))

# Show differences
missing = list(set(model.feature_names_in_) - set(X.columns))
extra   = list(set(X.columns) - set(model.feature_names_in_))
st.write(f"❌ Missing columns that model expects: {missing}")
st.write(f"➕ Extra columns not used by model:     {extra}")

# ── DEBUGGING: inspect model itself ───────────────────────────────────────────
st.write("🔍 **DEBUG: Loaded model parameters:**")
st.json(model.get_params())

# Now you’ll see exactly how your input lines up before prediction
# preds = model.predict(X)
# … rest of your code …

    preds = model.predict(X)

    # Display line chart
    st.subheader(f"📈 7‑Day Sales Forecast from {start_date}")
    chart_df = pd.DataFrame({
        "Date": dates,
        "Predicted Sales": preds
    }).set_index("Date")
    st.line_chart(chart_df)

    # ── Insights Panel ─────────────────────────────────────────────────────────
    st.markdown("### 🔍 Insights")
    max_day  = dates[preds.argmax()].strftime("%Y-%m-%d")
    min_day  = dates[preds.argmin()].strftime("%Y-%m-%d")
    avg_pred = preds.mean()

    st.write(f"- **Highest** predicted sales: {preds.max():,.0f} on **{max_day}**")
    st.write(f"- **Lowest**  predicted sales: {preds.min():,.0f} on **{min_day}**")
    st.write(f"- **Average** predicted sales over 7 days: {avg_pred:,.0f}")

    # Simple trend analysis
    if preds[-1] > preds[0]:
        st.write("➡️ Overall trend: **Increasing** sales over the week.")
    elif preds[-1] < preds[0]:
        st.write("⬇️ Overall trend: **Decreasing** sales over the week.")
    else:
        st.write("🔁 Overall trend: **Flat** sales over the week.")

else:
    st.info("👈 Set parameters and click **Run 7‑Day Forecast**")
