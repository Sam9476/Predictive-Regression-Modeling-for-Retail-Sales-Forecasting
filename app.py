# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ── 1) FIRST Streamlit command ─────────────────────────────────────────────────
st.set_page_config(page_title="📈 7‑Day Rossmann Forecast", layout="wide")

# ── 2) Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

@st.cache_data
def load_store_data():
    return pd.read_csv("store.csv")

model  = load_model()
scaler = load_scaler()
store_df = load_store_data()

# ── 3) App inputs ─────────────────────────────────────────────────────────────────
st.sidebar.header("🔧 Store & Promo Settings")
store_id = st.sidebar.number_input("Store ID", min_value=1, max_value=1115, value=1, step=1)
promo     = st.sidebar.selectbox("Promo today? (0 = No, 1 = Yes)", [0,1], index=1)

st.sidebar.header("📅 Forecast Settings")
start_date = st.sidebar.date_input("Start date", pd.Timestamp.today().date())

# Next 7 days
dates = pd.date_range(start_date, periods=7, freq="D")
date_strs = [d.strftime("%Y-%m-%d") for d in dates]

school_hols = st.sidebar.multiselect("School holidays in these dates", options=date_strs)
state_hols  = st.sidebar.multiselect("State holidays in these dates",  options=date_strs)

if not st.sidebar.button("🔮 Run 7‑Day Forecast"):
    st.info("👈 Set parameters and click **Run 7‑Day Forecast**")
    st.stop()

# ── 4) Build and preprocess features ─────────────────────────────────────────────
numeric_cols = [
    "Store", "DayOfWeek", "Day", "Month",
    "CompetitionDistance", "CompetitionMonths",
    "Promo", "Promo2", "Promo2Months", "PromoInMonth"
]

def make_features_for_date(store_id, date, promo_flag, is_school_hol, is_state_hol):
    # 4.1 Pull store metadata
    s = store_df.loc[store_df.Store == store_id].iloc[0]

    # 4.2 Compute competition months
    if pd.notna(s.CompetitionOpenSinceYear) and s.CompetitionOpenSinceYear>0:
        comp_since = pd.Timestamp(year=int(s.CompetitionOpenSinceYear),
                                  month=int(s.CompetitionOpenSinceMonth), day=1)
        comp_months = max(0, (date.year - comp_since.year)*12 + (date.month - comp_since.month))
    else:
        comp_months = 0

    # 4.3 Compute promo2 months & in‑month
    if s.Promo2==1 and pd.notna(s.Promo2SinceYear) and s.Promo2SinceYear>0:
        # week → approximate month
        promo2_since = pd.Timestamp.fromisocalendar(int(s.Promo2SinceYear), int(s.Promo2SinceWeek), 1)
        promo2_months = max(0, (date.year - promo2_since.year)*12 + (date.month - promo2_since.month))
        promo_interval_list = str(s.PromoInterval).split(",") if pd.notna(s.PromoInterval) else []
        promo_in_month = int(date.strftime("%b") in promo_interval_list)
    else:
        promo2_months = 0
        promo_in_month  = 0

    # 4.4 Raw numeric row
    raw = {
        "Store": store_id,
        "DayOfWeek": date.weekday()+1,
        "Day": date.day,
        "Month": date.month,
        "CompetitionDistance": s.CompetitionDistance,
        "CompetitionMonths": comp_months,
        "Promo": promo_flag,
        "Promo2": s.Promo2,
        "Promo2Months": promo2_months,
        "PromoInMonth": promo_in_month,
    }
    # 4.5 Scale numeric
    num_df = pd.DataFrame([raw])[numeric_cols]
    num_scaled = scaler.transform(num_df)
    num_df_scaled = pd.DataFrame(num_scaled, columns=numeric_cols)

    # 4.6 One‑hot categorical
    cats = {}
    # init all dummies to 0
    for col, vals in [
        ("StoreType", ["a","b","c","d"]),
        ("Assortment", ["a","b","c"]),
        ("StateHoliday", ["0","a","b","c"]),
        ("SchoolHoliday", [0,1])
    ]:
        for v in vals:
            cats[f"{col}_{v}"] = 0
    # fill the active dummy
    cats[f"StoreType_{s.StoreType}"]     = 1
    cats[f"Assortment_{s.Assortment}"]   = 1
    cats[f"StateHoliday_{'b' if date.strftime('%Y-%m-%d') in state_hols else '0'}"] = 1
    cats[f"SchoolHoliday_{1 if date.strftime('%Y-%m-%d') in school_hols else 0}"]   = 1

    return pd.concat([num_df_scaled, pd.DataFrame([cats])], axis=1)

# build feature matrix for all 7 days
feature_frames = [
    make_features_for_date(store_id, d, promo, d.strftime("%Y-%m-%d") in school_hols, d.strftime("%Y-%m-%d") in state_hols)
    for d in dates
]
X = pd.concat(feature_frames, ignore_index=True)

# ── 5) Predict & display ─────────────────────────────────────────────────────────
preds_scaled = model.predict(X)
# Your notebook trained on raw Sales, so no inverse‐transform needed.
# If you DID log1p or scaler.transform(y), invert here.
sales_preds = preds_scaled

# Line chart
st.subheader(f"📈 7‑Day Forecast starting {start_date}")
chart_df = pd.DataFrame({"Date": dates, "Predicted Sales": sales_preds}).set_index("Date")
st.line_chart(chart_df)

# Insights
st.markdown("### 🔍 Insights")
idx_max = int(np.argmax(sales_preds))
idx_min = int(np.argmin(sales_preds))
avg    = float(np.mean(sales_preds))

st.write(f"- **Highest** sales: {sales_preds[idx_max]:,.0f} on **{dates[idx_max].date()}**")
st.write(f"- **Lowest**  sales: {sales_preds[idx_min]:,.0f} on **{dates[idx_min].date()}**")
st.write(f"- **Average** sales: {avg:,.0f}")

trend = (
    "Increasing ↗️" if sales_preds[-1] > sales_preds[0]
    else "Decreasing ↘️" if sales_preds[-1] < sales_preds[0]
    else "Flat ↔️"
)
st.write(f"- **Trend** over week: {trend}")
