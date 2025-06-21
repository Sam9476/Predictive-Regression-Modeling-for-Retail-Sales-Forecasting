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

# ── 3) Define numeric columns ────────────────────────────────────────────────────
numeric_cols = [
    "Store", "DayOfWeek", "Day", "Month",
    "CompetitionDistance", "CompetitionMonths",
    "Promo", "Promo2", "Promo2Months", "PromoInMonth"
]

# ── 4) Sidebar inputs ────────────────────────────────────────────────────────────
st.sidebar.header("🛒 Store & Promo Settings")
store_id   = st.sidebar.number_input("Store ID", 1, 1115, 1, 1)
promo_flag = st.sidebar.selectbox("Promo today? (0=No,1=Yes)", [0,1], index=1)

st.sidebar.header("📅 Forecast Settings")
start_date = st.sidebar.date_input("Start Date", pd.Timestamp.today().date())
dates = pd.date_range(start_date, periods=7, freq="D")
date_strs = [d.strftime("%Y-%m-%d") for d in dates]

school_hols = st.sidebar.multiselect("School holiday dates", options=date_strs)
state_hols  = st.sidebar.multiselect("State holiday dates", options=date_strs)

if not st.sidebar.button("🔮 Run 7‑Day Forecast"):
    st.info("👈 Adjust settings and click **Run 7‑Day Forecast**")
    st.stop()

# ── 5) Feature builder ──────────────────────────────────────────────────────────
def make_features(store_id, date, promo_flag, is_school_hol, is_state_hol):
    s = store_df.loc[store_df.Store == store_id].iloc[0]
    # Competition months
    if pd.notna(s.CompetitionOpenSinceYear) and s.CompetitionOpenSinceYear>0:
        comp_since = pd.Timestamp(int(s.CompetitionOpenSinceYear),
                                  int(s.CompetitionOpenSinceMonth), 1)
        comp_months = max(0, (date.year - comp_since.year)*12 + (date.month - comp_since.month))
    else:
        comp_months = 0

    # Promo2 months & in-month
    if s.Promo2==1 and pd.notna(s.Promo2SinceYear) and s.Promo2SinceYear>0:
        p2_since = pd.Timestamp.fromisocalendar(int(s.Promo2SinceYear),
                                                 int(s.Promo2SinceWeek), 1)
        p2_months = max(0, (date.year - p2_since.year)*12 + (date.month - p2_since.month))
        interval = str(s.PromoInterval).split(",") if pd.notna(s.PromoInterval) else []
        p_in_month = int(date.strftime("%b") in interval)
    else:
        p2_months = 0
        p_in_month = 0

    # Raw numeric features
    raw = {
        "Store": store_id,
        "DayOfWeek": date.weekday()+1,
        "Day": date.day,
        "Month": date.month,
        "CompetitionDistance": s.CompetitionDistance,
        "CompetitionMonths": comp_months,
        "Promo": promo_flag,
        "Promo2": s.Promo2,
        "Promo2Months": p2_months,
        "PromoInMonth": p_in_month,
    }

    # Build, scale, and reconstruct numeric DataFrame
    num_arr = np.array([raw[col] for col in numeric_cols]).reshape(1, -1)
    scaled_arr = scaler.transform(num_arr)  # bypasses feature_names_ check
    num_scaled = pd.DataFrame(scaled_arr, columns=numeric_cols)

    # One-hot categorical
    cats = {}
    for col, vals in [
        ("StoreType", ["a","b","c","d"]),
        ("Assortment", ["a","b","c"]),
        ("StateHoliday", ["0","a","b","c"]),
        ("SchoolHoliday", [0,1]),
    ]:
        for v in vals:
            cats[f"{col}_{v}"] = 0
    cats[f"StoreType_{s.StoreType}"] = 1
    cats[f"Assortment_{s.Assortment}"] = 1
    cats[f"StateHoliday_{('b' if date.strftime('%Y-%m-%d') in state_hols else '0')}"] = 1
    cats[f"SchoolHoliday_{(1 if date.strftime('%Y-%m-%d') in school_hols else 0)}"] = 1

    return pd.concat([num_scaled, pd.DataFrame([cats])], axis=1)

# ── 6) Build feature matrix for all 7 days ────────────────────────────────────
feature_list = [make_features(store_id, d, promo_flag,
                              d.strftime("%Y-%m-%d") in school_hols,
                              d.strftime("%Y-%m-%d") in state_hols)
                for d in dates]
X = pd.concat(feature_list, ignore_index=True)

# ── 7) Predict & display ───────────────────────────────────────────────────────
preds = model.predict(X)
sales_preds = preds  # if your model outputs raw Sales; apply inverse-transform here if needed

# Chart
st.subheader(f"📈 7‑Day Sales Forecast from {start_date}")
df_chart = pd.DataFrame({"Date": dates, "Predicted Sales": sales_preds}).set_index("Date")
st.line_chart(df_chart)

# Insights
st.markdown("### 🔍 Insights")
i_max = int(np.argmax(sales_preds))
i_min = int(np.argmin(sales_preds))
st.write(f"- **Highest** sales: {sales_preds[i_max]:,.0f} on {dates[i_max].date()}")
st.write(f"- **Lowest**  sales: {sales_preds[i_min]:,.0f} on {dates[i_min].date()}")
st.write(f"- **Average** sales: {sales_preds.mean():,.0f}")
trend = "Increasing ↗️" if sales_preds[-1]>sales_preds[0] else "Decreasing ↘️" if sales_preds[-1]<sales_preds[0] else "Flat ↔️"
st.write(f"- **Trend** over week: {trend}")
