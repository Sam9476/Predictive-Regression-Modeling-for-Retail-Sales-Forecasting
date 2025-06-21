# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ── 1) FIRST Streamlit command ─────────────────────────────────────────────────
st.set_page_config(page_title="🔧 Debug Rossmann App", layout="wide")

# ── 2) Load model, scaler & store metadata ──────────────────────────────────────
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

# ── 3) Define numeric_cols exactly as in training ───────────────────────────────
numeric_cols = [
    "Store", "DayOfWeek", "Day", "Month",
    "CompetitionDistance", "CompetitionMonths",
    "Promo", "Promo2", "Promo2Months", "PromoInMonth"
]

# ── 4) Debug: check scaler vs numeric_cols ─────────────────────────────────────
st.write("### 🔍 DEBUG: Scaler.feature_names_in_ (if available)")
try:
    st.write(list(scaler.feature_names_in_))
except AttributeError:
    st.write("⚠️ scaler.feature_names_in_ not found (old sklearn).")
st.write("### 🔍 DEBUG: numeric_cols in app")
st.write(numeric_cols)

# ── 5) App sidebar inputs ───────────────────────────────────────────────────────
st.sidebar.header("Store & Promo Settings")
store_id = st.sidebar.number_input("Store ID", min_value=1, max_value=1115, value=1, step=1)
promo_flag = st.sidebar.selectbox("Promo today? (0=No, 1=Yes)", [0,1], index=1)

st.sidebar.header("Forecast Settings")
start_date = st.sidebar.date_input("Start date", pd.Timestamp.today().date())
dates = pd.date_range(start_date, periods=7, freq="D")
date_strs = [d.strftime("%Y-%m-%d") for d in dates]

school_hols = st.sidebar.multiselect("School holiday dates", options=date_strs)
state_hols  = st.sidebar.multiselect("State holiday dates",  options=date_strs)

if not st.sidebar.button("🔮 Run 7‑Day Forecast"):
    st.info("👈 Set parameters and click **Run 7‑Day Forecast**")
    st.stop()

# ── 6) Feature builder ──────────────────────────────────────────────────────────
def make_features_for_date(store_id, date, promo, school_hol_flag, state_hol_flag):
    # pull store metadata
    s = store_df.loc[store_df.Store == store_id].iloc[0]
    # compute CompetitionMonths
    if pd.notna(s.CompetitionOpenSinceYear) and s.CompetitionOpenSinceYear>0:
        comp_since = pd.Timestamp(year=int(s.CompetitionOpenSinceYear),
                                  month=int(s.CompetitionOpenSinceMonth), day=1)
        comp_months = max(0, (date.year - comp_since.year)*12 + (date.month - comp_since.month))
    else:
        comp_months = 0
    # compute Promo2Months & PromoInMonth
    if s.Promo2==1 and pd.notna(s.Promo2SinceYear) and s.Promo2SinceYear>0:
        promo2_since = pd.Timestamp.fromisocalendar(int(s.Promo2SinceYear),
                                                     int(s.Promo2SinceWeek), 1)
        promo2_months = max(0, (date.year - promo2_since.year)*12 + (date.month - promo2_since.month))
        p_interval = str(s.PromoInterval).split(",") if pd.notna(s.PromoInterval) else []
        promo_in_month = int(date.strftime("%b") in p_interval)
    else:
        promo2_months = 0
        promo_in_month = 0

    raw = {
        "Store": store_id,
        "DayOfWeek": date.weekday()+1,
        "Day": date.day,
        "Month": date.month,
        "CompetitionDistance": s.CompetitionDistance,
        "CompetitionMonths": comp_months,
        "Promo": promo,
        "Promo2": s.Promo2,
        "Promo2Months": promo2_months,
        "PromoInMonth": promo_in_month,
    }

    # ── Debug: show numeric DF before scaling ────────────────────────────────────
    num_df = pd.DataFrame([raw])[numeric_cols]
    st.write(f"### 🔍 DEBUG: raw numeric features for {date.date()}")
    st.write(num_df)

    # scale numeric
    num_scaled = scaler.transform(num_df)
    num_scaled_df = pd.DataFrame(num_scaled, columns=numeric_cols)

    # one-hot categories
    cats = {}
    for col, vals in [
        ("StoreType", ["a","b","c","d"]),
        ("Assortment", ["a","b","c"]),
        ("StateHoliday", ["0","a","b","c"]),
        ("SchoolHoliday", [0,1])
    ]:
        for v in vals:
            cats[f"{col}_{v}"] = 0
    cats[f"StoreType_{s.StoreType}"]   = 1
    cats[f"Assortment_{s.Assortment}"] = 1
    cats[f"StateHoliday_{'b' if date.strftime('%Y-%m-%d') in state_hols else '0'}"] = 1
    cats[f"SchoolHoliday_{1 if date.strftime('%Y-%m-%d') in school_hols else 0}"]   = 1

    return pd.concat([num_scaled_df, pd.DataFrame([cats])], axis=1)

# ── 7) Build X for all dates ─────────────────────────────────────────────────────
feature_frames = []
for d in dates:
    feature_frames.append(
        make_features_for_date(store_id, d, promo_flag,
                               d.strftime("%Y-%m-%d") in school_hols,
                               d.strftime("%Y-%m-%d") in state_hols)
    )
X = pd.concat(feature_frames, ignore_index=True)

# ── 8) Final debug before predict ───────────────────────────────────────────────
st.write("### 🔍 DEBUG: Final feature matrix (7×N)")
st.dataframe(X)
st.write("### 🔍 DEBUG: Feature matrix columns")
st.write(sorted(X.columns.tolist()))
st.write("### 🔍 DEBUG: Model expects")
st.write(sorted(model.feature_names_in_.tolist()))
missing = sorted(set(model.feature_names_in_) - set(X.columns))
extra   = sorted(set(X.columns) - set(model.feature_names_in_))
st.write("❌ Missing:", missing)
st.write("➕ Extra:", extra)

# ── 9) Predict and display ─────────────────────────────────────────────────────
preds = model.predict(X)
sales_preds = preds  # if no inversion needed

st.subheader(f"📈 7‑Day Forecast from {start_date}")
chart_df = pd.DataFrame({"Date": dates, "Predicted Sales": sales_preds}).set_index("Date")
st.line_chart(chart_df)

st.markdown("### 🔍 Insights")
idx_max = int(np.argmax(sales_preds))
idx_min = int(np.argmin(sales_preds))
avg_val = float(np.mean(sales_preds))
st.write(f"- Highest: {sales_preds[idx_max]:.3f} on {dates[idx_max].date()}")
st.write(f"- Lowest:  {sales_preds[idx_min]:.3f} on {dates[idx_min].date()}")
st.write(f"- Average: {avg_val:.3f}")
