import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ── 1) Initial Streamlit setup ───────────────────────────────────────────────
st.set_page_config(page_title="📈 Rossmann 7-Day Forecast", layout="wide")

# ── 2) Load model, scaler, and store metadata ────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

@st.cache_data
def load_store_data():
    return pd.read_csv("store.csv")

model = load_model()
scaler = load_scaler()
store_df = load_store_data()

# ── 3) Define expected feature structure ─────────────────────────────────────
numeric_cols = [
    "Store", "DayOfWeek", "Day", "Month",
    "CompetitionDistance", "CompetitionMonths",
    "Promo", "Promo2", "Promo2Months", "PromoInMonth"
]

onehot_cols = [
    "StoreType_a", "StoreType_b", "StoreType_c", "StoreType_d",
    "Assortment_a", "Assortment_b", "Assortment_c",
    "StateHoliday_0", "StateHoliday_a", "StateHoliday_b", "StateHoliday_c",
    "SchoolHoliday_0", "SchoolHoliday_1"
]

final_cols = numeric_cols + onehot_cols

# ── 4) Sidebar user input ─────────────────────────────────────────────────────
st.sidebar.header("📋 Store & Promo Settings")
store_id   = st.sidebar.number_input("Store ID", 1, 1115, 1)
promo_flag = st.sidebar.selectbox("Promo today? (0=No, 1=Yes)", [0, 1], index=1)

st.sidebar.header("📅 Forecast Window")
start_date = st.sidebar.date_input("Start date", pd.Timestamp.today().date())
dates = pd.date_range(start_date, periods=7, freq="D")
date_strs = [d.strftime("%Y-%m-%d") for d in dates]

school_hols = st.sidebar.multiselect("Select school holidays", options=date_strs)
state_hols  = st.sidebar.multiselect("Select state holidays", options=date_strs)

if not st.sidebar.button("🔮 Run 7-Day Forecast"):
    st.info("👈 Configure input and click the forecast button.")
    st.stop()

# ── 5) Feature builder for one day ────────────────────────────────────────────
def make_features(date):
    s = store_df.loc[store_df["Store"] == store_id].iloc[0]

    # CompetitionMonths
    if pd.notna(s.CompetitionOpenSinceYear) and pd.notna(s.CompetitionOpenSinceMonth):
        comp_since = pd.Timestamp(int(s.CompetitionOpenSinceYear), int(s.CompetitionOpenSinceMonth), 1)
        comp_months = max(0, (date.year - comp_since.year) * 12 + (date.month - comp_since.month))
    else:
        comp_months = 0

    # Promo2Months and PromoInMonth
    if s.Promo2 == 1 and pd.notna(s.Promo2SinceYear) and pd.notna(s.Promo2SinceWeek):
        p2_start = pd.Timestamp.fromisocalendar(int(s.Promo2SinceYear), int(s.Promo2SinceWeek), 1)
        p2_months = max(0, (date.year - p2_start.year) * 12 + (date.month - p2_start.month))
        promo_in_month = int(date.strftime("%b") in str(s.PromoInterval).split(','))
    else:
        p2_months = 0
        promo_in_month = 0

    # Numeric features (before scaling)
    raw_numeric = {
        "Store": store_id,
        "DayOfWeek": date.weekday() + 1,
        "Day": date.day,
        "Month": date.month,
        "CompetitionDistance": s.CompetitionDistance,
        "CompetitionMonths": comp_months,
        "Promo": promo_flag,
        "Promo2": s.Promo2,
        "Promo2Months": p2_months,
        "PromoInMonth": promo_in_month
    }

    # Apply scaling
    num_arr = np.array([raw_numeric[c] for c in numeric_cols]).reshape(1, -1)
    num_scaled = scaler.transform(num_arr)
    num_df = pd.DataFrame(num_scaled, columns=numeric_cols)

    # One-hot encoding
    onehots = dict.fromkeys(onehot_cols, 0)
    onehots[f"StoreType_{s.StoreType}"] = 1
    onehots[f"Assortment_{s.Assortment}"] = 1
    state_hol = "b" if date.strftime("%Y-%m-%d") in state_hols else "0"
    school_hol = 1 if date.strftime("%Y-%m-%d") in school_hols else 0
    onehots[f"StateHoliday_{state_hol}"] = 1
    onehots[f"SchoolHoliday_{school_hol}"] = 1
    onehot_df = pd.DataFrame([onehots])

    # Combine all features
    return pd.concat([num_df, onehot_df], axis=1)[final_cols]

# ── 6) Build full 7-day feature matrix ────────────────────────────────────────
X = pd.concat([make_features(d) for d in dates], ignore_index=True)

# ── 7) Predict ────────────────────────────────────────────────────────────────
y_pred = model.predict(X)

# ── 8) Display results ────────────────────────────────────────────────────────
st.subheader(f"📈 Sales Forecast from {start_date}")
chart_df = pd.DataFrame({"Date": dates, "Predicted Sales": y_pred})
chart_df.set_index("Date", inplace=True)
st.line_chart(chart_df)

# ── 9) Insights ───────────────────────────────────────────────────────────────
st.markdown("### 📊 Forecast Insights")
i_max = int(np.argmax(y_pred))
i_min = int(np.argmin(y_pred))
avg = np.mean(y_pred)

st.write(f"- 📌 **Highest** sales: `{y_pred[i_max]:,.0f}` on **{dates[i_max].date()}**")
st.write(f"- 🛑 **Lowest**  sales: `{y_pred[i_min]:,.0f}` on **{dates[i_min].date()}**")
st.write(f"- 📉 **Average** sales over 7 days: `{avg:,.0f}`")
trend = (
    "📈 Increasing" if y_pred[-1] > y_pred[0] else
    "📉 Decreasing" if y_pred[-1] < y_pred[0] else
    "🔁 Flat"
)
st.write(f"- 🔄 **Trend**: {trend}")
