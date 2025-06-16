import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ── 1) FIRST Streamlit command ─────────────────────────────────────────────────
st.set_page_config(page_title="📈 7‑Day Rossmann Forecast", layout="wide")

# ── 2) Load model & store metadata ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

@st.cache_data
def load_store_data():
    return pd.read_csv("store.csv")

model = load_model()
store_df = load_store_data()

# ── 3) Feature builder ──────────────────────────────────────────────────────────
def make_features(store_id, date, promo, school_hol, state_hol):
    # Pull store metadata
    s = store_df.loc[store_df["Store"] == store_id].iloc[0]
    # Compute promo2 months
    if s.Promo2 == 1 and not pd.isna(s.Promo2SinceYear):
        promo2_since = pd.Timestamp(year=int(s.Promo2SinceYear),
                                    month=int(pd.to_datetime(f'{s.Promo2SinceWeek}-1', format='%W-%w').month),
                                    day=1)
        promo2_months = max(0, (date.year - promo2_since.year) * 12 + (date.month - promo2_since.month))
        promo_in_month = int(date.strftime("%b") in str(s.PromoInterval).split(","))
    else:
        promo2_months = 0
        promo_in_month = 0

    # Months since competition
    if not pd.isna(s.CompetitionOpenSinceYear):
        comp_since = pd.Timestamp(year=int(s.CompetitionOpenSinceYear),
                                  month=int(s.CompetitionOpenSinceMonth),
                                  day=1)
        comp_months = max(0, (date.year - comp_since.year) * 12 + (date.month - comp_since.month))
    else:
        comp_months = 0

    # Base numeric features
    d = {
        "Store": store_id,
        "DayOfWeek": date.weekday() + 1,
        "Day": date.day,
        "Month": date.month,
        "CompetitionDistance": s.CompetitionDistance,
        "CompetitionMonths": comp_months,
        "Promo": promo,
        "Promo2": s.Promo2,
        "Promo2Months": promo2_months,
        "PromoInMonth": promo_in_month,
    }

    # One‑hot templates
    for col in (
        [f"StoreType_{x}" for x in ["a","b","c","d"]] +
        [f"Assortment_{x}" for x in ["a","b","c"]] +
        [f"StateHoliday_{x}" for x in ["0","a","b","c"]] +
        [f"SchoolHoliday_{x}" for x in [0,1]]
    ):
        d[col] = 0

    # Fill one‑hots
    d[f"StoreType_{s.StoreType}"]       = 1
    d[f"Assortment_{s.Assortment}"]     = 1
    d[f"StateHoliday_{state_hol}"]      = 1
    d[f"SchoolHoliday_{school_hol}"]    = 1

    return pd.DataFrame([d])

# ── 4) Sidebar inputs ───────────────────────────────────────────────────────────
st.sidebar.header("Store & Promo Info")
store_id = st.sidebar.number_input("Store ID", min_value=1, max_value=1115, value=1, step=1)
promo    = st.sidebar.selectbox("Is there a Promo today? (0 = No, 1 = Yes)", [0,1], index=1)

st.sidebar.header("Forecast Settings")
start_date = st.sidebar.date_input("Start Date", pd.Timestamp.today().date())

# Build the next 7 days
dates = pd.date_range(start_date, periods=7, freq="D")
date_strs = [d.strftime("%Y-%m-%d") for d in dates]

school_hols = st.sidebar.multiselect(
    "Select which dates are School Holidays",
    options=date_strs
)
state_hols = st.sidebar.multiselect(
    "Select which dates are State Holidays",
    options=date_strs
)

# ── 5) Run Forecast ─────────────────────────────────────────────────────────────
if st.sidebar.button("🔮 Run 7‑Day Forecast"):
    # Build feature matrix
    feats = []
    for d in dates:
        feats.append(make_features(
            store_id=store_id,
            date=d,
            promo=promo,
            school_hol=1 if d.strftime("%Y-%m-%d") in school_hols else 0,
            state_hol="b" if d.strftime("%Y-%m-%d") in state_hols else "0"
        ))
    X = pd.concat(feats, ignore_index=True)

    # Predict
    preds = model.predict(X)

    # Plot
    st.subheader(f"📈 7‑Day Sales Forecast from {start_date}")
    chart_df = pd.DataFrame({"Date": dates, "Predicted Sales": preds}).set_index("Date")
    st.line_chart(chart_df)

    # Insights
    st.markdown("### 🔍 Insights")
    max_i = np.argmax(preds)
    min_i = np.argmin(preds)
    st.write(f"- **Highest** predicted sales: {preds[max_i]:,.0f} on **{dates[max_i].date()}**")
    st.write(f"- **Lowest**  predicted sales: {preds[min_i]:,.0f} on **{dates[min_i].date()}**")
    st.write(f"- **Average** predicted sales: {preds.mean():,.0f}")

    if preds[-1] > preds[0]:
        st.write("➡️ Overall trend: **Increasing** sales over the week.")
    elif preds[-1] < preds[0]:
        st.write("⬇️ Overall trend: **Decreasing** sales over the week.")
    else:
        st.write("🔁 Overall trend: **Flat** sales over the week.")
else:
    st.info("👈 Set your parameters and click **Run 7‑Day Forecast**")

# ── 6) Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & XGBoost — features mirror your training pipeline.")
