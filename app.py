import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# 1) Page config
st.set_page_config(
    page_title="Rossmann 7-Day Sales Forecast",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2) Load artifacts
@st.cache_resource
def load_model(): return joblib.load("model.pkl")
@st.cache_resource
def load_scaler(): return joblib.load("scaler.pkl")
@st.cache_data
def load_store_data(): return pd.read_csv("store.csv")

model    = load_model()
scaler   = load_scaler()
store_df = load_store_data()

# 3) Feature columns
numeric_cols = [
    "Store","DayOfWeek","Day","Month",
    "CompetitionDistance","CompetitionMonths",
    "Promo","Promo2","Promo2Months","PromoInMonth"
]
onehot_cols = [
    "StoreType_a","StoreType_b","StoreType_c","StoreType_d",
    "Assortment_a","Assortment_b","Assortment_c",
    "StateHoliday_0","StateHoliday_a","StateHoliday_b","StateHoliday_c",
    "SchoolHoliday_0","SchoolHoliday_1"
]
final_cols = numeric_cols + onehot_cols

# 4) Sidebar inputs
st.sidebar.header("Store & Promotion Settings")
store_id   = st.sidebar.number_input("Store ID", 1, 1115, 1)
promo_flag = st.sidebar.selectbox("Promo Today?", [0,1], index=1,
                                  format_func=lambda x: "Yes" if x else "No")

st.sidebar.header("Forecast Window")
start_date = st.sidebar.date_input("Start Date", pd.Timestamp.today().date())
dates = pd.date_range(start_date, periods=7, freq="D")
date_strs = [d.strftime("%Y-%m-%d") for d in dates]

school_hols = st.sidebar.multiselect("School Holiday Dates", options=date_strs)
state_hols  = st.sidebar.multiselect("State Holiday Dates",  options=date_strs)

if not st.sidebar.button("🔮 Run Forecast"):
    st.sidebar.write("Configure inputs and click **Run Forecast**")
    st.stop()

# 5) Feature builder
def make_features(date):
    s = store_df.loc[store_df.Store==store_id].iloc[0]

    # CompetitionMonths
    if pd.notna(s.CompetitionOpenSinceYear) and pd.notna(s.CompetitionOpenSinceMonth):
        comp_start = pd.Timestamp(int(s.CompetitionOpenSinceYear),
                                  int(s.CompetitionOpenSinceMonth),1)
        comp_months = max(0, (date.year-comp_start.year)*12 + (date.month-comp_start.month))
    else:
        comp_months = 0

    # Promo2Months & PromoInMonth
    if s.Promo2==1 and pd.notna(s.Promo2SinceYear) and pd.notna(s.Promo2SinceWeek):
        p2_start = pd.Timestamp.fromisocalendar(int(s.Promo2SinceYear),
                                                 int(s.Promo2SinceWeek),1)
        p2_months = max(0, (date.year-p2_start.year)*12 + (date.month-p2_start.month))
        intervals = str(s.PromoInterval).split(",") if pd.notna(s.PromoInterval) else []
        promo_in_month = int(date.strftime("%b") in intervals)
    else:
        p2_months = 0
        promo_in_month = 0

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
        "PromoInMonth": promo_in_month
    }

    # Scale numeric
    arr = np.array([raw[c] for c in numeric_cols]).reshape(1,-1)
    scaled = scaler.transform(arr)
    num_df = pd.DataFrame(scaled, columns=numeric_cols)

    # One-hot encoding
    onehots = dict.fromkeys(onehot_cols, 0)
    onehots[f"StoreType_{s.StoreType}"] = 1
    onehots[f"Assortment_{s.Assortment}"] = 1
    state_flag  = "b" if date.strftime("%Y-%m-%d") in state_hols else "0"
    school_flag = 1   if date.strftime("%Y-%m-%d") in school_hols else 0
    onehots[f"StateHoliday_{state_flag}"] = 1
    onehots[f"SchoolHoliday_{school_flag}"] = 1
    onehot_df = pd.DataFrame([onehots])

    return pd.concat([num_df, onehot_df], axis=1)[final_cols]

# 6) Build & predict
X      = pd.concat([make_features(d) for d in dates], ignore_index=True)
y_pred = model.predict(X)

# 7) Transparent Matplotlib line chart
st.subheader(f"Sales Forecast from {start_date.strftime('%d-%m-%Y')} to {dates[-1].strftime('%d-%m-%Y')}")
fig, ax = plt.subplots(figsize=(8,4))
# make backgrounds transparent
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)
# plot line
ax.plot(dates, y_pred, marker="o", linewidth=2, color="cyan")
ax.set_title("7-Day Sales Forecast", color="white", fontsize=14)
ax.set_xlabel("Date", color="white", fontsize=12)
ax.set_ylabel("Predicted Sales ($)", color="white", fontsize=12)
# ticks & labels color
ax.tick_params(axis="x", colors="white", rotation=45)
ax.tick_params(axis="y", colors="white")
ax.xaxis.set_major_formatter(DateFormatter("%d-%m-%Y"))
ax.grid(alpha=0.3, color="gray")
st.pyplot(fig, clear_figure=True)

# 8) Insights
st.markdown("### 🔍 Forecast Insights")
i_max = int(np.argmax(y_pred))
i_min = int(np.argmin(y_pred))
avg   = float(np.mean(y_pred))
total = float(np.sum(y_pred))

st.write(f"- **Highest forecast:** {y_pred[i_max]:,.0f} on {dates[i_max].strftime('%d-%m-%Y')}")
st.write(f"- **Lowest forecast:** {y_pred[i_min]:,.0f} on {dates[i_min].strftime('%d-%m-%Y')}")
st.write(f"- **Average forecast over 7 days:** {avg:,.0f}")
st.write(f"- **Overall Trend:** {'📈 Increasing' if y_pred[-1]>y_pred[0] else '📉 Decreasing' if y_pred[-1]<y_pred[0] else '🔁 Flat'}")

# Individual Day Forecasts
st.markdown("#### Individual Day Forecasts")
df_out = pd.DataFrame({
    "Date (DD-MM-YYYY)": [d.strftime("%d-%m-%Y") for d in dates],
    "Predicted Sales": y_pred.astype(int)
})
st.table(df_out)
