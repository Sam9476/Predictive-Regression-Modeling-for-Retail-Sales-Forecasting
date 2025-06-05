import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────────────────────────
# 1.  Load the trained model (make sure model.pkl sits in the same folder as app.py)
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="📈 Sales Forecast Dashboard", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# ────────────────────────────────────────────────────────────────────────────────
# 2.  Define a function to build a single-row DataFrame with the exact 23 features
#     (including one-hot encoding) that your XGBoost model expects.
# ────────────────────────────────────────────────────────────────────────────────
def build_feature_vector(
    store,
    store_type,
    assortment,
    day_of_week,
    day,
    month,
    competition_distance,
    competition_months,
    promo,
    promo2,
    promo2_months,
    promo_in_month,
    state_holiday,
    school_holiday,
):
    # 2.1. Start with all numeric features
    numeric_feats = {
        "Store": store,
        "DayOfWeek": day_of_week,
        "Day": day,
        "Month": month,
        "CompetitionDistance": competition_distance,
        "CompetitionMonths": competition_months,
        "Promo": promo,
        "Promo2": promo2,
        "Promo2Months": promo2_months,
        "PromoInMonth": promo_in_month,
    }

    # 2.2. Prepare one-hot columns for categorical variables; initialize all to 0
    #      Based on your training pipeline, the full list of one-hot columns is:
    dummy_columns = [
        # StoreType
        "StoreType_a",
        "StoreType_b",
        "StoreType_c",
        "StoreType_d",
        # Assortment
        "Assortment_a",
        "Assortment_b",
        "Assortment_c",
        # StateHoliday
        "StateHoliday_0",
        "StateHoliday_a",
        "StateHoliday_b",
        "StateHoliday_c",
        # SchoolHoliday
        "SchoolHoliday_0",
        "SchoolHoliday_1",
    ]
    categorical_feats = {col: 0 for col in dummy_columns}

    # 2.3. Turn on exactly the one-hot that matches the user’s selection
    # StoreType can be 'a', 'b', 'c', or 'd'
    categorical_feats[f"StoreType_{store_type}"] = 1

    # Assortment can be 'a', 'b', or 'c'
    categorical_feats[f"Assortment_{assortment}"] = 1

    # StateHoliday: either '0', 'a', 'b', or 'c'
    categorical_feats[f"StateHoliday_{state_holiday}"] = 1

    # SchoolHoliday: either 0 or 1
    categorical_feats[f"SchoolHoliday_{school_holiday}"] = 1

    # 2.4. Combine numeric + categorical into one dictionary, then DataFrame
    all_feats = {**numeric_feats, **categorical_feats}
    df = pd.DataFrame([all_feats])
    return df


# ────────────────────────────────────────────────────────────────────────────────
# 3.  Streamlit layout & sidebar inputs
# ────────────────────────────────────────────────────────────────────────────────

st.title("📊 Rossmann Sales Forecast Dashboard")
st.markdown(
    """
    This interactive dashboard lets you **forecast daily sales** for a specific store and date‐related configuration.  
    Simply select each parameter in the sidebar, then click **Predict** to see the forecasted sales and a quick bar chart.  
    """
)

st.sidebar.header("Choose input parameters")

# 3.1. Store ID (1–1115)
store = st.sidebar.number_input(
    "Store ID",
    min_value=1,
    max_value=1115,
    value=1,
    step=1,
    help="Store number (1 to 1115).",
)

# 3.2. StoreType: a, b, c, or d
store_type = st.sidebar.selectbox(
    "Store Type",
    options=["a", "b", "c", "d"],
    index=0,
    help="Select store type (a, b, c, or d).",
)

# 3.3. Assortment: a (basic), b (extra), c (extended)
assortment = st.sidebar.selectbox(
    "Assortment",
    options=["a", "b", "c"],
    index=0,
    help="Select assortment category (a: basic, b: extra, c: extended).",
)

# 3.4. Day of Week (1=Monday … 7=Sunday)
day_of_week = st.sidebar.slider(
    "Day of Week",
    min_value=1,
    max_value=7,
    value=3,
    help="Day of week (1=Monday … 7=Sunday).",
)

# 3.5. Day of Month (1–31)
day = st.sidebar.slider(
    "Day of Month",
    min_value=1,
    max_value=31,
    value=15,
    help="Day of month (1 to 31).",
)

# 3.6. Month (1–12)
month = st.sidebar.slider(
    "Month",
    min_value=1,
    max_value=12,
    value=6,
    help="Month (1=January … 12=December).",
)

# 3.7. Competition Distance (meters)
competition_distance = st.sidebar.number_input(
    "Competition Distance (meters)",
    min_value=0.0,
    max_value=1e5,
    value=5000.0,
    step=100.0,
    help="Distance to nearest competitor (in meters).",
)

# 3.8. Competition Months (how many months since competition opened)
competition_months = st.sidebar.number_input(
    "Competition Months",
    min_value=0,
    max_value=2000,
    value=50,
    step=1,
    help="Months since competition opened (e.g., if competition opened 2 years ago, enter 24).",
)

# 3.9. Promo (0 = no promo, 1 = promo)
promo = st.sidebar.selectbox(
    "Promo (Y/N)",
    options=[0, 1],
    index=1,
    help="Is there a regular promo running? (0 = No, 1 = Yes).",
)

# 3.10. Promo2 (0 = no extended promo, 1 = extended promo)
promo2 = st.sidebar.selectbox(
    "Promo2 (Y/N)",
    options=[0, 1],
    index=0,
    help="Is there an extended promo (Promo2)? (0 = No, 1 = Yes).",
)

# 3.11. Promo2 Months (how many months since Promo2 started)
promo2_months = st.sidebar.number_input(
    "Promo2 Months",
    min_value=0,
    max_value=200,
    value=0,
    step=1,
    help="If Promo2 is active, months since Promo2 started. If no Promo2, leave 0.",
)

# 3.12. Promo In Month (0 = promo interval mismatch, 1 = promo interval match)
promo_in_month = st.sidebar.selectbox(
    "Promo In Month (0/1)",
    options=[0, 1],
    index=0,
    help="Does the current month fall inside the PromoInterval? (0 = No, 1 = Yes).",
)

# 3.13. State Holiday: '0' (none), 'a', 'b', or 'c'
state_holiday = st.sidebar.selectbox(
    "State Holiday",
    options=["0", "a", "b", "c"],
    index=0,
    help="Select state holiday code (0 = none, a/b/c = different holiday types).",
)

# 3.14. School Holiday: 0 or 1
school_holiday = st.sidebar.selectbox(
    "School Holiday",
    options=[0, 1],
    index=0,
    help="Is it a school holiday? (0 = No, 1 = Yes).",
)

# ────────────────────────────────────────────────────────────────────────────────
# 4.  Predict button
# ────────────────────────────────────────────────────────────────────────────────
if st.sidebar.button("🔮 Predict Sales"):
    # 4.1. Build the feature vector
    feature_df = build_feature_vector(
        store=store,
        store_type=store_type,
        assortment=assortment,
        day_of_week=day_of_week,
        day=day,
        month=month,
        competition_distance=competition_distance,
        competition_months=competition_months,
        promo=promo,
        promo2=promo2,
        promo2_months=promo2_months,
        promo_in_month=promo_in_month,
        state_holiday=state_holiday,
        school_holiday=school_holiday,
    )

    # 4.2. Run the model
    prediction = model.predict(feature_df)[0]

    # 4.3. Display the numeric forecast
    st.subheader("📈 Predicted Sales")
    st.metric(label="Sales (in currency units)", value=f"{prediction:,.0f}")

    # 4.4. Plot a simple bar chart of the predicted value
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["Forecast"], [prediction], color="#4e79a7")
    ax.set_ylabel("Sales")
    ax.set_title("Daily Sales Forecast")
    st.pyplot(fig)

    st.success("✅ Done! Change parameters in the sidebar to re-run prediction.")
else:
    st.info("🔧 Adjust inputs in the sidebar, then click **Predict Sales**.")


# ────────────────────────────────────────────────────────────────────────────────
# 5.  Footer / Attribution
# ────────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Built with ❤️ using Streamlit and XGBoost. Data pipeline and model trained on Rossmann dataset."
)
