import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Function to load data from GitHub
@st.cache_data
def load_data(url):
    """Loads data from a given URL (GitHub raw file link)."""
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"Error loading data from {url}: {e}")
        return None

# Function to load the trained model and scaler
@st.cache_resource
def load_model_and_scaler(model_url, scaler_url):
    """Loads the pre-trained model and scaler from GitHub raw file links."""
    try:
        model = joblib.load(model_url)
        scaler = joblib.load(scaler_url)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        return None, None

# Function to preprocess input data
def preprocess_data(df, scaler):
    """Preprocesses the input DataFrame."""
    # Handle missing values in CompetitionDistance - using the same strategy as in the notebook
    df['CompetitionDistance'].fillna(2*df['CompetitionDistance'].max(), inplace=True)

    # Handle CompetitionOpenSinceMonth and CompetitionOpenSinceYear
    df['CompetitionOpenSinceYear'] = df['CompetitionOpenSinceYear'].fillna(0).astype(int)
    df['CompetitionOpenSinceMonth'] = df['CompetitionOpenSinceMonth'].fillna(0).astype(int)
    df['CompetitionMonths'] = 0
    mask = (df['CompetitionOpenSinceYear'] > 0) & (df['CompetitionOpenSinceMonth'] > 0)
    df.loc[mask, 'CompetitionMonths'] = 12 * (df.loc[mask, 'Year'] - df.loc[mask, 'CompetitionOpenSinceYear']) + (df.loc[mask, 'Month'] - df.loc[mask, 'CompetitionOpenSinceMonth'])
    df.loc[df['CompetitionMonths'] < 0, 'CompetitionMonths'] = 0
    df.drop(['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear'], axis=1, inplace=True)

    # Handle Promo2SinceWeek and Promo2SinceYear
    df.loc[df['Promo2'] == 0, ['Promo2SinceWeek','Promo2SinceYear']] = 0
    df['Promo2Months'] = 0
    mask = (df['Promo2SinceYear'] > 0) & (df['Promo2SinceWeek'] > 0)
    df.loc[mask, 'Promo2Months'] = 12 * (df.loc[mask, 'Year'] - df.loc[mask, 'Promo2SinceYear']) + (4 * (df.loc[mask, 'Month'] - (df.loc[mask, 'Promo2SinceWeek'] / 4.0)))
    df.loc[df['Promo2Months'] < 0, 'Promo2Months'] = 0
    df.drop(['Promo2SinceWeek', 'Promo2SinceYear'], axis=1, inplace=True)

    # Handle PromoInterval and PromoInMonth
    df['PromoInMonth'] = 0
    month_map = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
                7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
    df['MonthName'] = df['Month'].map(month_map)
    mask = ~df['PromoInterval'].isna()
    df.loc[mask, 'PromoInMonth'] = df.loc[mask].apply(
        lambda x: 1 if x['MonthName'] in str(x['PromoInterval']).split(',') else 0,
        axis=1
    )
    df.drop(['MonthName', 'PromoInterval'], axis=1, inplace=True)

    # Select columns *before* one-hot encoding for scaling
    # This list must match the 'numeric_cols' list used when fitting the scaler in your notebook
    numeric_cols_for_scaling = ['Store', 'DayOfWeek', 'Day', 'Month', 'CompetitionDistance',
                                'CompetitionMonths', 'Promo', 'Promo2', 'Promo2Months', 'PromoInMonth']

    # Normalize numeric columns using the loaded scaler *before* one-hot encoding
    # Ensure these columns are in the same order as they were during fitting
    df[numeric_cols_for_scaling] = scaler.transform(df[numeric_cols_for_scaling])

    # Define categorical columns
    categorical_cols = ['StoreType', 'Assortment', 'StateHoliday', 'SchoolHoliday']

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)

    # Ensure all columns from training data are present (add missing ones with 0)
    # This list must exactly match the columns of your X_train DataFrame after all preprocessing steps
    # including scaling and one-hot encoding. Get this list from your notebook after creating X_train.
    expected_cols_after_dummies = [
        'Store', 'DayOfWeek', 'Day', 'Month', 'CompetitionDistance',
        'CompetitionMonths', 'Promo', 'Promo2', 'Promo2Months', 'PromoInMonth',
        'StoreType_a', 'StoreType_b', 'StoreType_c', 'StoreType_d',
        'Assortment_a', 'Assortment_b', 'Assortment_c',
        'StateHoliday_0', 'StateHoliday_a', 'StateHoliday_b', 'StateHoliday_c',
        'SchoolHoliday'
    ]
    for col in expected_cols_after_dummies:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns to match the order used during training
    df = df[expected_cols_after_dummies]


    # Convert boolean to int and float to int for specific columns (these were numeric before scaling)
    # Ensure these columns are now treated as integers after scaling
    float_to_int_cols = ['Promo', 'Promo2', 'PromoInMonth']
    # Need to re-evaluate if these should be int *after* scaling. Scaled values are typically floats.
    # Let's keep them as float for now, as scaling transforms them to floats.
    # If your model expects int, you might need to explicitly cast them after scaling.
    # For XGBoost, float input is usually fine.
    # df[float_to_int_cols] = df[float_to_int_cols].astype(int)


    return df

# Streamlit App
st.title("Rossmann Store Sales Prediction")

st.write("This app predicts daily sales for Rossmann stores using an XGBoost model.")

# Get GitHub raw file URLs from user input or hardcode
github_repo_url = st.text_input("Enter the base URL of your GitHub repository (e.g., https://raw.githubusercontent.com/username/repo_name/main/)", "")

if github_repo_url:
    store_data_url = f"{github_repo_url}/store.csv"
    model_url = f"{github_repo_url}/model.pkl"
    scaler_url = f"{github_repo_url}/scaler.pkl"

    # Load data and model
    store_df = load_data(store_data_url)
    model, scaler = load_model_and_scaler(model_url, scaler_url)

    if store_df is not None and model is not None and scaler is not None:
        st.sidebar.header("Input Features")

        # Get user input for prediction
        store_id = st.sidebar.selectbox("Store ID", store_df['Store'].unique())
        day_of_week = st.sidebar.selectbox("Day of Week", range(1, 8))
        day = st.sidebar.slider("Day of Month", 1, 31)
        month = st.sidebar.slider("Month", 1, 12)
        year = st.sidebar.slider("Year", 2013, 2015) # Assuming prediction for years within the training range

        # Retrieve store-specific features
        store_features = store_df[store_df['Store'] == store_id].iloc[0].to_dict()

        # Create a DataFrame for prediction
        input_data = {
            'Store': store_id,
            'DayOfWeek': day_of_week,
            'Day': day,
            'Month': month,
            'Year': year,
            'Open': 1, # Assuming the store is open for prediction - Note: 'Open' was dropped in notebook
            'Promo': st.sidebar.selectbox("Promo", [0, 1]),
            'StateHoliday': st.sidebar.selectbox("State Holiday", ['0', 'a', 'b', 'c']),
            'SchoolHoliday': st.sidebar.selectbox("School Holiday", [0, 1]),
            'StoreType': store_features['StoreType'],
            'Assortment': store_features['Assortment'],
            'CompetitionDistance': store_features['CompetitionDistance'],
            'CompetitionOpenSinceMonth': store_features['CompetitionOpenSinceMonth'],
            'CompetitionOpenSinceYear': store_features['CompetitionOpenSinceYear'],
            'Promo2': store_features['Promo2'],
            'Promo2SinceWeek': store_features['Promo2SinceWeek'],
            'Promo2SinceYear': store_features['Promo2SinceYear'],
            'PromoInterval': store_features['PromoInterval']
        }

        input_df = pd.DataFrame([input_data])

        # Preprocess the input data
        processed_input_df = preprocess_data(input_df.copy(), scaler)

        # Make prediction
        if st.sidebar.button("Predict Sales"):
            prediction = model.predict(processed_input_df)

            # To get the actual sales value, we need to inverse transform the prediction.
            # The scaler was fitted on the whole dataframe including Sales.
            # We need to create a dummy DataFrame with the predicted sales in the 'Sales' column
            # and other columns with arbitrary values to use the inverse_transform method correctly.
            # This is a workaround; a dedicated scaler for the target variable is recommended for deployment.

            # Create a dummy DataFrame with the correct column structure for inverse transformation
            # This structure should match the DataFrame *before* splitting but *after* all preprocessing
            # steps that were included when the scaler was fitted on the entire dataframe.
            # Based on your notebook, the scaler was fitted on the entire 'df' after feature engineering
            # but before splitting into X and y. The 'Sales' column was the last one.
            original_cols_before_split_and_scale = [
                 'Store', 'StoreType_a', 'StoreType_b', 'StoreType_c', 'StoreType_d', 'Assortment_a', 'Assortment_b', 'Assortment_c',
                'DayOfWeek', 'Day', 'Month',
                'CompetitionDistance', 'CompetitionMonths',
                'Promo', 'Promo2', 'Promo2Months', 'PromoInMonth',
                'StateHoliday_0', 'StateHoliday_a', 'StateHoliday_b', 'StateHoliday_c',
                'SchoolHoliday', 'Sales' # Sales was included here when scaling the whole df
            ]

            dummy_df_for_inverse = processed_input_df.copy()
            # Need to add dummy columns that were present in the original df but not in the input for inverse transform
            for col in original_cols_before_split_and_scale:
                if col not in dummy_df_for_inverse.columns:
                    # Assign an arbitrary value for columns not present in the input,
                    # as these are needed for the structure of inverse_transform
                    # For dummy variables, 0 is appropriate. For others, a mean or median might be needed
                    # but since the scaler was applied to the whole df, any value will be transformed.
                    # A more robust solution would be a separate scaler for the target.
                    dummy_df_for_inverse[col] = 0 # Or a more appropriate default/mean

            # Add the predicted sales to the 'Sales' column
            dummy_df_for_inverse['Sales'] = prediction

            # Reorder columns to match the original DataFrame structure before scaling
            dummy_df_for_inverse = dummy_df_for_inverse[original_cols_before_split_and_scale]


            # Inverse transform the entire dummy DataFrame.
            # The inverse transformed 'Sales' column is at the index corresponding to 'Sales'.
            sales_col_index = original_cols_before_split_and_scale.index('Sales')
            actual_prediction = scaler.inverse_transform(dummy_df_for_inverse)[:, sales_col_index]


            st.subheader("Predicted Sales:")
            # Ensure the prediction is not negative
            st.write(f"${max(0, actual_prediction[0]):,.2f}")

    else:
        st.warning("Please enter a valid GitHub repository URL and ensure files are accessible.")
