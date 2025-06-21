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

    # Select and reorder columns to match training data
    cols = [
        'Store', 'DayOfWeek', 'Day', 'Month', 'CompetitionDistance',
        'CompetitionMonths', 'Promo', 'Promo2', 'Promo2Months', 'PromoInMonth',
        'StoreType', 'Assortment', 'StateHoliday', 'SchoolHoliday'
    ]
    df = df[cols]

    # One-hot encode categorical columns - make sure to handle potential missing categories
    categorical_cols = ['StoreType', 'Assortment', 'StateHoliday', 'SchoolHoliday']
    df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)

    # Ensure all columns from training data are present (add missing ones with 0)
    # You would ideally save the list of columns from your training data
    # For this example, we'll assume the following dummy columns exist based on your notebook
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


    # Convert boolean to int and float to int for specific columns
    bool_columns = df.select_dtypes(include=['bool']).columns
    df[bool_columns] = df[bool_columns].astype(int)
    float_to_int_cols = ['Promo', 'Promo2', 'PromoInMonth']
    df[float_to_int_cols] = df[float_to_int_cols].astype(int)

    # Normalize numeric columns - need to apply the *same* scaler fitted on the training data
    numeric_cols = ['Store', 'DayOfWeek', 'Day', 'Month', 'CompetitionDistance',
                'CompetitionMonths', 'Promo', 'Promo2', 'Promo2Months', 'PromoInMonth'] # Exclude dummy columns and Sales
    df[numeric_cols] = scaler.transform(df[numeric_cols])


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
            'Open': 1, # Assuming the store is open for prediction
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

        # Ensure column order and presence match the model's expected input
        # This is a crucial step to avoid errors during prediction.
        # You need the exact list of columns that the model was trained on after preprocessing and one-hot encoding.
        # In your notebook, this would be the columns of your 'X' DataFrame.
        # For now, we'll assume 'expected_cols_after_dummies' from the preprocess_data function
        expected_model_cols = [
            'Store', 'DayOfWeek', 'Day', 'Month', 'CompetitionDistance',
            'CompetitionMonths', 'Promo', 'Promo2', 'Promo2Months', 'PromoInMonth',
            'StoreType_a', 'StoreType_b', 'StoreType_c', 'StoreType_d',
            'Assortment_a', 'Assortment_b', 'Assortment_c',
            'StateHoliday_0', 'StateHoliday_a', 'StateHoliday_b', 'StateHoliday_c',
            'SchoolHoliday'
        ] # This list must exactly match the order and names of columns in your training data X

        # Add missing columns to processed_input_df with value 0
        for col in expected_model_cols:
            if col not in processed_input_df.columns:
                processed_input_df[col] = 0

        # Reorder the columns of processed_input_df
        processed_input_df = processed_input_df[expected_model_cols]


        # Make prediction
        if st.sidebar.button("Predict Sales"):
            prediction = model.predict(processed_input_df)
            # The scaler was applied to the target variable (Sales) as well in your notebook.
            # To get the actual sales value, we need to inverse transform the prediction.
            # The scaler was fitted on the whole dataframe including Sales.
            # We need to create a dummy DataFrame with the predicted sales in the 'Sales' column
            # and other columns with arbitrary values to use the inverse_transform method correctly.
            # A more robust approach would be to save the scaler that was *only* fitted on the 'Sales' column.
            # Given the current notebook, we'll recreate a structure similar to the DataFrame before splitting.
            # This is a workaround; a dedicated scaler for the target variable is recommended for deployment.

            # Create a dummy DataFrame with the correct column structure for inverse transformation
            dummy_df_for_inverse = processed_input_df.copy() # Use the processed input features
            dummy_df_for_inverse['Sales'] = prediction # Add the predicted sales
            # Reorder columns to match the original DataFrame structure before scaling
            original_cols_order = [
                 'Store', 'StoreType_a', 'StoreType_b', 'StoreType_c', 'StoreType_d', 'Assortment_a', 'Assortment_b', 'Assortment_c',
                'DayOfWeek', 'Day', 'Month',
                'CompetitionDistance', 'CompetitionMonths',
                'Promo', 'Promo2', 'Promo2Months', 'PromoInMonth',
                'StateHoliday_0', 'StateHoliday_a', 'StateHoliday_b', 'StateHoliday_c',
                'SchoolHoliday', 'Sales' # Include Sales at the end as it was in your original scaled df
            ]
            # Ensure all original columns are present before reordering
            for col in original_cols_order:
                if col not in dummy_df_for_inverse.columns:
                    dummy_df_for_inverse[col] = 0

            dummy_df_for_inverse = dummy_df_for_inverse[original_cols_order]

            # Inverse transform the entire dummy DataFrame. The inverse transformed 'Sales' column is the actual prediction.
            actual_prediction = scaler.inverse_transform(dummy_df_for_inverse)[:, original_cols_order.index('Sales')]


            st.subheader("Predicted Sales:")
            st.write(f"${actual_prediction[0]:,.2f}")

    else:
        st.warning("Please enter a valid GitHub repository URL.")
