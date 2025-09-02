# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# --- 1. Load the Saved Model and Scaler ---

# Define file paths
model_filename = 'models/random_forest_fraud_detector.joblib'
scaler_filename = 'models/scaler.joblib'

# Load the trained model and the scaler
try:
    model = joblib.load(model_filename)
    scaler = joblib.load(scaler_filename)
except FileNotFoundError as e:
    st.error(f"Error: A required file was not found. Please make sure both '{model_filename}' and '{scaler_filename}' exist.")
    st.error(f"Details: {e}")
    st.info("Please run the Jupyter Notebook cell that saves the scaler.joblib file.")
    st.stop()


# --- 2. Create the Web App Interface ---

st.set_page_config(layout="wide")
st.title('High-Accuracy Fraud Detection Tool 🎯')

st.write("""
This tool uses the full set of 30 features to make the most accurate prediction.
Adjust the sliders and inputs below to simulate a transaction.
""")


# --- 3. Create Input Fields for All Features ---

st.header('Transaction Details')

# Create columns for a cleaner layout
col1, col2, col3 = st.columns(3)

with col1:
    time = st.number_input('Time (seconds since first transaction)', value=0.0, format="%.2f")
    amount = st.number_input('Amount ($)', value=100.0, format="%.2f")
    st.markdown("---")
    v1 = st.slider('V1', -57.0, 3.0, 0.0)
    v2 = st.slider('V2', -73.0, 23.0, 0.0)
    v3 = st.slider('V3', -49.0, 10.0, 0.0)
    v4 = st.slider('V4', -6.0, 17.0, 0.0)
    v5 = st.slider('V5', -114.0, 35.0, 0.0)
    v6 = st.slider('V6', -27.0, 74.0, 0.0)
    v7 = st.slider('V7', -44.0, 121.0, 0.0)
    v8 = st.slider('V8', -74.0, 21.0, 0.0)
    v9 = st.slider('V9', -14.0, 16.0, 0.0)

with col2:
    v10 = st.slider('V10', -25.0, 24.0, 0.0)
    v11 = st.slider('V11', -5.0, 13.0, 0.0)
    v12 = st.slider('V12', -19.0, 8.0, 0.0)
    v13 = st.slider('V13', -6.0, 8.0, 0.0)
    v14 = st.slider('V14', -20.0, 11.0, 0.0)
    v15 = st.slider('V15', -5.0, 9.0, 0.0)
    v16 = st.slider('V16', -15.0, 18.0, 0.0)
    v17 = st.slider('V17', -26.0, 10.0, 0.0)
    v18 = st.slider('V18', -10.0, 6.0, 0.0)
    v19 = st.slider('V19', -8.0, 6.0, 0.0)

with col3:
    v20 = st.slider('V20', -55.0, 40.0, 0.0)
    v21 = st.slider('V21', -35.0, 28.0, 0.0)
    v22 = st.slider('V22', -11.0, 11.0, 0.0)
    v23 = st.slider('V23', -45.0, 23.0, 0.0)
    v24 = st.slider('V24', -3.0, 5.0, 0.0)
    v25 = st.slider('V25', -11.0, 8.0, 0.0)
    v26 = st.slider('V26', -3.0, 4.0, 0.0)
    v27 = st.slider('V27', -23.0, 32.0, 0.0)
    v28 = st.slider('V28', -16.0, 34.0, 0.0)


# --- 4. Prediction Logic ---

if st.button('Predict Fraud Status', type="primary"):
    # Create a DataFrame from the user inputs in the correct order
    feature_names = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                     'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                     'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
    
    user_input = pd.DataFrame([[
        time, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14,
        v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, amount
    ]], columns=feature_names)

    # Use the LOADED scaler to transform the Time and Amount columns
    user_input[['Time', 'Amount']] = scaler.transform(user_input[['Time', 'Amount']])
    
    # Make a prediction
    prediction = model.predict(user_input)
    prediction_proba = model.predict_proba(user_input)

    # --- 5. Display the Result ---
    st.subheader('Prediction Result')
    
    if prediction[0] == 1:
        st.error('This transaction is likely to be FRAUDULENT!', icon="🚨")
        st.write(f"**Confidence:** {prediction_proba[0][1]*100:.2f}%")
    else:
        st.success('This transaction is likely to be LEGITIMATE.', icon="✅")
        st.write(f"**Confidence:** {prediction_proba[0][0]*100:.2f}%")