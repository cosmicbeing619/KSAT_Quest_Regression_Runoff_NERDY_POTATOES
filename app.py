import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Ksat Predictor", layout="centered")
st.title("🌱 Soil Saturated Hydraulic Conductivity Predictor")

model_path = "ksat_model.joblib"

# List of features user provides
user_features = [
    "bulkdensity", "clay(%)", "medium",
    "organiccarbon", "sand(%)", "silt(%)", "veryfine"
]

# Load model
if os.path.exists(model_path):
    model = joblib.load(model_path)

    st.subheader("Enter soil characteristics below:")
    with st.form("prediction_form"):
        inputs = {}
        for feature in user_features:
            inputs[feature] = st.number_input(feature, value=0.0, format="%.4f")
        submit = st.form_submit_button("Predict")

    if submit:
        try:
            # Build a row of zeros for all model features
            input_data = {feature: 0 for feature in model.booster_.feature_name()}

            # Fill in user-provided values
            for feature in user_features:
                if feature in input_data:
                    input_data[feature] = inputs[feature]

            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])

            # Predict
            prediction = model.predict(input_df)[0]
            st.success(f"🌊 Predicted Ksat: {prediction:.6f} cm/hr")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
else:
    st.warning("⚠️ Trained model not found. Please ensure 'ksat_model.joblib' is in the project folder.")