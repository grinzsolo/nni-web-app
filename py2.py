import streamlit as st
import numpy as np
import joblib

# โหลดโมเดลและ scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("NNI Predictor App")

a = st.number_input("Input LC")
b = st.number_input("Input MFR Rx1")
c = st.number_input("Input MFR Rx2")
d = st.number_input("Input MFR Pellet")

if st.button("Predict"):
    X = np.array([[a, b, c, d]])
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)
    st.success(f"Predicted E: {prediction[0]:.2f}")


