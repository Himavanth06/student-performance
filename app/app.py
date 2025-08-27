import streamlit as st
import numpy as np
import pickle
import os

# Load model and scaler
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "student_model.pkl")
with open(MODEL_PATH, "rb") as f:
    model, scaler = pickle.load(f)

st.title("🎓 Student Performance Prediction App")
st.write("Enter the student details below to predict the final grade (G3).")

# Inputs
studytime = st.number_input("Study Time (1–4)", min_value=1, max_value=4, value=2)
failures = st.number_input("Number of Past Failures", min_value=0, max_value=4, value=0)
absences = st.number_input("Absences", min_value=0, max_value=100, value=5)
g1 = st.number_input("First Period Grade (0–20)", min_value=0, max_value=20, value=10)

# Removed G2 to match scaler (only 4 features expected)

if st.button("Predict Final Grade"):
    # Prepare features
    features = np.array([[studytime, failures, absences, g1]])
    features_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(features_scaled)[0]
    st.success(f"Predicted Final Grade (G3): {round(prediction, 2)}")
