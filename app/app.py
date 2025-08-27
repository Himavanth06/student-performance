import streamlit as st
import pickle
import numpy as np
import os

# ------------------------------
# Load model, scaler, and feature list
# ------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "student_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model, scaler, FEATURES = pickle.load(f)

st.title("📊 Student Performance Prediction App")

st.markdown("Enter the student details below to predict the final grade (G3).")

# ------------------------------
# Collect user inputs
# ------------------------------
studytime = st.number_input("Study Time (1–4)", min_value=1, max_value=4, value=2)
failures = st.number_input("Number of Past Failures", min_value=0, max_value=5, value=0)
absences = st.number_input("Absences", min_value=0, max_value=50, value=5)
G1 = st.number_input("First Period Grade (0–20)", min_value=0, max_value=20, value=10)
G2 = st.number_input("Second Period Grade (0–20)", min_value=0, max_value=20, value=12)

# Put inputs into array in SAME order as training
input_data = [studytime, failures, absences, G1, G2]
features = np.array([input_data])  # shape (1,5)

# ------------------------------
# Debug info (optional)
# ------------------------------
st.write("✅ Input features shape:", features.shape)
st.write("✅ Scaler expects:", scaler.n_features_in_)

# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict Final Grade (G3)"):
    try:
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        st.success(f"🎯 Predicted Final Grade (G3): {prediction[0]:.2f}")
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
