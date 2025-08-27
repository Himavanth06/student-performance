import os
import pickle
import streamlit as st
import numpy as np

# -------------------------------
# Load trained model and scaler
# -------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "student_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model, scaler = pickle.load(f)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Student Performance Predictor", page_icon="📊")

st.title("📊 Student Performance Prediction App")
st.write("Enter student details below to predict exam performance:")

# Example input fields (you can expand these depending on your dataset)
study_time = st.slider("Weekly Study Time (hours)", 1, 20, 5)
failures = st.slider("Number of Past Class Failures", 0, 5, 0)
absences = st.slider("Number of School Absences", 0, 50, 2)
g1 = st.slider("First Period Grade (0–20)", 0, 20, 10)
g2 = st.slider("Second Period Grade (0–20)", 0, 20, 10)

# Collect input features
features = np.array([[study_time, failures, absences, g1, g2]])

# Scale input (use same scaler as training)
features_scaled = scaler.transform(features)

# Predict
if st.button("🔮 Predict Final Grade"):
    prediction = model.predict(features_scaled)
    st.success(f"Predicted Final Grade (G3): **{int(prediction[0])}** / 20")
