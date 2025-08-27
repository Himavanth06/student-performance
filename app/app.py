import streamlit as st
import pickle
import numpy as np

# Load model + scaler
model, scaler = pickle.load(open("../models/student_model.pkl", "rb"))

st.title("📘 Student Performance Prediction")
st.write("Enter details to predict if a student will Pass or Fail.")

# User Inputs
hours = st.number_input("Hours Studied", min_value=0, max_value=16, value=5)
attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=80)
prev_score = st.number_input("Previous Score (%)", min_value=0, max_value=100, value=70)
assignments = st.number_input("Assignments Submitted", min_value=0, max_value=10, value=7)

if st.button("Predict Result"):
    input_data = scaler.transform([[hours, attendance, prev_score, assignments]])
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        st.success("✅ Student is likely to PASS")
    else:
        st.error("❌ Student is likely to FAIL")
