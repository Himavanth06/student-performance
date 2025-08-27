import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
import os

# =========================
# Train model if not exists
# =========================
MODEL_PATH = "models/student_model.pkl"

if not os.path.exists(MODEL_PATH):
    # Load dataset
    df = pd.read_csv("data/student_performance.csv")

    # Encode target (Pass=1, Fail=0)
    le = LabelEncoder()
    df['exam_result'] = le.fit_transform(df['exam_result'])

    # Features and target
    X = df[['hours_studied', 'attendance', 'previous_score', 'assignments_submitted']]
    y = df['exam_result']

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Save
    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump((model, scaler, ['hours_studied', 'attendance', 'previous_score', 'assignments_submitted']), f)

# =========================
# Streamlit App
# =========================
st.title("📘 Student Performance Prediction App")
st.write("Enter the student details below to predict whether the student will Pass or Fail.")

# Load trained model
with open(MODEL_PATH, "rb") as f:
    model, scaler, feature_names = pickle.load(f)

# Inputs
hours_studied = st.number_input("📚 Hours Studied", min_value=0, max_value=24, value=5)
attendance = st.number_input("📅 Attendance (%)", min_value=0, max_value=100, value=75)
previous_score = st.number_input("📊 Previous Score", min_value=0, max_value=100, value=60)
assignments_submitted = st.number_input("📝 Assignments Submitted", min_value=0, max_value=10, value=5)

# Prediction
if st.button("🔍 Predict Result"):
    input_data = pd.DataFrame([[hours_studied, attendance, previous_score, assignments_submitted]],
                              columns=feature_names)
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    result = "✅ Pass" if prediction == 1 else "❌ Fail"
    st.subheader(f"Predicted Exam Result: {result}")
