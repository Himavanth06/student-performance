import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
from pathlib import Path

# Paths
DATA_PATH = "data/student_performance.csv"
MODEL_PATH = Path("models/student_model.pkl")

# 1. Load data
data = pd.read_csv(DATA_PATH)
data['exam_result'] = data['exam_result'].map({'Pass': 1, 'Fail': 0})

X = data[['hours_studied', 'attendance', 'previous_score', 'assignments_submitted']]
y = data['exam_result']

# 2. Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. Save model + scaler
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump((model, scaler), f)

print(f"✅ Model saved at {MODEL_PATH}")
