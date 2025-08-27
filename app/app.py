import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# 1. Load dataset
df = pd.read_csv("data/student-mat.csv")  # adjust path if needed

# 2. Use the same 5 features your app collects
X = df[['studytime', 'failures', 'absences', 'G1', 'G2']]
y = df['G3']

# 3. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Save model, scaler, and features list
with open("models/student_model.pkl", "wb") as f:
    pickle.dump((model, scaler, ['studytime', 'failures', 'absences', 'G1', 'G2']), f)

print("✅ Model trained and saved with 5 features at models/student_model.pkl")
