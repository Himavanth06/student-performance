Student Performance Prediction App

A Machine Learning-based web app built using Python and Streamlit that predicts whether a student will Pass or Fail based on their study habits and previous academic performance.

About the Project

This project demonstrates the end-to-end machine learning workflow:

1. Data Collection & Preprocessing**:  
   - Uses a CSV dataset (`student_performance.csv`) containing features such as hours studied, attendance, previous score, and assignments submitted.  
   - Converts target labels (`Pass`/`Fail`) to numeric for ML training.

2. Model Training:  
   - Logistic Regression is used as the classification model.  
   - Features are **scaled** using `StandardScaler`.  
   - Model is trained to predict the final exam result.

3. Model Saving & Deployment:  
   - The trained model and scaler are saved using `pickle` (`student_model.pkl`).  
   - The app loads the model dynamically and makes predictions for new inputs.

4. End-to-End ML Pipeline**:
   - Input → Preprocessing → Prediction → Output**
   - This demonstrates a complete ML workflow from data to real-time prediction.

Features

- Predict if a student will Pass or Fail.
- User-friendly Streamlit interface.
- **Automatic scaling of input features for accurate predictions.
- Lightweight and fast, suitable for deployment online.



Live Demo

Try the app online here: [Student Performance App](https://student-performance-8ymrepyapky4nddurkc6dr.streamlit.app/)



 Technologies Used

- Python 
- Pandas & NumPy (Data manipulation)  
- Scikit-learn (Machine Learning & preprocessing)  
- Streamlit (Web app deployment)  
- Pickle (Model serialization)


