# pp.py

import streamlit as st
import pandas as pd
import pickle
import joblib

# Load trained model and scaler
with open('logreg_churn_model.pkl', 'rb') as file:
    model = pickle.load(file)

scaler = joblib.load('scaler.pkl')

# Columns used in the trained model
top_features = [
    'TotalCharges','tenure','AvgMonthlySpend','MonthlyCharges',
    'InternetService_Fiber optic','PaymentMethod_Electronic check',
    'Contract_Two year','OnlineSecurity_Yes','gender','TechSupport_Yes',
    'PaperlessBilling','Contract_One year','Partner','SeniorCitizen',
    'OnlineBackup_Yes'
]

st.title("Customer Churn Prediction")
st.write("Enter customer details to predict if they are likely to churn.")

def user_input_features():
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Has Partner?", ["Yes", "No"])
    Dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1)
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=500.0, value=50.0)
    TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=20000.0, value=100.0)
    Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing?", ["Yes", "No"])
    PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    
    AvgMonthlySpend = TotalCharges / (tenure + 1)

    features = {
        'TotalCharges': TotalCharges,
        'tenure': tenure,
        'AvgMonthlySpend': AvgMonthlySpend,
        'MonthlyCharges': MonthlyCharges,
        'InternetService_Fiber optic': 1 if InternetService == 'Fiber optic' else 0,
        'PaymentMethod_Electronic check': 1 if PaymentMethod == 'Electronic check' else 0,
        'Contract_Two year': 1 if Contract == 'Two year' else 0,
        'OnlineSecurity_Yes': 1 if OnlineSecurity == 'Yes' else 0,
        'gender': 1 if gender == 'Male' else 0,
        'TechSupport_Yes': 1 if TechSupport == 'Yes' else 0,
        'PaperlessBilling': 1 if PaperlessBilling == 'Yes' else 0,
        'Contract_One year': 1 if Contract == 'One year' else 0,
        'Partner': 1 if Partner == 'Yes' else 0,
        'SeniorCitizen': SeniorCitizen,
        'OnlineBackup_Yes': 1 if OnlineBackup == 'Yes' else 0
    }

    # Ensure all top features exist
    for col in top_features:
        if col not in features:
            features[col] = 0

    input_df = pd.DataFrame(features, index=[0])

    numeric_features = ['tenure','MonthlyCharges','TotalCharges','AvgMonthlySpend']
    input_df[numeric_features] = scaler.transform(input_df[numeric_features])

    return input_df[top_features]

input_df = user_input_features()

if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    prediction_prob = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"The customer is likely to churn. Probability: {prediction_prob:.2f}")
    else:
        st.success(f"The customer is unlikely to churn. Probability: {prediction_prob:.2f}")
