# Customer-Churn-Predictio

# The app is deployed link -> (https://brotech-customer-churn.streamlit.app/)

Churn Prediction – End-to-End Data Science Project

1. Overview
This project focuses on building an end-to-end Customer Churn Prediction system.
The workflow includes data cleaning, exploratory data analysis, feature engineering, model training, hyperparameter tuning, evaluation, and deployment using Streamlit.
The final model uses Logistic Regression, selected based on the highest ROC-AUC score.

2. Dataset
The dataset contains customer information from a telecom company:
Demographics (gender, senior citizen, partner)
Service usage details (Internet, Streaming services, Tech support)
Contract and billing information
Monthly and total charges
Target variable: Churn

3. Project Workflow

3.1 Data Cleaning
Handled missing values in TotalCharges.
Converted categorical variables into numerical form.
Created new derived features such as AvgMonthlySpend.

3.2 Exploratory Data Analysis
Univariate and bivariate analysis of churn patterns.
Understanding relation between contract type, billing, tech support, and churn.

3.3 Feature Engineering
Selected important numerical and categorical features.
Applied One-Hot Encoding on categorical features.
Standardized numerical features.

3.4 Feature Selection
Identified top features contributing to churn prediction.

3.5 Model Training and Selection

Evaluated multiple models:
Logistic Regression
Random Forest
XGBoost

Metrics used:
Accuracy
Precision
Recall
F1-Score
ROC-AUC (preferred metric)

Logistic Regression delivered the best balance, especially the ROC-AUC score.

4. Hyperparameter Tuning
Performed GridSearchCV on Logistic Regression.

Best parameters:
C = 10
solver = "saga"
penalty = "l2"
max_iter = 1000

Final model performance:
Accuracy: ~79%
ROC-AUC: ~83%
These results are strong for real-world churn problems.

5. Model Deployment (Streamlit)

Deployment steps:
Saved scaler, and trained model using pickle.
Created a full Streamlit interface for user inputs.
Displayed churn prediction with probability.
The app allows users to enter customer details and receive a churn prediction instantly.

6. Files in This Project
Customer Churn Prediction.ipynb – Full EDA, preprocessing, feature engineering, and model building.
app.py – Streamlit prediction interface.
model.pkl – Trained Logistic Regression model.
scaler.pkl – StandardScaler used during training.
requirements.txt – Dependencies required for running the app.

7. How to Run the Streamlit App
Install dependencies:
pip install -r requirements.txt

Run the app:
streamlit run pp.py

Enter customer details and view the churn prediction.

8. Conclusion
This project demonstrates a full production-ready data science pipeline:

Data cleaning
EDA
Feature engineering
Model building
Hyperparameter tuning
Deployment

The final deployed solution helps businesses identify customers likely to churn and enables proactive retention strategies.
