import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model, scaler and expected feature list
model = pickle.load(open('loan_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
model_features = pickle.load(open('model_features.pkl', 'rb'))

st.title("🏦 Loan Approval Predictor")

# UI inputs
Gender = st.selectbox("Gender", ["Male", "Female"])
Married = st.selectbox("Married", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ['0', '1', '2', '3+'])
Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
ApplicantIncome = st.number_input("Applicant Income", min_value=0)
CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
LoanAmount = st.number_input("Loan Amount (in actual units, e.g., 120000)", min_value=0)
Loan_Amount_Term = st.number_input("Loan Term (in days)", min_value=0)
Credit_History = st.selectbox("Credit History", [1.0, 0.0])
Property_Area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

# Map categorical values to numeric (same as training)
gender_map = {'Male': 1, 'Female': 0}
married_map = {'Yes': 1, 'No': 0}
education_map = {'Graduate': 1, 'Not Graduate': 0}
self_employed_map = {'Yes': 1, 'No': 0}
property_map = {'Urban': 2, 'Semiurban': 1, 'Rural': 0}
dependents_map = {'0': 0, '1': 1, '2': 2, '3+': 4}

# Input dictionary
input_dict = {
    'Gender': gender_map[Gender],
    'Married': married_map[Married],
    'Dependents': dependents_map[Dependents],
    'Education': education_map[Education],
    'Self_Employed': self_employed_map[Self_Employed],
    'ApplicantIncome': ApplicantIncome,
    'CoapplicantIncome': CoapplicantIncome,
    'LoanAmount': LoanAmount,
    'Loan_Amount_Term': Loan_Amount_Term,
    'Credit_History': Credit_History,
    'Property_Area': property_map[Property_Area]
}

# Convert input to DataFrame
input_df = pd.DataFrame([input_dict])

# Scale numerical columns
numeric_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

# Reorder columns to match training
input_df = input_df[model_features]

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.success("✅ Loan will be Approved!")
    else:
        st.error("❌ Loan will NOT be Approved.")
