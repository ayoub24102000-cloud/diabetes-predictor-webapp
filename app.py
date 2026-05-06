import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Diabetes Predictor", layout="centered")
st.title("🩺 Diabetes Risk Prediction")
st.markdown("Enter patient clinical measurements. The model was trained using **TOA-balanced Random Forest** (best performing).")

@st.cache_resource
def load_models():
    models = {}
    base_path = "models"
    # تحميل النماذج بصيغة .joblib
    model_files = {
        "rf": os.path.join(base_path, "rf_toa_model.joblib"),
        "lr": os.path.join(base_path, "lr_toa_model.joblib"),
        "xgb": os.path.join(base_path, "xgb_toa_model.joblib"),
        "svm": os.path.join(base_path, "svm_toa_model.joblib")
    }
    for key, path in model_files.items():
        if os.path.exists(path):
            models[key] = joblib.load(path)
    # تحميل المقياس
    scaler_path = "scaler.joblib"
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    return models, scaler

models, scaler = load_models()

if not models:
    st.error("Models not found. Please run the training script first.")
    st.stop()

feature_names = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

st.subheader("Patient Data")
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
    glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=150, value=70)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin (mu U/ml)", min_value=0, max_value=1000, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, step=0.01)
    age = st.number_input("Age (years)", min_value=0, max_value=120, value=30)

input_values = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]

model_choice = st.selectbox(
    "Select Model",
    ["Random Forest (TOA) - Recommended", "Logistic Regression", "XGBoost", "SVM"]
)
model_key = {
    "Random Forest (TOA) - Recommended": "rf",
    "Logistic Regression": "lr",
    "XGBoost": "xgb",
    "SVM": "svm"
}[model_choice]

if st.button("Predict"):
    X_input = pd.DataFrame([input_values], columns=feature_names)
    if scaler is not None:
        X_input_scaled = scaler.transform(X_input)
    else:
        st.error("Scaler not found. Please check the installation.")
        st.stop()

    model = models.get(model_key)
    if model is None:
        st.error(f"Model {model_key} not loaded. Check models folder.")
        st.stop()

    prediction = model.predict(X_input_scaled)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_input_scaled)[0][1]

    result = "Diabetic" if prediction == 1 else "Non‑Diabetic"
    color = "red" if prediction == 1 else "green"
    st.subheader(f"Prediction: **:{color}[{result}]**")
    if proba is not None:
        st.write(f"Confidence (probability of diabetes): **{proba:.2%}**")

    st.info("This prediction is based on the TOA‑balanced Random Forest model, which achieved 81.5% precision and 77.6% recall on the PIMA dataset.")