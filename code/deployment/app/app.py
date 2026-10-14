"""
Stage 3: Web application (Streamlit)
Input fields + button. Sends the data to the API and shows the prediction.
"""
import os

import requests
import streamlit as st

# in docker-compose the API container is reachable by its service name "api"
API_URL = os.getenv("API_URL", "http://api:8000")

st.title("Diabetes Prediction")
st.write("Enter patient data and press the button to get a prediction.")

col1, col2 = st.columns(2)
with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose", min_value=1.0, max_value=300.0, value=120.0)
    blood_pressure = st.number_input("Blood pressure", min_value=1.0, max_value=200.0, value=70.0)
    skin_thickness = st.number_input("Skin thickness", min_value=1.0, max_value=100.0, value=25.0)
with col2:
    insulin = st.number_input("Insulin", min_value=1.0, max_value=900.0, value=100.0)
    bmi = st.number_input("BMI", min_value=1.0, max_value=70.0, value=30.0)
    pedigree = st.number_input("Diabetes pedigree function", min_value=0.0, max_value=3.0, value=0.4)
    age = st.number_input("Age", min_value=1, max_value=120, value=35)

if st.button("Predict"):
    payload = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": pedigree,
        "Age": age,
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()

        if result["prediction"] == 1:
            st.error(f"Diabetes is likely (probability {result['probability']:.0%})")
        else:
            st.success(f"Diabetes is unlikely (probability {result['probability']:.0%})")
    except requests.exceptions.RequestException as e:
        st.warning(f"Could not get a prediction from the API: {e}")
