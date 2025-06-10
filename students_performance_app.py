import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os

# File paths
MODEL_PATH = Path("models/best_random_forest_model.pkl")
SCALER_PATH = Path("models/scaler.pkl")

def load_model_and_scaler():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at: {MODEL_PATH}")
        st.stop()
    if not os.path.exists(SCALER_PATH):
        st.error(f"Scaler file not found at: {SCALER_PATH}")
        st.stop()

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def main():
    st.set_page_config(page_title="Student Performance Predictor", layout="centered")
    st.title("🎓 Student Performance Predictor")
    st.markdown("Fill in the student data below to predict their performance category (Passing, Average, or Low).")

    # Input form
    with st.form("prediction_form"):
        age = st.number_input("Age", min_value=10, max_value=25, value=18)
        gender = st.selectbox("Gender", options=["Male", "Female"])
        ethnicity = st.selectbox("Ethnicity", options=["Group A", "Group B", "Group C", "Group D", "Group E"])
        parental_education = st.selectbox("Education Level", ["High School", "Associate", "Bachelor", "Master", "PhD"])
        study_time = st.slider("Weekly Study Time (hours)", 0, 40, 10)
        absences = st.slider("Number of Absences", 0, 50, 5)
        tutoring = st.selectbox("Attends Tutoring?", ["Yes", "No"])
        parental_support = st.selectbox("Parental Support Level", ["Low", "Moderate", "High", "Very High", "Excellent"])
        extracurricular = st.selectbox("In Extracurricular Activities?", ["Yes", "No"])
        sports = st.slider("Weekly Hours in Sports", 0, 10, 2)
        music = st.slider("Weekly Hours in Music", 0, 10, 2)
        volunteering = st.slider("Weekly Hours Volunteering", 0, 10, 1)

        submitted = st.form_submit_button("Predict Performance")

    if submitted:
        gender_encoded = 1 if gender == "Male" else 0
        ethnicity_encoded = {"Group A": 0, "Group B": 1, "Group C": 2, "Group D": 3, "Group E": 4}[ethnicity]
        education_encoded = {"High School": 0, "Associate": 1, "Bachelor": 2, "Master": 3, "PhD": 4}[parental_education]
        tutoring_encoded = 1 if tutoring == "Yes" else 0
        support_encoded = {"Low": 0, "Moderate": 1, "High": 2, "Very High": 3, "Excellent": 4}[parental_support]
        extracurricular_encoded = 1 if extracurricular == "Yes" else 0

        input_data = pd.DataFrame([[
            age, gender_encoded, ethnicity_encoded, education_encoded, study_time,
            absences, tutoring_encoded, support_encoded, extracurricular_encoded,
            sports, music, volunteering
        ]], columns=[
            'Age', 'Gender', 'Ethnicity', 'ParentalEducation', 'StudyTimeWeekly',
            'Absences', 'Tutoring', 'ParentalSupport', 'Extracurricular',
            'Sports', 'Music', 'Volunteering'
        ])

        model, scaler = load_model_and_scaler()
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]

        # Display based on performance category
        if prediction == "Passing":
            st.success("✅ The student is likely to pass the semester.")
        elif prediction == "Average":
            st.warning("⚠️ The student is borderline. Recommend extra attention.")
        else:
            st.error("❌ The student is at high risk of failing. Immediate intervention advised.")

if __name__ == "__main__":
    main()
