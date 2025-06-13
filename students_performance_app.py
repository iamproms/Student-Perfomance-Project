import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# File paths
MODEL_PATH = Path("models/best_random_forest_model.pkl")
SCALER_PATH = Path("models/scaler.pkl")

# Initialize session state
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'show_insights' not in st.session_state:
    st.session_state.show_insights = False
if 'submitted' not in st.session_state:
    st.session_state.submitted = False

# Cache model and scaler loading
@st.cache_resource
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

def plot_feature_importances(model, feature_names):
    """Plot horizontal bar chart of feature importances"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)
        
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importances[indices], color='#1f77b4', align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        st.pyplot(plt)
    else:
        st.warning("Feature importances not available for this model type")

def main():
    st.set_page_config(page_title="Student Performance Predictor", layout="centered")
    st.title("🎓 Student Performance Predictor")
    st.markdown("Fill in the student data below to predict their performance category (Passing, Average, or Low).")
    
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    
    # Define feature names for visualizations
    feature_names = [
        'Age', 'Gender', 'Ethnicity', 'ParentalEducation', 'StudyTimeWeekly',
        'Absences', 'Tutoring', 'ParentalSupport', 'Extracurricular',
        'Sports', 'Music', 'Volunteering'
    ]

    # Input form in single column
    with st.form("prediction_form"):
        # All form elements in single column
        age = st.number_input("Age", min_value=10, max_value=30, value=18) # You can manually improve this range
        gender = st.selectbox("Gender", options=["Male", "Female"])
        parental_education = st.selectbox("Parental Education Level", 
                                         ["High School", "Associate", "Bachelor", "Master", "PhD"])
        study_time = st.slider("Weekly Study Time (hours)", 0, 40, 10)
        absences = st.slider("Number of Absences", 0, 50, 5)
        tutoring = st.selectbox("Attends Tutoring?", ["Yes", "No"])
        parental_support = st.selectbox("Parental Support Level", 
                                       ["Low", "Moderate", "High", "Very High", "Excellent"])
        extracurricular = st.selectbox("In Extracurricular Activities?", ["Yes", "No"])
        sports = st.slider("Weekly Hours in Sports", 0, 10, 2)
        music = st.slider("Weekly Hours in Music", 0, 10, 2)
        volunteering = st.slider("Weekly Hours Volunteering", 0, 10, 1)

        submitted = st.form_submit_button("Predict Performance")

    if submitted:
        # Process form inputs
        gender_encoded = 1 if gender == "Male" else 0
        ethnicity_encoded = 0  # Default value for all entries
        education_encoded = {"High School": 0, "Associate": 1, "Bachelor": 2, "Master": 3, "PhD": 4}[parental_education]
        tutoring_encoded = 1 if tutoring == "Yes" else 0
        support_encoded = {"Low": 0, "Moderate": 1, "High": 2, "Very High": 3, "Excellent": 4}[parental_support]
        extracurricular_encoded = 1 if extracurricular == "Yes" else 0

        input_data = pd.DataFrame([[
            age, gender_encoded, ethnicity_encoded, education_encoded, study_time,
            absences, tutoring_encoded, support_encoded, extracurricular_encoded,
            sports, music, volunteering
        ]], columns=feature_names)

        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]
        
        # Store results in session state
        st.session_state.prediction = prediction
        st.session_state.input_data = {
            'age': age,
            'study_time': study_time,
            'absences': absences,
            'tutoring_encoded': tutoring_encoded,
            'support_encoded': support_encoded
        }
        st.session_state.submitted = True
        st.session_state.show_insights = False

    # Display prediction if available
    if st.session_state.submitted and st.session_state.prediction:
        # Display prediction result
        st.subheader("Prediction Result")
        if st.session_state.prediction == "Passing":
            st.success("✅ The student is likely to pass the semester.")
        elif st.session_state.prediction == "Average":
            st.warning("⚠️ The student is borderline. Recommend extra attention.")
        else:
            st.error("❌ The student is at high risk of failing. Immediate intervention advised.")
        
        # Button to toggle insights
        if st.button("Show Detailed Insights", key="insights_toggle"):
            st.session_state.show_insights = not st.session_state.show_insights
        
        # Show insights if toggled
        if st.session_state.show_insights:
            # Visualizations section
            st.divider()
            st.subheader("Model Insights")
            
            # Feature importances
            st.markdown("### 📊 Feature Importances")
            st.markdown("Which factors most influence student performance predictions:")
            plot_feature_importances(model, feature_names)
            
            # Interpretation guide
            st.markdown("""
            **How to interpret this chart:**
            - Features at the top have the strongest impact on predictions
            - Longer bars indicate greater influence on the outcome
            - Features at the bottom have minimal impact on predictions
            """)
            
            # Key factors distribution
            st.divider()
            st.markdown("### 🔑 Key Factors Distribution")
            st.markdown("How different factors typically affect student performance:")
            
            # Create columns for distribution charts
            col1, col2, col3 = st.columns(3)
            with col1:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(x=[0, 5, 10, 15, 20, 25, 30, 35, 40], bins=8, color='#2ca02c', ax=ax)
                ax.set_title('Study Time Distribution')
                ax.set_xlabel('Hours per Week')
                st.pyplot(fig)
                
            with col2:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(x=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50], bins=10, color='#d62728', ax=ax)
                ax.set_title('Absences Impact')
                ax.set_xlabel('Number of Absences')
                st.pyplot(fig)
                
            with col3:
                fig, ax = plt.subplots(figsize=(6, 4))
                support_levels = ["Low", "Moderate", "High", "Very High", "Excellent"]
                sns.histplot(x=support_levels, color='#9467bd', ax=ax)
                ax.set_title('Parental Support')
                ax.set_xlabel('Support Level')
                plt.xticks(rotation=45)
                st.pyplot(fig)
            
            # Actionable recommendations
            st.divider()
            st.markdown("### 💡 Performance Improvement Tips")
            
            if st.session_state.prediction != "Passing":
                tips = []
                input_data = st.session_state.input_data
                
                if input_data['study_time'] < 15:
                    tips.append("⏱️ **Increase study time**: Aim for at least 15 hours/week")
                if input_data['absences'] > 10:
                    tips.append("📝 **Reduce absences**: Every absence reduces learning opportunities")
                if input_data['tutoring_encoded'] == 0:
                    tips.append("👨‍🏫 **Consider tutoring**: Targeted help addresses knowledge gaps")
                if input_data['support_encoded'] < 2:
                    tips.append("👪 **Enhance parental support**: Family engagement improves motivation")
                
                if tips:
                    st.info("Based on this student's profile, consider these interventions:")
                    for tip in tips:
                        st.markdown(f"- {tip}")
                else:
                    st.info("The student is doing well in key areas. Focus on maintaining good habits.")
            else:
                st.success("This student shows strong performance indicators. Focus on maintaining these positive habits!")

if __name__ == "__main__":
    main()