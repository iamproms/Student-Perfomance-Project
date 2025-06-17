import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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

def plot_feature_importances_plotly(model, feature_names):
    """Plot interactive feature importance chart with Plotly"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Feature': [feature_names[i] for i in sorted_idx],
            'Importance': importances[sorted_idx]
        })
        
        # Create the plot
        fig = px.bar(df, 
                     x='Importance', 
                     y='Feature', 
                     orientation='h',
                     color='Importance',
                     color_continuous_scale='Bluered',
                     title='<b>Feature Importances</b>',
                     labels={'Importance': 'Relative Importance'},
                     height=500)
        
        fig.update_layout(
            yaxis_title="",
            xaxis_title="Relative Importance",
            title_font_size=20,
            hovermode='y',
            template='plotly_white'
        )
        
        # Add custom hover text
        fig.update_traces(
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Feature importances not available for this model type")

def performance_gauge(prediction):
    """Create a performance gauge chart"""
    if prediction == "Passing":
        value = 90
        color = "green"
    elif prediction == "Average":
        value = 50
        color = "orange"
    else:
        value = 10
        color = "red"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Performance Potential"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps' : [
                {'range': [0, 40], 'color': "red"},
                {'range': [40, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "green"}]
        }
    ))
    
    fig.update_layout(height=300, margin=dict(t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

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
        age = st.number_input("Age", min_value=10, max_value=30, value=18)
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
        
        # Default values for Music and Volunteering
        music = 0  # Default value
        volunteering = 0  # Default value
        st.markdown("**Music & Volunteering**")
        st.info("Music and volunteering hours are set to 0 by default based on institutional data")

        submitted = st.form_submit_button("Predict Performance")

    if submitted:
        # Process form inputs
        gender_encoded = 1 if gender == "Male" else 0
        ethnicity_encoded = 0  # Default value for ethnicity
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
            'support_encoded': support_encoded,
            'sports': sports,
            'extracurricular': extracurricular,
            'parental_education': parental_education,
            'gender': gender
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
            st.subheader("📊 Model Insights")
            
            # Performance gauge
            st.markdown("### 📈 Performance Potential")
            performance_gauge(st.session_state.prediction)
            
            # Feature importances
            st.markdown("### 🔍 Feature Importances")
            st.markdown("Which factors most influence student performance predictions:")
            plot_feature_importances_plotly(model, feature_names)
            
            # Interpretation guide
            with st.expander("How to interpret this chart"):
                st.markdown("""
                - **Features at the top**: Have the strongest impact on predictions
                - **Longer bars**: Indicate greater influence on the outcome
                - **Color intensity**: Shows relative importance (darker = more important)
                - **Features at the bottom**: Have minimal impact on predictions
                """)
            
            # Student profile analysis
            st.divider()
            st.markdown("### 👤 Student Profile Analysis")
            
            input_data = st.session_state.input_data
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Weekly Study Time", f"{input_data['study_time']} hours", 
                          help="Recommended: 15-25 hours")
                st.metric("Absences", input_data['absences'], 
                          help="Recommended: <5 absences", delta_color="inverse")
                
            with col2:
                st.metric("Parental Support", input_data['parental_education'], 
                          help="Higher education levels correlate with better performance")
                st.metric("Extracurricular", "Active" if input_data['extracurricular'] == "Yes" else "Not Active", 
                          help="Students in activities show 23% better retention")
            
            # Risk factors
            st.divider()
            st.markdown("### ⚠️ Risk Factors")
            
            risk_factors = []
            if input_data['study_time'] < 10:
                risk_factors.append("Low study time")
            if input_data['absences'] > 10:
                risk_factors.append("High absences")
            if input_data['support_encoded'] < 2:
                risk_factors.append("Limited parental support")
            if input_data['tutoring_encoded'] == 0:
                risk_factors.append("No tutoring support")
                
            if risk_factors:
                for factor in risk_factors:
                    st.error(f"• {factor}")
            else:
                st.success("No major risk factors identified")
            
            # Actionable recommendations
            st.divider()
            st.markdown("### 💡 Performance Improvement Plan")
            
            tips = []
            if input_data['study_time'] < 15:
                tips.append(("⏱️ Increase study time", "Aim for 15-20 hours/week. Break into 45-min sessions"))
            if input_data['absences'] > 10:
                tips.append(("📝 Reduce absences", "Every absence reduces learning opportunities by 3-5%"))
            if input_data['tutoring_encoded'] == 0:
                tips.append(("👨‍🏫 Consider tutoring", "Targeted help can improve grades by 1-2 letter grades"))
            if input_data['support_encoded'] < 2:
                tips.append(("👪 Enhance parental support", "Family engagement improves motivation and accountability"))
            if input_data['sports'] > 15:
                tips.append(("⚖️ Balance sports commitment", "More than 15 hours/week can impact academic focus"))
                
            if tips:
                st.info("Based on this student's profile, we recommend:")
                for tip in tips:
                    st.markdown(f"#### {tip[0]}")
                    st.caption(tip[1])
            else:
                st.success("This student shows strong performance indicators. Focus on maintaining these positive habits!")
                
                # Positive reinforcement
                st.markdown("### 🌟 Strengths to Maintain")
                strengths = []
                if input_data['study_time'] >= 15:
                    strengths.append("Excellent study habits")
                if input_data['absences'] <= 5:
                    strengths.append("Strong attendance record")
                if input_data['support_encoded'] >= 3:
                    strengths.append("Excellent support system")
                if input_data['tutoring_encoded'] == 1:
                    strengths.append("Proactive academic support")
                    
                for strength in strengths:
                    st.success(f"• {strength}")

if __name__ == "__main__":
    main()