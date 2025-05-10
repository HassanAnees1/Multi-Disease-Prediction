import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Multi-Disease Predictor",
    page_icon="🏥",
    layout="wide"
)

# Title and description
st.title("🏥 Multi-Disease Healthcare Predictor")
st.markdown("""
This application provides predictions for multiple diseases using both classical machine learning and deep learning models.
Upload your patient data or use the input form to get predictions.
""")

# Disease selection
DISEASES = {
    'Diabetes': {
        'ml_model': 'models/diabetes/diabetes.pkl',
        'dl_model': 'models/diabetes_dl_model.h5',
        'scaler': 'models/diabetes/scaler.pkl',
        'features': [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
    },
    'Heart Disease': {
        'ml_model': 'models/heart_best_model.pkl',
        'dl_model': 'models/heart_dl_model.h5',
        'scaler': 'models/heart_scaler.pkl',
        'features': [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
    }
}

# Sidebar
st.sidebar.title("Settings")
disease = st.sidebar.selectbox("Select Disease", list(DISEASES.keys()))
model_type = st.sidebar.radio("Select Model Type", ["Classical ML", "Deep Learning"])

# Main content
st.header(f"{disease} Prediction")

# Input method selection
input_method = st.radio("Choose Input Method", ["Manual Input", "Upload CSV"])

if input_method == "Manual Input":
    # Create input form
    st.subheader("Patient Information")
    
    # Create two columns for input fields
    col1, col2 = st.columns(2)
    
    # Get feature names for selected disease
    features = DISEASES[disease]['features']
    
    # Create input fields
    input_data = {}
    for i, feature in enumerate(features):
        col = col1 if i < len(features)//2 else col2
        with col:
            input_data[feature] = st.number_input(
                feature,
                min_value=0.0,
                max_value=1000.0,
                value=0.0,
                step=0.1
            )
    
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])
    
else:
    # File uploader
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.write(input_df.head())
    else:
        st.warning("Please upload a CSV file or switch to manual input")
        input_df = None

# Prediction button
if st.button("Get Prediction") and input_df is not None:
    try:
        # Load models and scaler
        if model_type == "Classical ML":
            model = joblib.load(DISEASES[disease]['ml_model'])
            scaler = joblib.load(DISEASES[disease]['scaler'])
            
            # Scale input data
            input_scaled = scaler.transform(input_df)
            
            # Make prediction
            prediction = model.predict(input_scaled)
            probability = model.predict_proba(input_scaled)[:, 1]
            
        else:  # Deep Learning
            model = load_model(DISEASES[disease]['dl_model'])
            scaler = joblib.load(DISEASES[disease]['scaler'])
            
            # Scale input data
            input_scaled = scaler.transform(input_df)
            
            # Make prediction
            probability = model.predict(input_scaled).flatten()
            prediction = (probability > 0.5).astype(int)
        
        # Display results
        st.subheader("Prediction Results")
        
        # Create two columns for results
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Risk Level",
                "High Risk" if prediction[0] == 1 else "Low Risk",
                delta=None
            )
        
        with col2:
            st.metric(
                "Probability",
                f"{probability[0]:.2%}",
                delta=None
            )
        
        # Create probability gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability[0] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgray"},
                    {'range': [30, 70], 'color': "gray"},
                    {'range': [70, 100], 'color': "darkgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        
        fig.update_layout(
            title="Risk Probability Gauge",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display recommendations
        st.subheader("Recommendations")
        if prediction[0] == 1:
            st.warning("""
            Based on the prediction, this patient is at high risk. We recommend:
            - Schedule immediate follow-up with a healthcare provider
            - Review lifestyle factors and make necessary adjustments
            - Consider additional diagnostic tests
            - Monitor symptoms closely
            """)
        else:
            st.success("""
            Based on the prediction, this patient is at low risk. We recommend:
            - Continue regular health check-ups
            - Maintain healthy lifestyle habits
            - Monitor any changes in symptoms
            - Follow preventive care guidelines
            """)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Made with ❤️ for better healthcare</p>
</div>
""", unsafe_allow_html=True) 