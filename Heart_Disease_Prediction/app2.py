import joblib
import pandas as pd
import streamlit as st

# Load model
model = joblib.load("heart_disease_model.pkl")

# Page Config
st.set_page_config(
    page_title="❤️ Heart Disease Predictor",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Title and Info
st.markdown("<h1 style='text-align: center; color: crimson;'>❤️ Heart Disease Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Machine Learning based clinical decision support tool</p>", unsafe_allow_html=True)
st.warning("⚠️ This tool is for educational purposes only and not a medical diagnosis.")

# Input Form in Columns for Better UX
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=45)
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
        exang = st.selectbox("Exercise Induced Angina", [0, 1])
    
    with col2:
        sex = st.selectbox("Sex (0 = Female, 1= Male)", [0, 1])
        chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
        restecg = st.selectbox("Resting ECG", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
        oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1)
    
    with col3:
        slope = st.selectbox("Slope of ST Segment", [0, 1, 2])
        ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia", [0, 1, 2])
    
    submitted = st.form_submit_button("💓 Predict Risk")

# Prediction
if submitted:
    input_data = pd.DataFrame([{
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
        "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }])
    
    prediction = model.predict(input_data)[0]
    risk_prob = model.predict_proba(input_data)[0][1]
    
    st.markdown("---")
    if prediction == 1:
        st.error(f"⚠️ **High Risk of Heart Disease**\n\nProbability: {risk_prob:.2%}")
        st.progress(int(risk_prob * 100))
    else:
        st.success(f"✅ **Low Risk of Heart Disease**\n\nProbability: {risk_prob:.2%}")
        st.progress(int(risk_prob * 100), text="Confidence Level")
