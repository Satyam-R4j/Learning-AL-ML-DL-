import joblib
import pandas as pd
import streamlit as st

model = joblib.load("heart_disease_model.pkl")

st.title("❤️ Heart Disease Prediction System")
st.caption("Machine Learning based clinical decision support tool")
st.warning("This tool is for educational purposes only and not a medical diagnosis.")


age = st.number_input("Age", min_value=1, max_value=120, value=45)
sex = st.selectbox("sex (0 = Female, 1= Male)", [0, 1])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("oldpeak", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of ST Segment", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
thal = st.selectbox("Thal", [0, 1, 2])


input_data = pd.DataFrame(
    [
        {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
        }
    ]
)

if st.button("Predict"):
    prediction = model.predict(input_data)
    risk_prob = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠️ High risk of Heart Disease\n\nRisk Probability: {risk_prob:.2%}")
    else:
        st.success(f"✅ Low risk of Heart Disease\n\nRisk Probability: {risk_prob:.2%}")

st.progress(risk_prob)
