import streamlit as st
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("dropout_risk_xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page config
st.set_page_config(page_title="Dropout Risk Prediction", layout="centered")

# Title
st.title("🎓 Early Risk & Dropout Prediction System")

st.write("""
This application predicts **student dropout risk** using
machine learning based on academic and behavioral inputs.
""")

# Sidebar inputs
st.sidebar.header("Enter Student Details")

attendance = st.sidebar.slider("Attendance Percentage", 0, 100, 75)
units = st.sidebar.slider("Number of Enrolled Units", 1, 10, 5)
grade = st.sidebar.slider("Average Grade", 0.0, 10.0, 6.5)
engagement = st.sidebar.slider("Engagement Score", 1, 10, 5)
task_completion = st.sidebar.slider("Task Completion Rate", 0, 100, 70)

# Predict button
if st.sidebar.button("Predict Dropout Risk"):

    student_data = np.array([[attendance, units, grade, engagement, task_completion]])
    student_scaled = scaler.transform(student_data)

    prediction = model.predict(student_scaled)[0]
    probability = model.predict_proba(student_scaled)[0][1]

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Risk of Dropout")
    else:
        st.success("✅ Low Risk of Dropout")

    st.write(f"**Risk Probability:** {round(probability, 2)}")

st.markdown("---")
st.markdown("Developed using Explainable Machine Learning (XGBoost)")
