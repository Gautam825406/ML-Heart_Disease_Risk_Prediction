import streamlit as st
import pandas as pd
import joblib

# Page Config
st.set_page_config(page_title="Heart Stroke Prediction", page_icon="❤️", layout="wide")

# Load saved model, scaler, and expected columns
model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

# --- Custom CSS ---
st.markdown("""
    <style>
        html, body, .main {
            background-color: #f9f9f9;
            font-family: 'Segoe UI', sans-serif;
        }

        .title {
            text-align: center;
            font-size: 42px;
            font-weight: 700;
            color: #d62828;
            margin-bottom: 0;
        }

        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #6c757d;
            margin-top: 5px;
            margin-bottom: 30px;
        }

        .stButton > button {
            background-color: #0077b6;
            color: white;
            padding: 0.75em 1.5em;
            border-radius: 8px;
            font-size: 16px;
        }

        .stButton > button:hover {
            background-color: #023e8a;
        }

        .st-expanderHeader {
            font-size: 16px;
        }

        .stMarkdown {
            font-size: 15px;
        }

        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- Title Section ---
st.markdown('<div class="title">Heart Stroke Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Developed by Gautam Keshri • Know your heart risk in seconds</div>', unsafe_allow_html=True)

# --- Input Form ---
with st.form("heart_form"):
    st.subheader("Enter Your Health Parameters")
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 100, 40)
        sex = st.selectbox("Sex", ["M", "F"])
        chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
        fasting_bs = st.radio("Fasting Blood Sugar > 120 mg/dL", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    with col2:
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        max_hr = st.slider("Max Heart Rate", 60, 220, 150)
        exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
        oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    submit = st.form_submit_button("Predict")

# --- Prediction Output ---
if submit:
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("High Risk of Heart Disease. Please consult a cardiologist as soon as possible.")
    else:
        st.success("Low Risk of Heart Disease. Keep up the good lifestyle habits!")

    st.info("This prediction is based on a machine learning model and should not replace professional medical advice.")

# --- Expandable Tips ---
with st.expander("Health Tips to Prevent Heart Stroke"):
    st.markdown("""
    - Eat a heart-healthy diet (low sodium, more vegetables & fiber)
    - Avoid tobacco and limit alcohol
    - Maintain regular physical activity (30+ min/day)
    - Control blood pressure, cholesterol, and blood sugar
    - Get enough sleep and manage stress
    - Schedule regular health checkups
    """)

# --- Footer ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<center><small>Developed by Gautam Keshri • Powered by Machine Learning</small></center>",
    unsafe_allow_html=True
)
