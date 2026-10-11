"""
Heart Disease Prediction - Assignment 2, Task 2 Deployment
----------------------------------------------------------
A small Streamlit web app that serves the trained Support Vector Classifier
from the "Heart_Disease_Prediction.ipynb" notebook as a live cloud web app,
so anyone can use the model from a browser.

It loads heart_svc_trained_model.pkl and standard_scaler.pkl produced by that
notebook, and recreates the exact same LabelEncoders (fit on the same fixed
category lists) used during training, so predictions match the notebook exactly.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

# ---------------------------------------------------------------------------
# Load the trained model and scaler (same files the notebook produced)
# ---------------------------------------------------------------------------

MODEL_PATH = Path(__file__).parent / "heart_svc_trained_model.pkl"
SCALER_PATH = Path(__file__).parent / "standard_scaler.pkl"

@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

@st.cache_resource
def build_label_encoders():
    """
    Recreates the exact same LabelEncoders used in the notebook.
    LabelEncoder assigns codes based on the sorted order of the category
    list, so fitting on the same fixed lists reproduces the same codes
    deterministically - no need to pickle/ship the encoders separately.
    """
    encoders = {}
    encoders["cp"] = LabelEncoder().fit(
        ["Asymptomatic", "Atypical Angina", "Non-Anginal Pain", "Typical Angina"]
    )
    encoders["exang"] = LabelEncoder().fit(["No", "Yes"])
    return encoders

model, scaler = load_artifacts()
encoders = build_label_encoders()

# ---------------------------------------------------------------------------
# Page content
# ---------------------------------------------------------------------------

st.title("❤️ Heart Disease Prediction")
st.markdown(
    """
    This app is part of **Assignment 2 - Task 2: Developing an End-to-End
    Machine Learning System** (AIC354). Enter a patient's details below and
    the trained Support Vector Classifier (SVC) will predict whether heart
    disease is present.

    *Model accuracy on held-out test data: **80%** (UCI Cleveland dataset, 303 patients).
    Uses 4 input attributes as required: cp, thalach, exang, oldpeak.*
    """
)

st.divider()

col1, col2 = st.columns(2)
with col1:
    cp = st.selectbox(
        "Chest Pain Type (cp)",
        ["Asymptomatic", "Atypical Angina", "Non-Anginal Pain", "Typical Angina"],
        index=0,
    )
    thalach = st.number_input(
        "Maximum Heart Rate Achieved (thalach)", min_value=71.0, max_value=202.0,
        value=150.0, step=1.0,
    )
with col2:
    exang = st.selectbox("Exercise Induced Angina (exang)", ["No", "Yes"])
    oldpeak = st.number_input(
        "ST Depression - Oldpeak", min_value=0.0, max_value=6.2, value=1.0, step=0.1,
    )

if st.button("Predict Heart Disease", type="primary", use_container_width=True):
    user_input = pd.DataFrame({
        "cp": [cp],
        "thalach": [thalach],
        "exang": [exang],
        "oldpeak": [oldpeak],
    })

    encoded = user_input.copy()
    encoded["cp"] = encoders["cp"].transform(user_input["cp"])
    encoded["exang"] = encoders["exang"].transform(user_input["exang"])

    # Same StandardScaler as the notebook (fit on training data only)
    scaled = scaler.transform(encoded)
    prediction = model.predict(scaled)[0]

    st.divider()
    if prediction == 1:
        st.error("### ❤️‍🩹 HEART DISEASE DETECTED")
    else:
        st.success("### ✅ NO HEART DISEASE")

    with st.expander("See the encoded + scaled feature vector sent to the model"):
        st.dataframe(encoded, hide_index=True)

st.divider()
st.caption(
    "Built for teaching purposes on the UCI Heart Disease (Cleveland) dataset - "
    "303 patients, 4 input attributes. Not a medical diagnostic tool; "
    "consult a physician for real medical decisions."
)
