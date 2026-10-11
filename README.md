# Heart Disease Prediction System

Part of **Assignment 2 - Task 2: Developing an End-to-End Machine Learning System** (AIC354, COMSATS University Islamabad).

Predicts the presence of heart disease from **4 input attributes** (`cp`, `thalach`, `exang`, `oldpeak`) using a Support Vector Classifier, following the exact same 10-step teaching structure as the Titanic Passenger Survival Prediction notebook: Import -> Load -> Understand -> Preprocess -> Feature Extraction -> Label Encoding -> Train -> Test -> Application -> Feedback -> Deploy.

- **Dataset:** [Heart Disease - UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease) (Cleveland database, 303 patients)
- **Model:** `SVC(gamma='auto')` on StandardScaler-normalized inputs, stratified 80/20 split
- **Accuracy:** 0.80 on the held-out test set

## Files

| File | Purpose |
|---|---|
| `Heart_Disease_Prediction.ipynb` | The complete end-to-end notebook (live deployment link at the top) |
| `heart-disease.csv` | Dataset downloaded from UCI (303 rows, 14 attributes) |
| `heart_svc_trained_model.pkl` | Trained model file |
| `standard_scaler.pkl` | Fitted StandardScaler (used by notebook and web app) |
| `heart-app/` | Streamlit web app deployed on Streamlit Community Cloud |

## Live Demo

**(deployment link inserted at the top of the notebook)**

## Run the Notebook Locally

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
python -m ipykernel install --user --name=heart-venv
jupyter notebook
```

## Run the Web App Locally

```bash
cd heart-app
streamlit run app.py
```
