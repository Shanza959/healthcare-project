import streamlit as st
import numpy as np
import pickle
import os
import tensorflow as tf

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Healthcare Test Result Prediction",
    page_icon="🏥",
    layout="wide"
)

# ── Paths ────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(BASE, '..', 'Backend')

# ── Load Models ──────────────────────────────────────────────
@st.cache_resource
def load_models():
    ml_model = pickle.load(open(os.path.join(BACKEND, 'best_ml_model.pkl'), 'rb'))
    scaler   = pickle.load(open(os.path.join(BACKEND, 'scaler.pkl'),        'rb'))
    le       = pickle.load(open(os.path.join(BACKEND, 'label_encoder.pkl'), 'rb'))
    dl_model = tf.keras.models.load_model(os.path.join(BACKEND, 'best_dl_model.keras'))
    return ml_model, scaler, le, dl_model

ml_model, scaler, le, dl_model = load_models()
class_names = list(le.classes_)   # e.g. ['Abnormal', 'Inconclusive', 'Normal']

# ── Header ───────────────────────────────────────────────────
st.title("🏥 Healthcare Test Result Prediction")
st.markdown("Patient ki details fill karo — ML ya DL model se test result predict hoga.")
st.divider()

# ── Input Form ───────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    age            = st.number_input("Age",            min_value=1,    max_value=100,   value=35)
    billing_amount = st.number_input("Billing Amount", min_value=0.0,  max_value=100000.0, value=5000.0, step=100.0)

with col2:
    gender     = st.selectbox("Gender",     ["Male", "Female"])
    blood_type = st.selectbox("Blood Type", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
    admission  = st.selectbox("Admission Type", ["Emergency", "Elective", "Urgent"])

with col3:
    medical_condition = st.selectbox("Medical Condition",
                                     ["Diabetes", "Hypertension", "Asthma",
                                      "Heart Disease", "Obesity", "Arthritis"])
    insurance = st.selectbox("Insurance Provider",
                             ["Aetna", "Blue Cross", "Cigna", "Medicare", "UnitedHealthcare"])
    medication = st.selectbox("Medication",
                              ["Aspirin", "Ibuprofen", "Lisinopril", "Metformin",
                               "Paracetamol", "Penicillin"])

st.divider()

# Model select
model_type = st.radio("Model Select karo:", ["ML Model (Random Forest / GB)", "DL Model (LSTM / CNN)"],
                      horizontal=True)

# ── Encoding Maps ────────────────────────────────────────────
gender_map    = {"Male": 1, "Female": 0}
blood_map     = {"A+": 0, "A-": 1, "B+": 2, "B-": 3, "AB+": 4, "AB-": 5, "O+": 6, "O-": 7}
admit_map     = {"Emergency": 0, "Elective": 1, "Urgent": 2}
condition_map = {"Arthritis": 0, "Asthma": 1, "Diabetes": 2,
                 "Heart Disease": 3, "Hypertension": 4, "Obesity": 5}
insurance_map = {"Aetna": 0, "Blue Cross": 1, "Cigna": 2, "Medicare": 3, "UnitedHealthcare": 4}
med_map       = {"Aspirin": 0, "Ibuprofen": 1, "Lisinopril": 2,
                 "Metformin": 3, "Paracetamol": 4, "Penicillin": 5}

# ── Predict Button ───────────────────────────────────────────
if st.button("🔍 Predict Test Result", use_container_width=True):

    # Raw feature vector  (same order as X_final columns in notebook)
    raw = np.array([[
        age,
        gender_map[gender],
        blood_map[blood_type],
        condition_map[medical_condition],
        insurance_map[insurance],
        billing_amount,
        admit_map[admission],
        med_map[medication]
    ]])

    scaled = scaler.transform(raw)
    n_features = scaled.shape[1]

    if "ML" in model_type:
        proba      = ml_model.predict_proba(scaled)[0]
        pred_class = int(np.argmax(proba))
    else:
        dl_input   = scaled.reshape(1, n_features, 1).astype(np.float32)
        proba      = dl_model.predict(dl_input, verbose=0)[0]
        pred_class = int(np.argmax(proba))

    label = class_names[pred_class]
    conf  = float(proba[pred_class])

    st.divider()
    st.subheader("🔬 Prediction Result")

    r1, r2 = st.columns(2)
    with r1:
        if label == "Normal":
            st.success(f"✅ Test Result: **{label}**")
        elif label == "Abnormal":
            st.error(f"❌ Test Result: **{label}**")
        else:
            st.warning(f"⚠️ Test Result: **{label}**")

    with r2:
        st.metric("Confidence", f"{conf*100:.1f}%")
        st.progress(conf)

    # All class probabilities
    st.markdown("**Class-wise Probabilities:**")
    prob_cols = st.columns(len(class_names))
    for i, (cls, prob) in enumerate(zip(class_names, proba)):
        with prob_cols[i]:
            st.metric(cls, f"{prob*100:.1f}%")
