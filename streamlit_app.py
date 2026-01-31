import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------------------------
# Load trained SVM model
# --------------------------------------------------
model = joblib.load("svm_model.joblib")

# --------------------------------------------------
# Prediction functions
# --------------------------------------------------
def predict_single(data):
    """
    Predict probability of positive postoperative lymph node status
    """
    probability = model.predict_proba([data])[0, 1]
    return probability


def predict_batch(df):
    """
    Predict probabilities for batch input
    """
    probabilities = model.predict_proba(df)[:, 1]
    return probabilities


# --------------------------------------------------
# Streamlit UI configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Postoperative Lymph Node Status Prediction",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Postoperative Lymph Node Status Prediction Tool")

st.markdown(
    """
    Welcome to the **Postoperative Lymph Node Status Prediction Tool**.  
    This application predicts the **probability of positive lymph node status after surgery**
    based on pathological and clinical features.

    You may either:
    - Enter values manually for **single-case prediction**, or  
    - Upload a **CSV file** for **batch prediction**.
    """
)

# --------------------------------------------------
# Sidebar input
# --------------------------------------------------
st.sidebar.header("Input Variables")
st.sidebar.markdown("Please select or enter the following parameters:")

vascular_invasion = st.sidebar.selectbox(
    "Vascular Invasion",
    options=[0, 1],
    format_func=lambda x: "Absent" if x == 0 else "Present"
)

lymph_node_side = st.sidebar.selectbox(
    "Unilateral or Bilateral Lymph Nodes",
    options=[0, 1],
    format_func=lambda x: "Unilateral" if x == 0 else "Bilateral"
)

scc = st.sidebar.number_input(
    "SCC (Squamous Cell Carcinoma Antigen)",
    min_value=0.0,
    step=0.1
)

max_lymph_node_diameter = st.sidebar.number_input(
    "Maximum Lymph Node Diameter (mm)",
    min_value=0.0,
    step=0.1
)

invasion_depth = st.sidebar.number_input(
    "Invasion Depth (mm)",
    min_value=0.0,
    step=0.1
)

# --------------------------------------------------
# Single prediction
# --------------------------------------------------
st.subheader("🔍 Single Case Prediction")

if st.button("Predict Lymph Node Status"):
    input_data = [
        vascular_invasion,
        lymph_node_side,
        scc,
        max_lymph_node_diameter,
        invasion_depth
    ]

    prob = predict_single(input_data)

    st.markdown("### 🧪 **Prediction Result**")
    st.markdown(
        f"""
        <h2 style='color:#1f77b4; text-align:center;'>
        Probability of Positive Postoperative Lymph Node Status: <br><br>
        <strong>{prob:.2f}</strong>
        </h2>
        """,
        unsafe_allow_html=True
    )

# --------------------------------------------------
# Batch prediction
# --------------------------------------------------
st.subheader("📁 Batch Prediction")

st.markdown(
    """
    Upload a CSV file containing the following columns **in the same order**:

    1. Vascular invasion  
    2. Unilateral or bilateral lymph nodes  
    3. SCC  
    4. Maximum lymph node diameter  
    5. Invasion depth
    """
)

uploaded_file = st.file_uploader("Upload CSV File", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)

    probabilities = predict_batch(batch_data)
    batch_data["Predicted Status Probability"] = probabilities

    st.markdown("### 📊 **Batch Prediction Results**")
    st.dataframe(
        batch_data.style.applymap(
            lambda x: "background-color: #FFF3CD" if x > 0.5 else "",
            subset=["Predicted Status Probability"]
        )
    )

    csv = batch_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Prediction Results",
        data=csv,
        file_name="lymph_node_status_predictions.csv",
        mime="text/csv"
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown(
    """
    ---
    © Shengjing Hospital of China Medical University
    """,
    unsafe_allow_html=True
)
