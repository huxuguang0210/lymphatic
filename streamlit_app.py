import streamlit as st
import pandas as pd
import joblib
import os

# --------------------------------------------------
# Load trained SVM model
# --------------------------------------------------
MODEL_PATH = "svm_model.joblib"

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found: svm_model.joblib")
    st.stop()

model = joblib.load(MODEL_PATH)

# --------------------------------------------------
# Prediction functions
# --------------------------------------------------
def predict_single(data):
    """
    Predict probability of positive postoperative lymph node status
    """
    return model.predict_proba([data])[0, 1]


def predict_batch(df):
    """
    Predict probabilities for batch input
    """
    return model.predict_proba(df)[:, 1]


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
    This application predicts the **probability of positive postoperative lymph node status**
    based on pathological and imaging-related indicators.

    All input variables are **binary encoded according to predefined clinical thresholds**.
    This tool is intended **for research use only**.
    """
)

# --------------------------------------------------
# Sidebar input (ENCODING STRICTLY FIXED)
# --------------------------------------------------
st.sidebar.header("Input Variables")
st.sidebar.markdown("Select values according to clinical findings:")

vascular_invasion = st.sidebar.selectbox(
    "Vascular Invasion",
    options=[0, 1],
    format_func=lambda x: "Negative" if x == 0 else "Positive"
)

lymph_node_side = st.sidebar.selectbox(
    "Unilateral or Bilateral Lymph Nodes (Imaging)",
    options=[0, 1],
    format_func=lambda x: "Unilateral" if x == 0 else "Bilateral"
)

scc = st.sidebar.selectbox(
    "SCC Level",
    options=[0, 1],
    format_func=lambda x: "< 3.95" if x == 0 else "≥ 3.95"
)

max_lymph_node_diameter = st.sidebar.selectbox(
    "Maximum Lymph Node Diameter",
    options=[0, 1],
    format_func=lambda x: "< 1.45 cm" if x == 0 else "≥ 1.45 cm"
)

invasion_depth = st.sidebar.selectbox(
    "Depth of Invasion",
    options=[0, 1],
    format_func=lambda x: "< 1/2" if x == 0 else " ≥ 1/2"
)

# --------------------------------------------------
# Single prediction
# --------------------------------------------------
st.subheader("🔍 Single Case Prediction")

if st.button("Predict Lymph Node Status"):
    # IMPORTANT: feature order strictly follows training order
    input_data = [
        vascular_invasion,
        lymph_node_side,
        scc,
        max_lymph_node_diameter,
        invasion_depth
    ]

    probability = predict_single(input_data)

    st.markdown("### 🧪 Prediction Result")
    st.markdown(
        f"""
        <h2 style='color:#1f77b4; text-align:center;'>
        Probability of Positive Postoperative Lymph Node Status<br><br>
        <strong>{probability:.2f}</strong>
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
    Upload a CSV file with **five binary columns in the following order**:

    1. Vascular invasion (0=negative, 1=positive)  
    2. Unilateral or bilateral lymph nodes (0=unilateral, 1=bilateral)  
    3. SCC (0=<3.95, 1=≥3.95)  
    4. Maximum lymph node diameter (0=<1.45 cm, 1=≥1.45 cm)  
    5. Invasion depth (0=<1/2, 1=≥1/2)
    """
)

uploaded_file = st.file_uploader("Upload CSV File", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)

    probabilities = predict_batch(batch_data)
    batch_data["Predicted Status Probability"] = probabilities

    st.markdown("### 📊 Batch Prediction Results")
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
# Encoding reference (for clinical transparency)
# --------------------------------------------------
st.markdown(
    """
    ---
    ### 🔎 Variable Encoding Reference

    - **Vascular invasion**: 0 = negative, 1 = positive  
    - **Unilateral or bilateral lymph nodes**: 0 = unilateral, 1 = bilateral  
    - **SCC**: 0 = < 3.95, 1 = ≥ 3.95  
    - **Maximum lymph node diameter**: 0 = < 1.45 cm, 1 = ≥ 1.45 cm  
    - **Invasion depth**: 0 = < 1/2, 1 = ≥ 1/2  

    ---
    © Shengjing Hospital of China Medical University
    """,
    unsafe_allow_html=True
)
