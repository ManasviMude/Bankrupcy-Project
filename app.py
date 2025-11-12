import streamlit as st
import pandas as pd
import pickle

# ---------------------------
# Load trained Logistic Regression model
# ---------------------------
with open('final_logreg_model.pkl', 'rb') as f:
    model = pickle.load(f)

# ---------------------------
# Streamlit Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Bankruptcy Prediction App",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------------------
# Custom CSS for a Clean Light UI
# ---------------------------
st.markdown("""
    <style>
        body {
            background-color: #f5f7fa;
            color: #2c2c2c;
            font-family: 'Segoe UI', sans-serif;
        }

        .stApp {
            background-color: #ffffff;
            padding: 2.5rem;
            border-radius: 1.5rem;
            box-shadow: 0px 6px 18px rgba(0,0,0,0.05);
        }

        .main-title {
            text-align: center;
            color: #002855;
            font-weight: 800;
            font-size: 2.2rem;
            background: linear-gradient(90deg, #003366, #0077b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.3rem;
        }

        .subtitle {
            text-align: center;
            color: #444;
            font-size: 1.05rem;
            margin-bottom: 2rem;
        }

        .section-box {
            background-color: #f0f4f8;
            border-radius: 12px;
            padding: 1.5rem 1.5rem 0.5rem 1.5rem;
            margin-bottom: 2rem;
            box-shadow: inset 0 0 8px rgba(0,0,0,0.03);
        }

        .section-header {
            color: #003366;
            font-size: 1.3rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }

        .stSelectbox label {
            font-weight: 600;
            color: #2c3e50;
        }

        .stButton button {
            background: linear-gradient(to right, #4A90E2, #50E3C2);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 1.3rem;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease-in-out;
        }

        .stButton button:hover {
            transform: scale(1.05);
            background: linear-gradient(to right, #50E3C2, #4A90E2);
        }

        .footer {
            text-align: center;
            color: #888;
            margin-top: 2.5rem;
            font-size: 0.9rem;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Header Section
# ---------------------------
st.markdown("<h1 class='main-title'>🏦 Bankruptcy Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Predict a company's financial health using Logistic Regression</p>", unsafe_allow_html=True)

# ---------------------------
# Input Section (with 0, 0.5, 1 options)
# ---------------------------
st.markdown("<div class='section-box'>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>📊 Enter Company Financial Indicators</div>", unsafe_allow_html=True)

options = [0.0, 0.5, 1.0]

col1, col2 = st.columns(2)
with col1:
    industrial_risk = st.selectbox("Industrial Risk", options, index=1)
    management_risk = st.selectbox("Management Risk", options, index=1)
    financial_flexibility = st.selectbox("Financial Flexibility", options, index=1)

with col2:
    credibility = st.selectbox("Credibility", options, index=1)
    competitiveness = st.selectbox("Competitiveness", options, index=1)
    operating_risk = st.selectbox("Operating Risk", options, index=1)

st.markdown("</div>", unsafe_allow_html=True)  # close section box

# ---------------------------
# Prepare DataFrame for Prediction
# ---------------------------
data = pd.DataFrame({
    'industrial_risk': [industrial_risk],
    'management_risk': [management_risk],
    'financial_flexibility': [financial_flexibility],
    'credibility': [credibility],
    'competitiveness': [competitiveness],
    'operating_risk': [operating_risk]
})

# ---------------------------
# Prediction Button
# ---------------------------
st.markdown("---")
if st.button("🔍 Predict Bankruptcy"):
    prediction = model.predict(data)[0]
    st.markdown("---")

    if prediction == 0:
        st.error("⚠️ **Result:** The company is predicted to be at risk of **Bankruptcy**.")
        st.markdown(
            "<div style='background-color:#ffe6e6;padding:12px;border-left:6px solid #ff4d4d;border-radius:8px;margin-top:10px;'>"
            "<b>💡 Suggestion:</b> Improve liquidity, reduce operational risk, and enhance management efficiency."
            "</div>",
            unsafe_allow_html=True
        )
    else:
        st.success("✅ **Result:** The company is predicted to be **Financially Healthy**.")
        st.markdown(
            "<div style='background-color:#e6ffed;padding:12px;border-left:6px solid #00cc66;border-radius:8px;margin-top:10px;'>"
            "<b>💡 Suggestion:</b> Maintain strong financial flexibility and competitiveness to ensure long-term stability."
            "</div>",
            unsafe_allow_html=True
        )

# ---------------------------
# Footer
# ---------------------------
st.markdown("<p class='footer'></p>", unsafe_allow_html=True)
