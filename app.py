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
# Custom CSS for a beautiful light UI
# ---------------------------
st.markdown("""
    <style>
        body {
            background-color: #f7f9fc;
            color: #2c2c2c;
            font-family: 'Segoe UI', sans-serif;
        }
        .stApp {
            background-color: #ffffff;
            padding: 2.5rem;
            border-radius: 1.5rem;
            box-shadow: 0px 6px 18px rgba(0,0,0,0.05);
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
        .main-title {
            color: #003366;
            font-weight: 700;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            text-align: center;
            color: #4f4f4f;
            font-size: 1.05rem;
            margin-bottom: 2rem;
        }
        .stNumberInput label {
            font-weight: 600;
            color: #2c3e50;
        }
        .footer {
            text-align: center;
            color: #888;
            margin-top: 2rem;
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
# Input Section
# ---------------------------
st.markdown("### 📊 Enter Company Financial Indicators")

col1, col2 = st.columns(2)
with col1:
    industrial_risk = st.number_input("Industrial Risk", 0.0, 1.0, 0.5)
    management_risk = st.number_input("Management Risk", 0.0, 1.0, 0.5)
    financial_flexibility = st.number_input("Financial Flexibility", 0.0, 1.0, 0.5)

with col2:
    credibility = st.number_input("Credibility", 0.0, 1.0, 0.5)
    competitiveness = st.number_input("Competitiveness", 0.0, 1.0, 0.5)
    operating_risk = st.number_input("Operating Risk", 0.0, 1.0, 0.5)

# Prepare DataFrame for prediction
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
            "<div style='background-color:#ffe6e6;padding:10px;border-left:6px solid #ff4d4d;border-radius:8px;'>"
            "<b>Suggestion:</b> Review liquidity, reduce operational risk, and increase management efficiency."
            "</div>", unsafe_allow_html=True
        )
    else:
        st.success("✅ **Result:** The company is predicted to be **Financially Healthy**.")
        st.markdown(
            "<div style='background-color:#e6ffed;padding:10px;border-left:6px solid #00cc66;border-radius:8px;'>"
            "<b>Suggestion:</b> Maintain financial flexibility and competitiveness to stay stable."
            "</div>", unsafe_allow_html=True
        )

# ---------------------------
# Footer
# ---------------------------
st.markdown("<p class='footer'>Developed with ❤️ using Streamlit and Scikit-learn | Designed by ChatGPT ✨</p>", unsafe_allow_html=True)
