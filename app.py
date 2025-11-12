import streamlit as st
import pandas as pd
import pickle

# ---------------------------
# Load trained Logistic Regression model
# ---------------------------
with open('final_logreg_model.pkl', 'rb') as f:
    model = pickle.load(f)

# ---------------------------
# Streamlit App UI
# ---------------------------
st.set_page_config(page_title="Bankruptcy Prediction App", layout="centered")
st.title("🏦 Bankruptcy Prediction App (Logistic Regression)")
st.markdown("""
Predict whether a company is likely to go **Bankrupt** or remain **Financially Healthy**  
based on its financial indicators.
""")

# Input fields
st.subheader("📊 Enter Company Financial Indicators")
industrial_risk = st.number_input("Industrial Risk", 0.0, 1.0, 0.5)
management_risk = st.number_input("Management Risk", 0.0, 1.0, 0.5)
financial_flexibility = st.number_input("Financial Flexibility", 0.0, 1.0, 0.5)
credibility = st.number_input("Credibility", 0.0, 1.0, 0.5)
competitiveness = st.number_input("Competitiveness", 0.0, 1.0, 0.5)
operating_risk = st.number_input("Operating Risk", 0.0, 1.0, 0.5)

# Prepare dataframe for prediction
data = pd.DataFrame({
    'industrial_risk': [industrial_risk],
    'management_risk': [management_risk],
    'financial_flexibility': [financial_flexibility],
    'credibility': [credibility],
    'competitiveness': [competitiveness],
    'operating_risk': [operating_risk]
})

# Prediction
if st.button("🔍 Predict Bankruptcy"):
    prediction = model.predict(data)[0]
    if prediction == 0:
        st.error("⚠️ The company is predicted to be at risk of **Bankruptcy**.")
    else:
        st.success("✅ The company is predicted to be **Financially Healthy**.")
        
# Optional batch prediction via CSV
st.markdown("---")
st.subheader("📂 Upload CSV for Batch Prediction (Optional)")
uploaded_file = st.file_uploader("Upload a CSV file with company financial data", type=["csv"])

if uploaded_file:
    batch_data = pd.read_csv(uploaded_file)
    preds = model.predict(batch_data)
    batch_data['Prediction'] = ['Bankruptcy' if p == 0 else 'Non-Bankruptcy' for p in preds]
    st.write("✅ Predictions complete:")
    st.dataframe(batch_data)

st.markdown("---")
st.caption("Developed with ❤️ using Streamlit and Scikit-learn")
