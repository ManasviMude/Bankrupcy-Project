import streamlit as st
import pandas as pd
import pickle
import io

# Optional: for PDF reading
from PyPDF2 import PdfReader

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
        
# -----------------------------------------------------------
# 📂 Upload File Section (CSV, XLSX, PDF)
# -----------------------------------------------------------
st.markdown("---")
st.subheader("📤 Upload File for Batch Prediction (CSV, XLSX, or PDF)")

uploaded_file = st.file_uploader("Upload a file with company financial data", type=["csv", "xlsx", "pdf"])

if uploaded_file:
    file_name = uploaded_file.name.lower()
    
    # Handle CSV
    if file_name.endswith('.csv'):
        batch_data = pd.read_csv(uploaded_file)
        st.info("✅ CSV file successfully loaded!")

    # Handle Excel
    elif file_name.endswith('.xlsx'):
        batch_data = pd.read_excel(uploaded_file)
        st.info("✅ Excel file successfully loaded!")

    # Handle PDF
    elif file_name.endswith('.pdf'):
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        st.text_area("📄 Extracted Text from PDF:", text[:2000])
        st.warning("⚠️ PDF files are treated as plain text for now. Please ensure data is structured before model prediction.")
        batch_data = None

    else:
        st.error("❌ Unsupported file type.")
        batch_data = None

    # If the file was a CSV or Excel, make predictions
    if batch_data is not None:
        try:
            preds = model.predict(batch_data)
            batch_data['Prediction'] = ['Bankruptcy' if p == 0 else 'Non-Bankruptcy' for p in preds]
            st.success("✅ Predictions complete:")
            st.dataframe(batch_data)

            # Download option
            csv_output = batch_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Predictions as CSV",
                data=csv_output,
                file_name="batch_predictions.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

st.markdown("---")
st.caption("Developed with ❤️ using Streamlit and Scikit-learn")
